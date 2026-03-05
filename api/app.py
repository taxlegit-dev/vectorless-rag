import os
import tempfile
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from docx import Document

from pageindex import page_index, md_to_tree
from pageindex.utils import create_clean_structure_for_description, generate_doc_description
from api.db import (
    get_latest_rag_document_by_domains,
    get_rag_document_tree,
    init_db,
    insert_rag_document,
)
from api.retrieval import tree_search

app = FastAPI(title="PageIndex RAG API")

_cors_env = os.getenv("CORS_ORIGINS", "")
if _cors_env.strip():
    _origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
else:
    _origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup() -> None:
    init_db()


def _parse_domains(domains: Optional[str]) -> list[str]:
    if not domains:
        return []
    parts = [d.strip().upper() for d in domains.split(",") if d.strip()]
    return parts


def _wrap_as_markdown(title: str, content: str) -> str:
    clean_title = title.strip() or "Document"
    return f"# {clean_title}\n\n{content.strip()}\n"


def _markdown_from_docx(file_bytes: bytes, filename: str) -> str:
    doc = Document(BytesIO(file_bytes))
    lines = []
    has_heading = False

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = paragraph.style.name if paragraph.style else ""
        style_lower = style_name.lower()
        if style_lower.startswith("heading"):
            has_heading = True
            parts = style_name.split()
            level = 1
            if len(parts) > 1 and parts[-1].isdigit():
                level = max(1, min(6, int(parts[-1])))
            lines.append("#" * level + " " + text)
        else:
            lines.append(text)

    if not has_heading:
        title = os.path.splitext(filename)[0]
        lines.insert(0, "# " + (title or "Document"))

    return "\n\n".join(lines) + "\n"


def _markdown_from_xlsx(file_bytes: bytes, filename: str) -> str:
    xls = pd.ExcelFile(BytesIO(file_bytes))
    lines = [f"# {os.path.splitext(filename)[0] or 'Spreadsheet'}"]

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        lines.append(f"## Sheet: {sheet}")
        lines.append("```csv")
        lines.append(df.to_csv(index=False).strip())
        lines.append("```")

    return "\n\n".join(lines) + "\n"


def _markdown_from_csv(file_bytes: bytes, filename: str) -> str:
    df = pd.read_csv(BytesIO(file_bytes))
    title = os.path.splitext(filename)[0] or "CSV"
    return _wrap_as_markdown(title, "```csv\n" + df.to_csv(index=False).strip() + "\n```")


async def _tree_from_markdown(
    markdown_text: str,
    filename: str,
    model: str,
    if_add_node_text: str,
    if_add_node_summary: str,
    if_add_doc_description: str,
) -> dict:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as tmp:
            tmp.write(markdown_text)
            tmp_path = tmp.name

        tree = await md_to_tree(
            md_path=tmp_path,
            if_thinning=False,
            min_token_threshold=5000,
            if_add_node_summary=if_add_node_summary,
            summary_token_threshold=200,
            model=model,
            if_add_doc_description=if_add_doc_description,
            if_add_node_text=if_add_node_text,
            if_add_node_id="yes",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    tree["doc_name"] = os.path.splitext(filename)[0]
    return tree


@app.post("/rag/documents")
async def upload_document(
    file: UploadFile = File(...),
    uploaded_by_email: str = Form(...),
    domains: Optional[str] = Form(None),
    model: str = Form("gpt-4o-2024-11-20"),
    if_add_node_text: str = Form("yes"),
    if_add_node_summary: str = Form("no"),
    if_add_doc_description: str = Form("no"),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = os.path.splitext(file.filename)[1].lower()
    allowed = {".pdf", ".md", ".markdown", ".txt", ".docx", ".xlsx", ".csv"}
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use PDF, MD, TXT, DOCX, XLSX, or CSV.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    if ext == ".pdf":
        doc = BytesIO(file_bytes)
        tree = page_index(
            doc,
            model=model,
            if_add_node_text=if_add_node_text,
            if_add_node_summary=if_add_node_summary,
            if_add_doc_description="no",
            if_add_node_id="yes",
        )
    else:
        if ext in {".md", ".markdown"}:
            content = file_bytes.decode("utf-8", errors="replace")
            markdown_text = content
        elif ext == ".txt":
            content = file_bytes.decode("utf-8", errors="replace")
            markdown_text = _wrap_as_markdown(os.path.splitext(file.filename)[0], content)
        elif ext == ".docx":
            markdown_text = _markdown_from_docx(file_bytes, file.filename)
        elif ext == ".xlsx":
            markdown_text = _markdown_from_xlsx(file_bytes, file.filename)
        elif ext == ".csv":
            markdown_text = _markdown_from_csv(file_bytes, file.filename)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        tree = await _tree_from_markdown(
            markdown_text=markdown_text,
            filename=file.filename,
            model=model,
            if_add_node_text=if_add_node_text,
            if_add_node_summary=if_add_node_summary,
            if_add_doc_description="no",
        )

    if not isinstance(tree, dict):
        raise HTTPException(status_code=500, detail="Failed to generate document tree")

    tree.pop("doc_description", None)
    structure = tree.get("structure")
    if structure and if_add_node_summary.lower() == "yes":
        clean_structure = create_clean_structure_for_description(structure)
        doc_summary = generate_doc_description(clean_structure, model=model)
    else:
        doc_summary = None
    doc_id = insert_rag_document(
        source_file_name=file.filename,
        uploaded_by_email=uploaded_by_email,
        domains=_parse_domains(domains),
        tree_json=tree,
        doc_summary=doc_summary,
    )

    return {
        "document_id": doc_id,
        "doc_name": tree.get("doc_name"),
        "doc_summary": doc_summary,
    }


class QueryRequest(BaseModel):
    document_id: Optional[str] = None
    question: str
    model: str = "gpt-4o-2024-11-20"
    max_hops: int = 6
    domains: Optional[list[str]] = None


@app.post("/rag/query")
def query_document(req: QueryRequest) -> dict:
    doc_id = req.document_id
    tree_json = None

    if doc_id:
        tree_json = get_rag_document_tree(doc_id)
    else:
        result = get_latest_rag_document_by_domains(req.domains)
        if result:
            doc_id, tree_json = result

    if not tree_json:
        raise HTTPException(
            status_code=404,
            detail="Document not found (provide document_id or domains)",
        )

    leaf, path = tree_search(req.question, tree_json, model=req.model, max_hops=req.max_hops)

    return {
        "document_id": doc_id,
        "path": [node.get("title") for node in path],
        "node": {
            "title": leaf.get("title"),
            "node_id": leaf.get("node_id"),
            "start_index": leaf.get("start_index"),
            "end_index": leaf.get("end_index"),
        },
        "context": leaf.get("text", ""),
    }
