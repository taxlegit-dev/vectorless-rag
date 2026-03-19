import os
import re
import tempfile
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from docx import Document

from pageindex import page_index, md_to_tree #Ye PageIndex library hai jo document ko tree structure me convert karta hai.
from pageindex.utils import ChatGPT_API, create_clean_structure_for_description, generate_doc_description #Ye document summary generate karne ke liye functions hain.

from api.db import (
    get_all_rag_documents_with_meta_by_domains,
    get_rag_document_tree,
    init_db,
    insert_rag_document,
)
from api.retrieval import tree_search #👉 Tree me semantic search karne ke liye.

app = FastAPI(title="PageIndex RAG API") #API server create karta hai.

_cors_env = os.getenv("CORS_ORIGINS", "") #Environment variable se allowed domains read karta hai.
if _cors_env.strip(): #Agar .env me CORS defined hai → use karo.
    _origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
else:  #Default frontend URLs.
    _origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(   #CORS middleware add karta hai.
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")  #Server start hone par:
# 👉 database tables create ho jate hain
def _startup() -> None:
    init_db()


def _parse_domains(domains: Optional[str]) -> list[str]: #Domain Parse Function , Input example:ICFR,CSR,NGO ---> Output: ["ICFR","CSR","NGO"]
    if not domains:
        return []
    parts = [d.strip().upper() for d in domains.split(",") if d.strip()]
    return parts


def _parse_index_array(index_array: Optional[str]) -> list[str]:
    if not index_array:
        return []
    parts = [w.strip() for w in index_array.split(",") if w.strip()]
    return parts


def _wrap_as_markdown(title: str, content: str) -> str:   # Text ko markdown format me convert karta hai.
    clean_title = title.strip() or "Document"
    return f"# {clean_title}\n\n{content.strip()}\n"


def _markdown_from_docx(file_bytes: bytes, filename: str) -> str: # DOCX file ko markdown me convert karta hai.
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


def _markdown_from_xlsx(file_bytes: bytes, filename: str) -> str:   # Excel ko convert karta hai.
    xls = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")
    lines = [f"# {os.path.splitext(filename)[0] or 'Spreadsheet'}"]

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
        lines.append(f"## Sheet: {sheet}")
        lines.append("```csv")
        lines.append(df.to_csv(index=False).strip())
        lines.append("```")

    return "\n\n".join(lines) + "\n"


def _markdown_from_csv(file_bytes: bytes, filename: str) -> str:  # CSV ko markdown me convert karta hai.
    df = pd.read_csv(BytesIO(file_bytes))
    title = os.path.splitext(filename)[0] or "CSV"
    return _wrap_as_markdown(title, "```csv\n" + df.to_csv(index=False).strip() + "\n```")


async def _tree_from_markdown(   # Ye function markdown ko document tree me convert karta hai.
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
    index_array: Optional[str] = Form(None),
    model: str = Form("gpt-4o-2024-11-20"),
    if_add_node_text: str = Form("yes"),
    if_add_node_summary: str = Form("yes"),
    if_add_doc_description: str = Form("yes"),
) -> dict:
    if_add_node_summary = "yes"
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
    index_words = _parse_index_array(index_array)
    doc_id = insert_rag_document(
        source_file_name=file.filename,
        uploaded_by_email=uploaded_by_email,
        domains=_parse_domains(domains),
        index_array=index_words,
        summarization=True,
        tree_json=tree,
        doc_summary=doc_summary,
    )

    return {
        "document_id": doc_id,
        "doc_name": tree.get("doc_name"),
        "doc_summary": doc_summary,
        "index_array": index_words,
        "summarization": True,
    }


class QueryRequest(BaseModel):    # Request body structure define karta hai.
    document_id: Optional[str] = None
    question: str
    model: str = "gpt-4o-2024-11-20"
    max_hops: int = 6
    domains: Optional[list[str]] = None


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "as", "at", "from", "this",
    "that", "these", "those", "process", "processes",
}

_ACRONYM_EXPANSIONS = {
    "p2p": "procurement to pay",
    "o2c": "order to cash",
    "r2r": "record to report",
    "icfr": "internal control over financial reporting",
}

_GENERIC_TITLES = {
    "overview",
    "introduction",
    "summary",
    "table of contents",
    "toc",
    "contents",
}

_MAX_CONTEXT_CHARS = 800


def _normalize_question(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return ""

    # Remove short acronym-only parentheses like "(P2P)" to reduce mismatch noise.
    def _strip_acronym_parens(match: re.Match) -> str:
        inner = (match.group(1) or "").strip()
        if inner and len(inner) <= 10 and re.fullmatch(r"[A-Z0-9&/ -]+", inner):
            return ""
        return match.group(0)

    q = re.sub(r"\(([^)]{1,20})\)", _strip_acronym_parens, q)
    q = q.replace("/", " ").replace("-", " ")
    q = re.sub(r"\s+", " ", q).strip()

    # Expand common process acronyms when present.
    tokens = re.findall(r"[A-Za-z0-9]+", q.lower())
    expansions = [ _ACRONYM_EXPANSIONS[t] for t in tokens if t in _ACRONYM_EXPANSIONS ]
    if expansions:
        q = (q + " " + " ".join(expansions)).strip()
    return q


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _normalize_index_terms(index_array: Optional[list[str]]) -> set[str]:
    terms: set[str] = set()
    for raw in index_array or []:
        if not raw:
            continue
        value = raw.strip().lower()
        if not value:
            continue
        terms.add(value)
        tokens = re.findall(r"[A-Za-z0-9]+", value)
        if len(tokens) > 1:
            terms.update(tokens)
    return terms


def _index_match(query_terms: set[str], index_array: Optional[list[str]]) -> bool:
    if not query_terms or not index_array:
        return False
    index_terms = _normalize_index_terms(index_array)
    return bool(query_terms & index_terms)


def _summary_match(query_terms: set[str], doc_summary: Optional[str]) -> bool:
    if not query_terms or not doc_summary:
        return False
    summary_terms = set(_tokenize(doc_summary))
    return bool(query_terms & summary_terms)


def _leaf_content(leaf: dict) -> str:
    for key in ("text", "summary", "prefix_summary"):
        val = leaf.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _sanitize_context(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"```csv[\s\S]*?```", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > _MAX_CONTEXT_CHARS:
        cleaned = cleaned[:_MAX_CONTEXT_CHARS].rstrip()
    return cleaned


def _is_generic_title(title: str) -> bool:
    if not title:
        return False
    t = title.strip().lower()
    return t in _GENERIC_TITLES


def _score_leaf(question: str, leaf: dict, path: list[dict]) -> float:
    content = _leaf_content(leaf)
    title = leaf.get("title") or ""
    path_titles = " ".join([n.get("title", "") for n in (path or [])])
    combined = " ".join([title, path_titles, content])

    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return 0.0
    c_tokens = set(_tokenize(combined))
    overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))

    length_bonus = min(len(content) / 500.0, 1.0)
    title_boost = 0.1 if any(t in (title or "").lower() for t in q_tokens) else 0.0
    generic_penalty = 0.1 if _is_generic_title(title) else 0.0
    return overlap + length_bonus + title_boost - generic_penalty


def _is_low_quality(question: str, leaf: dict, path: list[dict]) -> bool:
    content = _leaf_content(leaf)
    if not content.strip():
        return True
    q_tokens = set(_tokenize(question))
    c_tokens = set(_tokenize(content))
    overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens)) if q_tokens else 0.0
    if len(content) < 40 and overlap < 0.15:
        return True
    return False


@app.post("/rag/query")
def query_document(req: QueryRequest) -> dict:
    doc_id = req.document_id

    normalized_question = _normalize_question(req.question)
    question_text = normalized_question or req.question
    query_terms = set(_tokenize(question_text))

    if doc_id:
        # Specific document ID diya hai
        tree_json = get_rag_document_tree(doc_id)
        if not tree_json:
            raise HTTPException(status_code=404, detail="Document not found")
        candidate_docs = [(doc_id, tree_json, [], None)]
    else:
        # Sabhi matching documents fetch karo (with meta for index/summary match)
        all_docs = get_all_rag_documents_with_meta_by_domains(req.domains)
        if not all_docs:
            raise HTTPException(
                status_code=404,
                detail="Document not found (provide document_id or domains)",
            )

        if query_terms:
            index_matched = [
                doc for doc in all_docs if _index_match(query_terms, doc[2])
            ]
            if index_matched:
                candidate_docs = index_matched
            else:
                summary_matched = [
                    doc for doc in all_docs if _summary_match(query_terms, doc[3])
                ]
                if summary_matched:
                    candidate_docs = summary_matched
                else:
                    fallback = ChatGPT_API(
                        model=req.model,
                        prompt=(
                            "Answer the user's question clearly and concisely.\n\n"
                            f"Question: {question_text}"
                        ),
                    )
                    return {
                        "document_id": None,
                        "path": [],
                        "node": {
                            "title": "fallback",
                            "node_id": None,
                        },
                        "context": fallback,
                    }
        else:
            candidate_docs = all_docs

    # Har document mein search karo, best context lo
    best_leaf = None
    best_path = None
    best_doc_id = None
    best_score = float("-inf")

    best_overall_leaf = None
    best_overall_path = None
    best_overall_doc_id = None
    best_overall_score = float("-inf")

    for d_id, tree, _, _ in candidate_docs:
        try:
            leaf, path = tree_search(
                question_text,
                tree,
                model=req.model, 
                max_hops=req.max_hops
            )
            score = _score_leaf(question_text, leaf, path)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_leaf = leaf
                best_overall_path = path
                best_overall_doc_id = d_id

            if not _is_low_quality(question_text, leaf, path):
                if score > best_score:
                    best_score = score
                    best_leaf = leaf
                    best_path = path
                    best_doc_id = d_id
        except Exception as e:
            print(f"tree_search failed for doc {d_id}: {e}")
            continue

    if best_leaf is None:
        if best_overall_leaf is None:
            raise HTTPException(status_code=500, detail="Search failed across all documents")
        best_leaf = best_overall_leaf
        best_path = best_overall_path
        best_doc_id = best_overall_doc_id

    return {
        "document_id": best_doc_id,
        "path": [node.get("title") for node in best_path],
        "node": {
            "title": best_leaf.get("title"),
            "node_id": best_leaf.get("node_id"),
        },
        "context": _sanitize_context(_leaf_content(best_leaf)),
    }
