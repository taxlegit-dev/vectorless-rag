import os
import tempfile
from io import BytesIO
from typing import Optional

import pandas as pd
from docx import Document
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from pageindex import page_index, md_to_tree
from pageindex.utils import (
    create_clean_structure_for_description,
    generate_doc_description,
)

from api.db import insert_rag_document

router = APIRouter()


def _parse_domains(domains: Optional[str]) -> list[str]:
    if not domains:
        return []
    parts = [d.strip().upper() for d in domains.split(",") if d.strip()]
    return parts


def _parse_index_array(index_array: Optional[str]) -> list[str]:
    if not index_array:
        return []
    parts = [w.strip() for w in index_array.split(",") if w.strip()]
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


def _detect_subprocess_column(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """
    Detects S.no column and subprocess column from DataFrame headers.
    Returns (sno_col, subprocess_col) or (None, None) if not found.
    """
    sno_col = None
    subprocess_col = None

    for col in df.columns:
        col_str = str(col).lower().strip()

        # Detect S.no column
        if col_str in (
            "s.no.", "s no.", "sno", "sl.no.", "sl no",
            "s.no", "sno.", "s no", "sr.no.", "sr no",
            "serial no", "serial no.", "s. no.", "s. no",
        ):
            sno_col = col

        # Detect subprocess column
        if any(k in col_str for k in (
            "sub-process", "subprocess", "sub process",
            "process head", "process name", "sub-processes",
        )):
            subprocess_col = col

    return sno_col, subprocess_col


def _split_sheet_by_subprocess(
    df: pd.DataFrame,
    sno_col: str,
    subprocess_col: str,
) -> list[tuple[str, pd.DataFrame]]:
    """
    Splits a DataFrame into chunks grouped by subprocess.
    A new subprocess starts when S.no has a numeric value AND subprocess column is non-empty.
    Returns list of (subprocess_name, chunk_df).
    """
    chunks: list[tuple[str, pd.DataFrame]] = []
    current_name: Optional[str] = None
    current_rows: list[list] = []

    for _, row in df.iterrows():
        sno_val = str(row[sno_col]).strip()
        sp_val  = str(row[subprocess_col]).strip()

        # S.no has a digit and subprocess name is present → new subprocess block
        sno_is_number = sno_val not in ("", "nan") and any(c.isdigit() for c in sno_val)
        sp_is_valid   = sp_val not in ("", "nan")

        if sno_is_number and sp_is_valid:
            # Save previous block before starting new one
            if current_name and current_rows:
                chunk_df = pd.DataFrame(current_rows, columns=df.columns)
                chunks.append((current_name, chunk_df))
            # Start new block
            current_name = sp_val
            current_rows = [row.tolist()]
        else:
            # Continuation row (task/step under current subprocess)
            if current_name is not None:
                current_rows.append(row.tolist())

    # Save last block
    if current_name and current_rows:
        chunk_df = pd.DataFrame(current_rows, columns=df.columns)
        chunks.append((current_name, chunk_df))

    return chunks


def _find_header_row(xls: pd.ExcelFile, sheet_name: str) -> int:
    """
    Scans rows to find the one that contains both a S.no-like column and a
    subprocess-like column — this is the real header row when the sheet has
    preamble rows (company name, title, blank rows, etc.).
    Returns the 0-based row index to pass as `header=` to read_excel.
    """
    subprocess_keywords = (
        "sub-process", "subprocess", "sub process",
        "process head", "process name", "sub-processes",
    )
    sno_keywords = {
        "s.no.", "s no.", "sno", "sl.no.", "sl no",
        "s.no", "sno.", "s no", "sr.no.", "sr no",
        "serial no", "serial no.", "s. no.", "s. no",
    }

    df_raw = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl", header=None)

    for idx, row in df_raw.iterrows():
        cells = [str(v).lower().strip() for v in row]
        has_sno = any(c in sno_keywords for c in cells)
        has_subprocess = any(any(k in c for k in subprocess_keywords) for c in cells)
        if has_sno and has_subprocess:
            return int(idx)

    return 0  # fallback: first row is the header


def _markdown_from_xlsx(file_bytes: bytes, filename: str) -> str:
    xls = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")
    doc_title = os.path.splitext(filename)[0] or "Spreadsheet"
    lines = [f"# {doc_title}"]

    for sheet in xls.sheet_names:
        header_row = _find_header_row(xls, sheet)
        df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl", header=header_row)

        # Skip completely empty sheets
        if df.empty:
            continue

        lines.append(f"## Sheet: {sheet}")

        sno_col, subprocess_col = _detect_subprocess_column(df)

        if sno_col and subprocess_col:
            chunks = _split_sheet_by_subprocess(df, sno_col, subprocess_col)

            if chunks:
                # Successfully split by subprocess — create ### heading per subprocess
                for subprocess_name, chunk_df in chunks:
                    # Clean up the chunk: drop fully empty rows
                    chunk_df = chunk_df.dropna(how="all")
                    lines.append(f"### {subprocess_name}")
                    lines.append("```csv")
                    lines.append(chunk_df.to_csv(index=False).strip())
                    lines.append("```")
            else:
                # Subprocess column found but no chunks parsed — fallback to full sheet
                lines.append("```csv")
                lines.append(df.to_csv(index=False).strip())
                lines.append("```")
        else:
            # No subprocess structure detected — dump whole sheet as before
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
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".md", mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(markdown_text)
            tmp_path = tmp.name

        tree = await md_to_tree(
            md_path=tmp_path,
            if_thinning=False,
            min_token_threshold=100,       # lowered so subprocess nodes don't get merged
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


@router.post("/rag/documents")
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
            markdown_text = _wrap_as_markdown(
                os.path.splitext(file.filename)[0], content
            )
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