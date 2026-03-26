import re
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pageindex.utils import ChatGPT_API

from api.db import get_all_rag_documents_with_meta_by_domains, get_rag_document_tree
from api.retrieval import tree_search

router = APIRouter()


class QueryRequest(BaseModel):  # Request body structure define karta hai.
    document_id: Optional[str] = None
    question: str
    model: str = "gpt-4o-mini"
    max_hops: int = 6
    domains: Optional[list[str]] = None


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "by",
    "as",
    "at",
    "from",
    "this",
    "that",
    "these",
    "those",
    "process",
    "processes",
}

_ACRONYM_EXPANSIONS = {
    "p2p": "procurement to pay",
    "o2c": "order to cash",
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

_MAX_CONTEXT_CHARS = 20000
_EXTRACTION_TOP_K = 20
_EXTRACTION_PER_GROUP = 10
_EXTRACTION_QUERY_KEYWORDS = (
    "subprocess",
    "sub-process",
    "list",
    "all subprocess",
    "sub process names",
    "process head",
    "process heads",
    "process owner",
    "process owners",
    "process lead",
    "process leads",
)
_EXTRACTION_CONTENT_TERMS = (
    "subprocess",
    "sub-process",
    "sub process",
    "process head",
    "process heads",
    "process owner",
    "process owners",
    "process lead",
    "process leads",
)


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
    expansions = [_ACRONYM_EXPANSIONS[t] for t in tokens if t in _ACRONYM_EXPANSIONS]
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
    # Keep CSV content but drop the code fences.
    cleaned = re.sub(r"```csv\s*([\s\S]*?)```", r"\1", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > _MAX_CONTEXT_CHARS:
        cleaned = cleaned[:_MAX_CONTEXT_CHARS].rstrip()
    return cleaned


def _is_extraction_query(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in _EXTRACTION_QUERY_KEYWORDS)


def _is_extraction_content(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    for k in _EXTRACTION_CONTENT_TERMS:
        if k in t:
            return True
    return False


def _group_key_from_path(path: list[dict]) -> str:
    # Try to keep sheet-level diversity (e.g., "Sheet: Index")
    for node in path or []:
        title = (node.get("title") or "").strip()
        if title.lower().startswith("sheet:"):
            return title
    return "unknown"


def _get_children(node: dict) -> list[dict]:
    # "structure" is the top-level key in tree_json dicts returned by the DB
    for key in ("children", "nodes", "items", "sections", "structure"):
        val = node.get(key)
        if isinstance(val, list) and val:
            return val
    return []


def _collect_leaves(node: dict, path: Optional[list[dict]] = None) -> list[tuple[dict, list[dict]]]:
    if path is None:
        path = []
    if not isinstance(node, dict):
        return []
    children = _get_children(node)
    if not children:
        return [(node, path)]
    leaves: list[tuple[dict, list[dict]]] = []
    for child in children:
        leaves.extend(_collect_leaves(child, path + [node]))
    return leaves


def _score_leaf_for_extraction(question: str, leaf: dict, path: list[dict]) -> float:
    score = _score_leaf(question, leaf, path)
    content = _leaf_content(leaf).lower()
    title = (leaf.get("title") or "").lower()
    if _is_extraction_content(content) or _is_extraction_content(title):
        score += 0.25
    return score


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


@router.post("/rag/query")
def query_document(req: QueryRequest) -> dict:
    doc_id = req.document_id

    normalized_question = _normalize_question(req.question)
    question_text = normalized_question or req.question
    query_terms = set(_tokenize(question_text))
    is_extraction = _is_extraction_query(question_text)

    def _index_match_score(q_terms: set[str], index_array: Optional[list[str]]) -> float:
        if not q_terms or not index_array:
            return 0.0
        index_terms = _normalize_index_terms(index_array)
        if not index_terms:
            return 0.0
        overlap = len(q_terms & index_terms)
        return overlap / max(1, len(q_terms))

    if doc_id:
        tree_json = get_rag_document_tree(doc_id)
        if not tree_json:
            raise HTTPException(status_code=404, detail="Document not found")
        candidate_docs = [(doc_id, tree_json, [], None)]
    else:
        all_docs = get_all_rag_documents_with_meta_by_domains(req.domains)
        if not all_docs:
            raise HTTPException(
                status_code=404,
                detail="Document not found (provide document_id or domains)",
            )

        if query_terms:
            index_scored = []
            for doc in all_docs:
                score = _index_match_score(query_terms, doc[2])
                if score > 0:
                    index_scored.append((score, doc))
            if index_scored:
                index_scored.sort(key=lambda item: item[0], reverse=True)
                candidate_docs = [index_scored[0][1]]
            else:
                summary_matched = [doc for doc in all_docs if _summary_match(query_terms, doc[3])]
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

    best_leaf = None
    best_path = None
    best_doc_id = None
    best_score = float("-inf")

    best_overall_leaf = None
    best_overall_path = None
    best_overall_doc_id = None
    best_overall_score = float("-inf")

    best_multi_context = None
    best_multi_leaf = None
    best_multi_path = None
    best_multi_doc_id = None
    best_multi_score = float("-inf")

    for d_id, tree, _, _ in candidate_docs:
        try:
            if is_extraction:
                leaves = _collect_leaves(tree)
                scored: list[tuple[float, dict, list[dict], str]] = []
                for leaf, path in leaves:
                    content = _leaf_content(leaf)
                    if not content.strip():
                        continue
                    if not _is_extraction_content(content) and not _is_extraction_content(
                        (leaf.get("title") or "")
                    ):
                        continue
                    score = _score_leaf_for_extraction(question_text, leaf, path)
                    group_key = _group_key_from_path(path)
                    scored.append((score, leaf, path, group_key))

                if scored:
                    scored.sort(key=lambda item: item[0], reverse=True)
                    by_group: dict[str, list[tuple[float, dict, list[dict], str]]] = {}
                    for item in scored:
                        by_group.setdefault(item[3], []).append(item)

                    selected: list[tuple[float, dict, list[dict], str]] = []
                    for group_items in by_group.values():
                        selected.extend(group_items[:_EXTRACTION_PER_GROUP])

                    if len(selected) < _EXTRACTION_TOP_K:
                        remaining = [s for s in scored if s not in selected]
                        selected.extend(remaining[: _EXTRACTION_TOP_K - len(selected)])

                    top = selected[:_EXTRACTION_TOP_K]
                    # Sort shorter nodes first so concise Index nodes appear before
                    # large O2C detail nodes — this ensures all subprocess names
                    # fit within the context char limit before detail nodes consume it.
                    top_for_context = sorted(top, key=lambda item: len(_leaf_content(item[1])))
                    combined = "\n\n".join([_leaf_content(leaf) for _, leaf, _, _ in top_for_context])
                    combined = re.sub(r",,+", ",", combined)
                    print("TOTAL LEAVES:", len(leaves))
                    print("FILTERED LEAVES:", len(scored))
                    print("TOP SELECTED:", len(top))
                    print("CONTEXT LENGTH:", len(combined))
                    combined = _sanitize_context(combined)
                    doc_score = sum(s for s, _, _, _ in top)
                    if doc_score > best_multi_score:
                        best_multi_score = doc_score
                        best_multi_context = combined
                        best_multi_leaf = top[0][1]
                        best_multi_path = top[0][2]
                        best_multi_doc_id = d_id
                    continue

            leaf, path = tree_search(
                question_text,
                tree,
                model=req.model,
                max_hops=req.max_hops,
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

    if is_extraction and best_multi_context:
        return {
            "document_id": best_multi_doc_id,
            "path": [node.get("title") for node in (best_multi_path or [])],
            "node": {
                "title": best_multi_leaf.get("title") if best_multi_leaf else None,
                "node_id": best_multi_leaf.get("node_id") if best_multi_leaf else None,
            },
            "context": best_multi_context,
        }

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
