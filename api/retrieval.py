from typing import Any, Dict, List, Tuple

from pageindex.utils import ChatGPT_API, extract_json


def _format_options(children: List[Dict[str, Any]]) -> str:
    lines = []
    for i, child in enumerate(children):
        title = child.get("title", "")
        summary = child.get("summary") or child.get("prefix_summary") or ""
        if summary:
            lines.append(f"{i}. {title} - {summary}")
        else:
            lines.append(f"{i}. {title}")
    return "\n".join(lines)


def _pick_child_index(question: str, children: List[Dict[str, Any]], model: str) -> int:
    options_text = _format_options(children)
    prompt = f"""
You are selecting the single best section for answering a user query.
Return ONLY JSON in this format: {{"id": <number>}}.

User question: {question}

Sections:
{options_text}
"""

    response = ChatGPT_API(model=model, prompt=prompt)
    data = extract_json(response)
    idx = data.get("id") if isinstance(data, dict) else None

    if isinstance(idx, str) and idx.isdigit():
        idx = int(idx)
    if not isinstance(idx, int) or idx < 0 or idx >= len(children):
        idx = 0
    return idx


def _get_children(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    children = node.get("nodes")
    return children if isinstance(children, list) else []


def tree_search(question: str, tree_json: Dict[str, Any], model: str, max_hops: int = 6) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not isinstance(tree_json, dict):
        raise ValueError("tree_json must be a dict")

    nodes = tree_json.get("structure", tree_json)
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("tree_json has no structure nodes")

    path: List[Dict[str, Any]] = []
    current_nodes = nodes

    for _ in range(max_hops):
        idx = _pick_child_index(question, current_nodes, model)
        node = current_nodes[idx]
        path.append(node)
        children = _get_children(node)
        if not children:
            return node, path
        current_nodes = children

    return path[-1], path
