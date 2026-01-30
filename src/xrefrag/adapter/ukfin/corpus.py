# xrefrag/adapter/ukfin/corpus.py
"""
UKFIN v1 Corpus Generator (PRA Rulebook HTML) — paragraph/list-item passages

raw/pra_rulebook/*.html -> processed/passage_corpus.jsonl

Key Fixes / Updates
-------------------
1) BAD ancestor filtering is TOKEN-BASED (prevents false positives like "navigation" matching "nav").
2) Main container selection has a robust fallback: choose the DIV with the highest (p+li) count.
3) Adds per-doc debug stats so "0 passages" cases are explainable.
4) NAV/TITLE-LIKE filtering extended for PRA:
   - "Legal Instruments that change this part/rule/chapter ..."
   - "Parts ...", "Policy Statements ...", etc.
5) Anchor alias support:
   - Emits "anchor_id" (canonical) and "anchor_ids" (aliases) for each passage.
   - Aliases are built by assigning preceding anchors (ids/names) to the next kept passage block.

Output Schema (JSONL)
---------------------
{
  "passage_uid": "...",
  "passage_id": "<doc_id>::<eId>",
  "doc_id": "<doc_id>",
  "eId": "<local_id>",
  "tag": "paragraph",
  "source_tag": "p" | "li",
  "title": "...",
  "heading_path": ["...", "..."],
  "passage": "...",
  "doc_url": "<canonical_url_no_fragment>",
  "passage_url": "<doc_url>#<eId>" | "",
  "anchor_id": "<eId>",
  "anchor_ids": ["<aliases...>"],
  "refs": [{"href": "...", "text": "..."}]
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

from bs4 import BeautifulSoup
from tqdm import tqdm

# -----------------------------
# Text / URL utilities
# -----------------------------
WS_RE = re.compile(r"\s+")
DOT_LEADER_RE = re.compile(r"\.{4,}")

# Minimal but strong boilerplate tokens (token-based ancestor filter)
BAD_TOKENS = {
    "cookie",
    "consent",
    "sidebar",
    "advert",
    "advertisement",
    "promo",
    "banner",
}

# PRA nav/title-like text patterns
LEGAL_INSTR_CHANGE_RE = re.compile(
    r"^legal instruments that change this (part|rule|chapter)\b", re.IGNORECASE
)
NAV_PREFIX_RE = re.compile(
    r"^(legal instruments that change this (part|rule|chapter)\b|parts?\b|policy statements?\b|annex(es)?\b|appendix\b)",
    re.IGNORECASE,
)


def clean_whitespace(text: str) -> str:
    if not text:
        return ""
    return WS_RE.sub(" ", text).strip()


def safe_read(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, encoding=enc, errors="strict") as f:
                return f.read()
        except Exception:
            continue
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_html(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def normalize_url(u: str) -> str:
    """
    Normalize for doc-level identity:
      - lower scheme+host
      - drop fragment
      - trim trailing slash (except root)
      - keep query
    """
    if not u:
        return ""
    u = u.strip()
    try:
        p = urlparse(u)
    except Exception:
        return u

    scheme = (p.scheme or "https").lower()
    netloc = (p.netloc or "").lower()
    path = p.path or ""
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    return urlunparse((scheme, netloc, path, p.params, p.query, ""))


def short_hash(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


def slugify_doc_id_from_url(url: str) -> str:
    nu = normalize_url(url)
    p = urlparse(nu)
    raw = (p.netloc + p.path).strip("/")
    raw = raw.replace("/", "_")
    raw = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_").lower()
    return f"pra_rulebook__{raw}" if raw else "pra_rulebook__unknown"


def doc_id_from_filename(path: str) -> str:
    base = os.path.basename(path)
    if base.lower().endswith(".html"):
        base = base[:-5]
    base = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_").lower()
    return f"pra_rulebook__{base}" if base else "pra_rulebook__unknown"


def make_passage_uid(doc_url: str, eId: str, fallback: str = "") -> str:
    key = (doc_url or fallback or "") + "#" + (eId or "")
    h = hashlib.blake2b(key.encode("utf-8", errors="ignore"), digest_size=8)
    return h.hexdigest()


# -----------------------------
# HTML heuristics
# -----------------------------
BLOCK_TAGS = {"p", "li"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

MAIN_CONTAINER_CANDIDATES: list[tuple[str, dict[str, Any]]] = [
    ("main", {}),
    ("article", {}),
    ("div", {"id": "content"}),
    ("div", {"id": "main"}),
    ("div", {"role": "main"}),
    ("div", {"class": "content"}),
    ("div", {"class": "main-content"}),
]


def extract_canonical_url(soup: BeautifulSoup) -> str | None:
    link = soup.find("link", attrs={"rel": lambda x: x and "canonical" in str(x).lower()})
    if link and link.get("href"):
        return link["href"].strip()

    meta = soup.find("meta", attrs={"property": "og:url"})
    if meta and meta.get("content"):
        return meta["content"].strip()

    return None


def extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.get_text(strip=True):
        return clean_whitespace(soup.title.get_text(" ", strip=True))
    h1 = soup.find("h1")
    if h1:
        return clean_whitespace(h1.get_text(" ", strip=True))
    return ""


def strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for name in ("header", "footer", "nav", "aside"):
        for t in soup.find_all(name):
            t.decompose()


def pick_main_container(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Choose a container likely holding the main text.

    Strategy:
    1) Try standard containers (main/article/content/main).
    2) Fallback: choose the DIV with the highest (p + li) count.
    3) Fallback: body.
    """
    strip_noise(soup)

    for name, attrs in MAIN_CONTAINER_CANDIDATES:
        found = soup.find(name, attrs=attrs)
        if found:
            return found

    body = soup.body if soup.body else soup

    best = None
    best_score = -1
    for div in body.find_all("div", recursive=True):
        score = len(div.find_all("p")) + len(div.find_all("li"))
        if score > best_score:
            best_score = score
            best = div

    if best is not None and best_score > 0:
        return best

    return body


def _tokens_from_attr_value(val: str) -> list[str]:
    if not val:
        return []
    return [t for t in re.split(r"[^a-zA-Z0-9]+", val.lower()) if t]


def has_bad_ancestor(el) -> bool:
    """
    Token-based bad-ancestor detection (avoids substring false positives).
    """
    cur = el
    while cur is not None and getattr(cur, "name", None) is not None:
        nm = (cur.name or "").lower()
        if nm in {"body", "html"}:
            break
        if nm in {"nav", "header", "footer", "aside"}:
            return True

        tokens = set()
        for c in cur.get("class", []) or []:
            tokens.update(_tokens_from_attr_value(str(c)))
        eid = cur.get("id", "") or ""
        tokens.update(_tokens_from_attr_value(str(eid)))

        if tokens & BAD_TOKENS:
            return True

        cur = cur.parent
    return False


def element_local_id(el) -> str | None:
    """
    Prefer block element id, else nested <a id|name>.
    """
    if el.has_attr("id") and el["id"]:
        return str(el["id"]).strip()

    a = el.find("a", attrs={"id": True})
    if a and a.get("id"):
        return str(a["id"]).strip()

    a = el.find("a", attrs={"name": True})
    if a and a.get("name"):
        return str(a["name"]).strip()

    return None


def extract_links(el) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for a in el.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        txt = clean_whitespace(a.get_text(" ", strip=True)) or href
        out.append({"href": href, "text": txt})
    return out


def is_nav_or_title_like(text: str) -> bool:
    t = clean_whitespace(text or "")
    if not t:
        return True
    # TOC / dot leaders
    if DOT_LEADER_RE.search(t):
        return True
    # Strong PRA nav patterns
    if LEGAL_INSTR_CHANGE_RE.match(t):
        return True
    if NAV_PREFIX_RE.match(t):
        return True
    return False


def is_meaningful_passage(text: str, min_chars: int, min_words: int) -> bool:
    if not text:
        return False
    if len(text) < min_chars:
        return False
    if len(text.split()) < min_words:
        return False
    if is_nav_or_title_like(text):
        return False
    return True


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class DocInfo:
    doc_id: str
    file_path: str
    doc_url: str  # canonical normalized (no fragment)
    title: str


# -----------------------------
# Corpus builder
# -----------------------------
def load_docs(raw_dir: str) -> list[DocInfo]:
    files = sorted(
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.lower().endswith(".html")
    )
    docs: list[DocInfo] = []
    for fp in tqdm(files, desc="Indexing HTML (doc metadata)"):
        html = safe_read(fp)
        soup = parse_html(html)
        can = extract_canonical_url(soup)
        title = extract_title(soup)

        if can:
            doc_url = normalize_url(can)
            doc_id = slugify_doc_id_from_url(doc_url)
        else:
            doc_url = ""
            doc_id = doc_id_from_filename(fp)

        docs.append(DocInfo(doc_id=doc_id, file_path=fp, doc_url=doc_url, title=title))
    return docs


def _collect_anchor_tokens(el) -> list[str]:
    """
    Collect anchor identifiers from an element:
      - el.id
      - <a id=...> and <a name=...> inside el
    """
    out: list[str] = []
    if getattr(el, "get", None):
        if el.get("id"):
            out.append(str(el.get("id")).strip())

    # explicit anchors
    for a in getattr(el, "find_all", lambda *args, **kwargs: [])("a"):
        if a.get("id"):
            out.append(str(a.get("id")).strip())
        if a.get("name"):
            out.append(str(a.get("name")).strip())

    # de-dup preserve order
    seen = set()
    uniq = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def build_passages_for_doc(
    doc: DocInfo,
    min_chars: int,
    min_words: int,
    debug: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    html = safe_read(doc.file_path)
    soup = parse_html(html)
    container = pick_main_container(soup)

    passages: list[dict[str, Any]] = []
    seen_local: set[str] = set()
    gen_counter = 0

    st = {
        "blocks_seen": 0,
        "blocks_bad_ancestor": 0,
        "blocks_too_short_or_nav": 0,
        "blocks_written": 0,
        "anchors_seen": 0,
        "anchors_assigned": 0,
    }

    # Anchor alias assignment:
    # Track anchors encountered since the previous kept passage, assign them to the next kept passage.
    pending_anchors: list[str] = []

    # Heading context (updated on the fly)
    heading_path: list[str] = []

    # Traverse in document order for headings + blocks + anchor carriers
    # We include headings to maintain context, and all elements to collect anchors.
    # (This is heavier than p/li-only, but still manageable for ~333 docs.)
    for el in container.descendants:
        if not getattr(el, "name", None):
            continue

        name = (el.name or "").lower()

        if name in HEADING_TAGS:
            txt = clean_whitespace(el.get_text(" ", strip=True))
            if txt:
                heading_path = (heading_path + [txt])[-4:]
            # headings often carry anchors (id), collect as pending
            anchors_here = _collect_anchor_tokens(el)
            if anchors_here:
                st["anchors_seen"] += len(anchors_here)
                pending_anchors.extend(anchors_here)
            continue

        # collect anchors on any element (common: wrapper divs, spans)
        anchors_here = _collect_anchor_tokens(el)
        if anchors_here:
            st["anchors_seen"] += len(anchors_here)
            pending_anchors.extend(anchors_here)

        if name not in BLOCK_TAGS:
            continue

        st["blocks_seen"] += 1

        if has_bad_ancestor(el):
            st["blocks_bad_ancestor"] += 1
            continue

        text = clean_whitespace(el.get_text(" ", strip=True))
        if not is_meaningful_passage(text, min_chars=min_chars, min_words=min_words):
            st["blocks_too_short_or_nav"] += 1
            continue

        local_id = element_local_id(el)
        if not local_id:
            gen_counter += 1
            local_id = f"p{gen_counter:06d}"

        if local_id in seen_local:
            local_id = f"{local_id}__dup_{short_hash(text)}"
        seen_local.add(local_id)

        # Build anchor aliases for this kept passage
        # Always include its own local_id; also include pending anchors.
        anchor_ids: list[str] = []
        anchor_ids.append(local_id)

        if pending_anchors:
            # remove duplicates and self
            seen_a = set(anchor_ids)
            for a in pending_anchors:
                a = (a or "").strip()
                if not a or a in seen_a:
                    continue
                seen_a.add(a)
                anchor_ids.append(a)
            st["anchors_assigned"] += len(anchor_ids) - 1
            pending_anchors = []  # reset after assignment

        passage_url = f"{doc.doc_url}#{local_id}" if doc.doc_url else ""
        refs = extract_links(el)

        passage_uid = make_passage_uid(doc.doc_url, local_id, fallback=doc.file_path)

        passages.append(
            {
                "passage_uid": passage_uid,
                "passage_id": f"{doc.doc_id}::{local_id}",
                "doc_id": doc.doc_id,
                "eId": local_id,
                "tag": "paragraph",
                "source_tag": name,
                "title": doc.title,
                "heading_path": heading_path,
                "passage": text,
                "doc_url": doc.doc_url,
                "passage_url": passage_url,
                "anchor_id": local_id,
                "anchor_ids": anchor_ids,  # includes local_id + aliases
                "refs": refs,
            }
        )
        st["blocks_written"] += 1

        if debug and st["blocks_written"] <= 3:
            print(f"[DEBUG SAMPLE] {doc.doc_id}::{local_id} ({name}) -> {text[:160]}")
            if len(anchor_ids) > 1:
                print(
                    f"             anchor_ids (aliases): {anchor_ids[:8]}{' ...' if len(anchor_ids) > 8 else ''}"
                )

    return passages, st


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate_corpus(
    raw_dir: str,
    out_path: str,
    max_docs: int = 0,
    min_chars: int = 25,
    min_words: int = 4,
    debug_first_n: int = 0,
    print_zero_docs: bool = False,
) -> dict[str, Any]:
    """
    Programmatic entrypoint for corpus generation.
    Returns a summary dict for adapter_report.json.
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    docs = load_docs(raw_dir)
    if max_docs and max_docs > 0:
        docs = docs[:max_docs]

    all_rows: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    dup_uids = 0
    empty_docs = 0

    # Aggregate stats (global)
    agg = {"blocks_seen": 0, "blocks_bad_ancestor": 0, "blocks_too_short": 0, "blocks_written": 0}

    # note: build_passages_for_doc already assigns passage_uid
    for i, d in enumerate(tqdm(docs, desc="Extracting passages")):
        debug = i < max(debug_first_n, 0)
        rows, st = build_passages_for_doc(
            d,
            min_chars=min_chars,
            min_words=min_words,
            debug=debug,
        )

        for k in agg:
            agg[k] += int(st.get(k, 0))

        if not rows:
            empty_docs += 1
            if print_zero_docs:
                print(f"[ZERO PASSAGES] {d.file_path} :: {st}")

        # global dedup by passage_uid
        for r in rows:
            uid = (r.get("passage_uid") or "").strip()
            if not uid:
                # fallback (shouldn't happen if build_passages_for_doc sets it)
                uid = make_passage_uid(
                    r.get("doc_url", ""), r.get("eId", ""), fallback=r.get("passage_id", "")
                )
                r["passage_uid"] = uid

            if uid in seen_uids:
                dup_uids += 1
                continue
            seen_uids.add(uid)
            all_rows.append(r)

    write_jsonl(out_path, all_rows)

    return {
        "docs_processed": len(docs),
        "docs_with_0_passages": empty_docs,
        "passages_written": len(all_rows),
        "duplicate_passage_uids": dup_uids,
        "aggregate_block_stats": agg,
        "out_path": out_path,
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="runs/adapter_ukfin/raw/pra_rulebook")
    ap.add_argument(
        "--out_path", type=str, default="runs/adapter_ukfin/processed/passage_corpus.jsonl"
    )
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--min_chars", type=int, default=25)
    ap.add_argument("--min_words", type=int, default=4)
    ap.add_argument("--debug_first_n", type=int, default=2)
    ap.add_argument("--print_zero_docs", action="store_true")
    args = ap.parse_args()

    rep = generate_corpus(
        raw_dir=args.raw_dir,
        out_path=args.out_path,
        max_docs=args.max_docs,
        min_chars=args.min_chars,
        min_words=args.min_words,
        debug_first_n=args.debug_first_n,
        print_zero_docs=args.print_zero_docs,
    )

    print("------------------------------------------------")
    print("UKFIN v1 (PRA Rulebook) corpus generation complete.")
    print(f"Docs processed:              {rep['docs_processed']}")
    print(f"Docs with 0 passages:        {rep['docs_with_0_passages']}")
    print(f"Passages written:            {rep['passages_written']}")
    print(f"Duplicate passage_uids:      {rep['duplicate_passage_uids']}")
    print("")
    print("[Aggregate block stats]")
    for k, v in rep["aggregate_block_stats"].items():
        print(f"  {k}: {v}")
    print("")
    print(f"Output:                      {rep['out_path']}")
    print("------------------------------------------------")


if __name__ == "__main__":
    main()
