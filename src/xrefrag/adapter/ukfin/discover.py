# src/xrefrag/adapter/ukfin/discover.py
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from lxml import html

from xrefrag.adapter.ukfin.types import WebCorpusScope

logger = logging.getLogger(__name__)

_DATE_SUFFIX_RE = re.compile(r"\d{2}-\d{2}-\d{4}$")


@dataclass(frozen=True)
class PraDiscoverReport:
    seed_urls: list[str]
    discovered_total: int
    written_total: int
    output_path: str
    timestamp_utc: str


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _polite_get(url: str, scope: WebCorpusScope) -> str:
    headers = {
        "User-Agent": scope.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    r.raise_for_status()
    # PRA is utf-8 in practice; requests will decode reasonably via apparent encoding
    return r.text


def _is_pra_rule_or_guidance_link(href: str) -> bool:
    # keep only rule/guidance leaf pages that end with DD-MM-YYYY
    if not href:
        return False
    if ("/pra-rules/" not in href) and ("/guidance/" not in href):
        return False
    return _DATE_SUFFIX_RE.search(href) is not None


def _normalize_url(base_url: str, href: str) -> str | None:
    # Make absolute, strip fragments/query noise
    abs_url = urljoin(base_url, href)
    u = urlparse(abs_url)
    if not u.scheme.startswith("http"):
        return None
    if "prarulebook.co.uk" not in (u.netloc or ""):
        return None
    # keep path only (PRA pages are stable by path)
    clean = f"{u.scheme}://{u.netloc}{u.path}"
    return clean


def discover_pra_allowlist(
    *,
    scope: WebCorpusScope,
    output_txt_path: Path,
    registry_dir: Path | None = None,
    subset_docs: int | None = None,
) -> PraDiscoverReport:
    """
    Discover PRA Rulebook URLs from the two index pages and write them to output_txt_path.

    - subset_docs: if set, only write the first N URLs (sorted), useful for quick demos.
    - registry_dir: if set, write a discover report JSON alongside.
    """
    base = str(scope.base_url or "https://www.prarulebook.co.uk/").rstrip("/") + "/"

    seed_urls = [
        urljoin(base, "pra-rules"),
        urljoin(base, "guidance"),
    ]

    discovered: set[str] = set()

    for seed in seed_urls:
        logger.info("PRA discover: fetching seed page %s", seed)
        page = _polite_get(seed, scope)
        doc = html.fromstring(page)

        # All <a href="...">
        for href in doc.xpath("//a/@href"):
            if not _is_pra_rule_or_guidance_link(href):
                continue
            norm = _normalize_url(base, href)
            if norm:
                discovered.add(norm)

    urls = sorted(discovered)
    discovered_total = len(urls)

    if subset_docs is not None:
        urls = urls[: max(0, int(subset_docs))]

    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    output_txt_path.write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")

    report = PraDiscoverReport(
        seed_urls=seed_urls,
        discovered_total=discovered_total,
        written_total=len(urls),
        output_path=str(output_txt_path),
        timestamp_utc=_now(),
    )

    if registry_dir is not None:
        registry_dir.mkdir(parents=True, exist_ok=True)
        rep_path = registry_dir / "pra_rulebook_discover.json"
        rep_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        logger.info("PRA discover: wrote %s", rep_path)

    logger.info(
        "PRA discover: discovered=%d, written=%d -> %s",
        report.discovered_total,
        report.written_total,
        report.output_path,
    )
    return report
