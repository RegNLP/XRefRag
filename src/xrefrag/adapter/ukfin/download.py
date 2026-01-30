# src/xrefrag/adapter/ukfin/download.py
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from xrefrag.adapter.ukfin.types import UkFinCorpusSource, WebCorpusScope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchResult:
    source: str
    url: str
    raw_path: str
    ok: bool
    status_code: int | None
    error: str | None
    bytes: int | None = None


def _read_allowlist(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Allowlist file not found: {path}")

    urls: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        urls.append(s)

    # stable + dedup
    return sorted(set(urls))


def _safe_name(url: str) -> str:
    u = urlparse(url)
    base = (u.netloc + u.path).strip("/").replace("/", "_").replace(":", "_")
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    if not base:
        base = "url"
    return f"{base}__{h}.html"


def _polite_get(url: str, scope: WebCorpusScope) -> tuple[bytes, int]:
    headers = {
        "User-Agent": scope.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return r.content, r.status_code


def download_allowlist(
    *,
    source: UkFinCorpusSource,
    scope: WebCorpusScope,
    raw_dir: Path,
    registry_dir: Path,
    subset_docs: int | None = None,
    sleep_seconds: float | None = None,
) -> list[FetchResult]:
    """
    Download HTML pages listed in allowlist_urls_path for a given source.
    Writes:
      raw_dir/<source>/*.html
      registry_dir/<source>_fetch.jsonl
    """
    delay = scope.sleep_seconds if sleep_seconds is None else float(sleep_seconds)

    if not scope.allowlist_urls_path:
        raise ValueError(f"{source.value}: allowlist_urls_path is required (got None)")

    urls = _read_allowlist(scope.allowlist_urls_path)

    # Enforce caps
    cap = max(0, int(scope.max_docs_total))
    if cap > 0:
        urls = urls[:cap]
    if subset_docs is not None:
        urls = urls[: max(0, int(subset_docs))]

    out_subdir = raw_dir / source.value
    out_subdir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    # For run directories, overwrite is usually cleaner than append
    reg_path = registry_dir / f"{source.value}_fetch.jsonl"

    results: list[FetchResult] = []
    with reg_path.open("w", encoding="utf-8") as reg:
        for url in tqdm(urls, desc=f"Downloading {source.value}"):
            fname = _safe_name(url)
            out_path = out_subdir / fname

            # Skip if already downloaded
            if out_path.exists() and out_path.stat().st_size > 0:
                fr = FetchResult(
                    source=source.value,
                    url=url,
                    raw_path=str(out_path),
                    ok=True,
                    status_code=None,  # unknown
                    error=None,
                    bytes=out_path.stat().st_size,
                )
                results.append(fr)
                reg.write(json.dumps(asdict(fr), ensure_ascii=False) + "\n")
                continue

            try:
                content, status = _polite_get(url, scope)
                out_path.write_bytes(content)
                fr = FetchResult(
                    source=source.value,
                    url=url,
                    raw_path=str(out_path),
                    ok=True,
                    status_code=status,
                    error=None,
                    bytes=len(content),
                )
            except requests.HTTPError as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                fr = FetchResult(
                    source=source.value,
                    url=url,
                    raw_path=str(out_path),
                    ok=False,
                    status_code=code,
                    error=str(e),
                    bytes=None,
                )
            except Exception as e:
                fr = FetchResult(
                    source=source.value,
                    url=url,
                    raw_path=str(out_path),
                    ok=False,
                    status_code=None,
                    error=str(e),
                    bytes=None,
                )

            results.append(fr)
            reg.write(json.dumps(asdict(fr), ensure_ascii=False) + "\n")
            reg.flush()

            time.sleep(delay)

    ok_count = sum(1 for r in results if r.ok)
    logger.info(
        "Downloaded %d/%d URLs for %s. Registry: %s",
        ok_count,
        len(results),
        source.value,
        reg_path,
    )
    return results
