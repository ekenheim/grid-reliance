#!/usr/bin/env python3
"""
Test ENTSO-E Transparency Platform API with your security token.

Loads ENTSOE_TOKEN from .env (repo root) or environment, then runs one
day-ahead prices request for Sweden to verify the key works.

API: https://web-api.tp.entsoe.eu/api
Docs: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html

Usage:
  From repo root:  python data-engineering/scripts/fetch_entsoe_test.py
  Or:             cd data-engineering && python scripts/fetch_entsoe_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# Load .env from repo root
_repo_root = Path(__file__).resolve().parents[2]
_env = _repo_root / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass

ENTSOE_BASE = "https://web-api.tp.entsoe.eu/api"
# Document type A44 = Day-ahead prices; Sweden = 10Y1001A1001A44P
DOCUMENT_TYPE_A44 = "A44"
DOMAIN_SWEDEN = "10Y1001A1001A44P"


def test_entsoe_token(token: str) -> tuple[bool, str]:
    """Call ENTSO-E API (day-ahead prices Sweden, 1 day). Returns (success, message)."""
    period_start = "202401010000"  # 2024-01-01 00:00
    period_end = "202401012300"    # 2024-01-01 23:00
    params = {
        "securityToken": token.strip().strip('"'),
        "documentType": DOCUMENT_TYPE_A44,
        "in_Domain": DOMAIN_SWEDEN,
        "out_Domain": DOMAIN_SWEDEN,
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    url = f"{ENTSOE_BASE}?{urlencode(params)}"
    try:
        req = Request(url, headers={"Accept": "application/xml"})
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return False, str(e)
    if "Acknowledgement_MarketDocument" in body or "Publication_MarketDocument" in body:
        return True, f"OK (got {len(body)} chars)"
    if "Invalid security token" in body or "401" in body or "403" in body:
        return False, "Invalid or unauthorized token (check ENTSOE_TOKEN)"
    return False, body[:500] if len(body) > 500 else body


def main() -> int:
    token = os.environ.get("ENTSOE_TOKEN", "").strip().strip('"')
    if not token:
        print("ENTSOE_TOKEN not set. Add it to .env (repo root) or export it.", file=sys.stderr)
        return 1
    ok, msg = test_entsoe_token(token)
    if ok:
        print("ENTSO-E token works:", msg)
        return 0
    print("ENTSO-E request failed:", msg, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
