#!/usr/bin/env python3
"""
Fetch ENTSO-E actual generation (by production type) and actual total load
for the Nordic bidding zones and upload to the Bronze bucket.

Writes monthly CSV files to:
  Bronze: entsoe/generation/actual_generation_YYYY_MM.csv
  Bronze: entsoe/load/actual_load_YYYY_MM.csv

Schema (same columns entsoe_ingest.py expects):
  timestamp   : ISO-8601 UTC  (e.g. 2024-01-01T00:00:00Z)
  region_id   : zone code     (e.g. SE1, NO2)
  variable    : "wind_gen_mwh" | "load_mwh"
  value_mwh   : float

Usage:
  python fetch_entsoe.py --start 2024-01-01 --end 2024-12-31
  python fetch_entsoe.py --start 2024-01-01 --end 2024-12-31 --upload-bronze
  python fetch_entsoe.py --start 2024-01-01 --end 2024-12-31 --upload-bronze --output-dir /tmp/entsoe

Environment (token):
  ENTSOE_TOKEN  — ENTSO-E Transparency Platform security token
                  https://transparency.entsoe.eu → My Account → Security Token

Environment (Bronze upload, all optional):
  BRONZE_ENDPOINT   — e.g. http://rook-ceph-rgw-ceph-objectstore.rook-ceph.svc:80
  BRONZE_ACCESS_KEY — AWS_ACCESS_KEY_ID for Bronze
  BRONZE_SECRET_KEY — AWS_SECRET_ACCESS_KEY for Bronze
  BRONZE_BUCKET     — defaults to grid-resilience-bronze
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

# Load .env from repo root so secrets are available when running locally
_repo_root = Path(__file__).resolve().parents[2]
_env = _repo_root / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass

ENTSOE_BASE = "https://web-api.tp.entsoe.eu/api"

# Nordic bidding zone EIC area codes for the ENTSO-E Transparency Platform.
# Keys are our internal region_id values (matching spark_era5_ingest zone assignments).
NORDIC_ZONES: dict[str, str] = {
    "SE1": "10Y1001A1001A44P",
    "SE2": "10Y1001A1001A45N",
    "SE3": "10Y1001A1001A46L",
    "SE4": "10Y1001A1001A47J",
    "NO1": "10YNO-1--------2",
    "NO2": "10YNO-2--------T",
    "DK1": "10YDK-1--------W",
    "DK2": "10YDK-2--------M",
    "FI":  "10YFI-1--------U",
}

# PSR types for wind generation (ENTSO-E production type codes)
WIND_PSR_TYPES = {
    "B18": "wind_offshore",
    "B19": "wind_onshore",
}


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def parse_date(s: str) -> datetime:
    """Parse YYYY-MM-DD -> datetime (UTC midnight)."""
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(f"Expected YYYY-MM-DD, got '{s}'") from None


def month_range(start: datetime, end: datetime):
    """Yield (year, month) pairs from start to end (both inclusive by month)."""
    y, m = start.year, start.month
    ey, em = end.year, end.month
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def month_window(year: int, month: int) -> tuple[str, str]:
    """Return (period_start, period_end) strings in ENTSO-E format YYYYMMDDhhmm."""
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    return (
        f"{year}{month:02d}01" "0000",
        f"{year}{month:02d}{last_day:02d}" "2300",
    )


# ---------------------------------------------------------------------------
# ENTSO-E HTTP helpers
# ---------------------------------------------------------------------------

def _get(params: dict, token: str, retries: int = 3) -> str:
    params["securityToken"] = token
    url = f"{ENTSOE_BASE}?{urlencode(params)}"
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"Accept": "application/xml"})
            with urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"ENTSO-E request failed after {retries} attempts: {last_exc}") from last_exc


def _check_error(xml_text: str, context: str = "") -> None:
    """Raise if the ENTSO-E response is an acknowledgement error."""
    if "Acknowledgement_MarketDocument" in xml_text:
        root = ET.fromstring(xml_text)
        ns = {"ns": "urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0"}
        reason = root.findtext("ns:Reason/ns:text", default="(no reason)", namespaces=ns)
        raise RuntimeError(f"ENTSO-E error{' (' + context + ')' if context else ''}: {reason}")


# ---------------------------------------------------------------------------
# Generation fetch (A75 / GL_Actual_Generation_perType)
# ---------------------------------------------------------------------------

def _parse_generation(xml_text: str, region_id: str) -> list[dict]:
    """Parse ActualGenerationPerProductionType XML into list of row dicts."""
    root = ET.fromstring(xml_text)
    ns_uri = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
    ns = {"ns": ns_uri} if ns_uri else {}

    def find(el, tag):
        return el.find(f"ns:{tag}", ns) if ns else el.find(tag)

    def findall(el, tag):
        return el.findall(f"ns:{tag}", ns) if ns else el.findall(tag)

    rows: list[dict] = []
    for ts_block in findall(root, "TimeSeries"):
        psr_el = find(ts_block, "MktPSRType")
        psr_type_el = find(psr_el, "psrType") if psr_el is not None else None
        psr_code = psr_type_el.text.strip() if psr_type_el is not None else ""
        if psr_code not in WIND_PSR_TYPES:
            continue  # only wind production types

        period_el = find(ts_block, "Period")
        if period_el is None:
            continue
        start_el = find(period_el, "timeInterval")
        if start_el is None:
            continue
        start_str = (find(start_el, "start") or start_el).text
        if start_str is None:
            start_el2 = find(period_el, "timeInterval/start")
            start_str = start_el2.text if start_el2 is not None else None
        if start_str is None:
            continue

        # Resolution: PT60M or PT15M etc.
        res_el = find(period_el, "resolution")
        resolution_minutes = 60
        if res_el is not None:
            res_txt = res_el.text.strip().upper()
            if "PT15M" in res_txt:
                resolution_minutes = 15
            elif "PT30M" in res_txt:
                resolution_minutes = 30

        # Parse start time
        start_dt_str = start_str.strip().replace("Z", "+00:00")
        try:
            period_start = datetime.fromisoformat(start_dt_str).replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                period_start = datetime.strptime(start_dt_str[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        for point_el in findall(period_el, "Point"):
            pos_el = find(point_el, "position")
            qty_el = find(point_el, "quantity")
            if pos_el is None or qty_el is None:
                continue
            try:
                position = int(pos_el.text.strip())
                quantity = float(qty_el.text.strip())
            except (ValueError, AttributeError):
                continue
            offset_minutes = (position - 1) * resolution_minutes
            from datetime import timedelta
            ts = period_start + timedelta(minutes=offset_minutes)
            rows.append({
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "region_id": region_id,
                "variable": "wind_gen_mwh",
                "value_mwh": quantity,
            })
    return rows


def fetch_generation(token: str, zone_id: str, eic: str, year: int, month: int) -> list[dict]:
    """Fetch actual wind generation for one zone/month. Returns list of row dicts."""
    period_start, period_end = month_window(year, month)
    params = {
        "documentType": "A75",
        "processType": "A16",
        "in_Domain": eic,
        "out_Domain": eic,
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    try:
        xml_text = _get(params, token)
        _check_error(xml_text, context=f"generation {zone_id} {year}-{month:02d}")
        return _parse_generation(xml_text, zone_id)
    except Exception as exc:
        print(f"  Warning: generation fetch failed for {zone_id} {year}-{month:02d}: {exc}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Load fetch (A65 / Actual_TotalLoad)
# ---------------------------------------------------------------------------

def _parse_load(xml_text: str, region_id: str) -> list[dict]:
    """Parse ActualTotalLoad XML into list of row dicts."""
    root = ET.fromstring(xml_text)
    ns_uri = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
    ns = {"ns": ns_uri} if ns_uri else {}

    def find(el, tag):
        return el.find(f"ns:{tag}", ns) if ns else el.find(tag)

    def findall(el, tag):
        return el.findall(f"ns:{tag}", ns) if ns else el.findall(tag)

    rows: list[dict] = []
    for ts_block in findall(root, "TimeSeries"):
        period_el = find(ts_block, "Period")
        if period_el is None:
            continue
        ti_el = find(period_el, "timeInterval")
        if ti_el is None:
            continue
        start_el = find(ti_el, "start")
        if start_el is None or start_el.text is None:
            continue
        start_dt_str = start_el.text.strip().replace("Z", "+00:00")
        try:
            period_start = datetime.fromisoformat(start_dt_str).replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                period_start = datetime.strptime(start_dt_str[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        res_el = find(period_el, "resolution")
        resolution_minutes = 60
        if res_el is not None:
            res_txt = res_el.text.strip().upper()
            if "PT15M" in res_txt:
                resolution_minutes = 15
            elif "PT30M" in res_txt:
                resolution_minutes = 30

        for point_el in findall(period_el, "Point"):
            pos_el = find(point_el, "position")
            qty_el = find(point_el, "quantity")
            if pos_el is None or qty_el is None:
                continue
            try:
                position = int(pos_el.text.strip())
                quantity = float(qty_el.text.strip())
            except (ValueError, AttributeError):
                continue
            offset_minutes = (position - 1) * resolution_minutes
            from datetime import timedelta
            ts = period_start + timedelta(minutes=offset_minutes)
            rows.append({
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "region_id": region_id,
                "variable": "load_mwh",
                "value_mwh": quantity,
            })
    return rows


def fetch_load(token: str, zone_id: str, eic: str, year: int, month: int) -> list[dict]:
    """Fetch actual total load for one zone/month."""
    period_start, period_end = month_window(year, month)
    params = {
        "documentType": "A65",
        "processType": "A16",
        "outBiddingZone_Domain": eic,
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    try:
        xml_text = _get(params, token)
        _check_error(xml_text, context=f"load {zone_id} {year}-{month:02d}")
        return _parse_load(xml_text, zone_id)
    except Exception as exc:
        print(f"  Warning: load fetch failed for {zone_id} {year}-{month:02d}: {exc}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# S3 Bronze upload (mirrors fetch_era5_nordic.py)
# ---------------------------------------------------------------------------

def upload_to_bronze(local_path: Path, s3_key: str) -> bool:
    endpoint = os.environ.get("BRONZE_ENDPOINT", "").strip()
    if not endpoint:
        print("BRONZE_ENDPOINT not set; skipping upload.", file=sys.stderr)
        return False
    if not endpoint.startswith("http"):
        endpoint = f"http://{endpoint}"
    import boto3
    from botocore.config import Config
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("BRONZE_ACCESS_KEY", ""),
        aws_secret_access_key=os.environ.get("BRONZE_SECRET_KEY", ""),
        region_name=os.environ.get("BRONZE_REGION", "us-east-1"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    bucket = os.environ.get("BRONZE_BUCKET", "grid-resilience-bronze")
    try:
        client.upload_file(str(local_path), bucket, s3_key)
        print(f"  Uploaded to s3://{bucket}/{s3_key}")
        return True
    except Exception as exc:
        print(f"  Upload failed: {exc}", file=sys.stderr)
        return False


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "region_id", "variable", "value_mwh"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch ENTSO-E generation & load for Nordic zones.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=Path("entsoe_data"), help="Local output directory")
    parser.add_argument("--upload-bronze", action="store_true", help="Upload CSVs to Bronze (S3)")
    args = parser.parse_args()

    token = os.environ.get("ENTSOE_TOKEN", "").strip().strip('"')
    if not token:
        print(
            "ENTSOE_TOKEN not set. Add it to .env or export it.\n"
            "Get token: https://transparency.entsoe.eu → My Account → Security Token",
            file=sys.stderr,
        )
        return 1

    start_dt = parse_date(args.start)
    end_dt = parse_date(args.end)

    gen_ok = load_ok = 0
    for year, month in month_range(start_dt, end_dt):
        print(f"Processing {year}-{month:02d} …")
        gen_rows: list[dict] = []
        load_rows: list[dict] = []

        for zone_id, eic in NORDIC_ZONES.items():
            gen_rows.extend(fetch_generation(token, zone_id, eic, year, month))
            load_rows.extend(fetch_load(token, zone_id, eic, year, month))
            # Brief pause to stay within ENTSO-E rate limits
            time.sleep(0.5)

        gen_name = f"actual_generation_{year}_{month:02d}.csv"
        load_name = f"actual_load_{year}_{month:02d}.csv"
        gen_path = args.output_dir / gen_name
        load_path = args.output_dir / load_name

        if gen_rows:
            _write_csv(gen_rows, gen_path)
            print(f"  Wrote {len(gen_rows)} rows -> {gen_path}")
            gen_ok += 1
            if args.upload_bronze:
                upload_to_bronze(gen_path, f"entsoe/generation/{gen_name}")
        else:
            print(f"  No generation data for {year}-{month:02d} (all zones)")

        if load_rows:
            _write_csv(load_rows, load_path)
            print(f"  Wrote {len(load_rows)} rows -> {load_path}")
            load_ok += 1
            if args.upload_bronze:
                upload_to_bronze(load_path, f"entsoe/load/{load_name}")
        else:
            print(f"  No load data for {year}-{month:02d} (all zones)")

    print(f"Done. generation months: {gen_ok}, load months: {load_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
