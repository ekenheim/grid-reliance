#!/usr/bin/env python3
"""
Fetch ERA5 single-levels (10m wind) for Nordic region from ECMWF CDS API.

Writes NetCDF to local disk and optionally uploads to S3-compatible Bronze
(MinIO or Ceph RGW). Requests one month per call to stay within CDS limits.

Usage:
  Set up ~/.cdsapirc (url + key from https://cds.climate.copernicus.eu/profile).
  Then:
    python fetch_era5_nordic.py --start 2023-01 --end 2023-12 --output-dir ./era5_nordic
    python fetch_era5_nordic.py --start 2023-01 --end 2023-03 --upload-bronze  # upload to Bronze

Environment (optional, for Bronze upload):
  BRONZE_ENDPOINT (e.g. http://localhost:9000 or Ceph RGW URL)
  BRONZE_ACCESS_KEY / BRONZE_SECRET_KEY
  BRONZE_BUCKET (e.g. grid-resilience-bronze)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Load .env from repo root so CDSAPI_URL / CDSAPI_KEY are set if present
_repo_root = Path(__file__).resolve().parents[2]
_env = _repo_root / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass

# Nordic bounding box: North, West, South, East (lat/lon)
# Covers Norway, Sweden, Finland, Denmark, Baltic
NORDIC_AREA = [72, -10, 54, 35]

ERA5_DATASET = "reanalysis-era5-single-levels"
# Wind at 10m for capacity factor / drought correlation
ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


def parse_month(s: str) -> tuple[int, int]:
    """Parse 'YYYY-MM' -> (year, month)."""
    parts = s.split("-")
    if len(parts) != 2:
        raise ValueError(f"Expected YYYY-MM, got {s}")
    return int(parts[0]), int(parts[1])


def month_days(year: int, month: int) -> int:
    if month in (4, 6, 9, 11):
        return 30
    if month == 2:
        return 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
    return 31


def month_range(start: str, end: str):
    """Yield (year, month) from start to end inclusive."""
    y1, m1 = parse_month(start)
    y2, m2 = parse_month(end)
    if (y1, m1) > (y2, m2):
        raise ValueError(f"Start {start} after end {end}")
    y, m = y1, m1
    while (y, m) <= (y2, m2):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def fetch_month(cds, year: int, month: int, output_path: Path) -> bool:
    """Request one month of ERA5 Nordic and save as NetCDF. Returns True on success."""
    date_str = f"{year}-{month:02d}-01/{year}-{month:02d}-{month_days(year, month):02d}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "product_type": "reanalysis",
        "variable": ERA5_VARIABLES,
        "date": date_str,
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": NORDIC_AREA,
        "format": "netcdf",
    }
    try:
        cds.retrieve(ERA5_DATASET, request, str(output_path))
        return output_path.is_file()
    except Exception as e:
        print(f"CDS request failed for {year}-{month:02d}: {e}", file=sys.stderr)
        return False


def upload_to_bronze(local_path: Path, s3_key: str) -> bool:
    """Upload a file to S3-compatible Bronze using env BRONZE_*."""
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
        print(f"Uploaded to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        print(f"Upload failed: {e}", file=sys.stderr)
        return False


def _load_cdsapirc() -> tuple[str, str] | None:
    """Load url and key from env (CDSAPI_URL, CDSAPI_KEY) or .cdsapirc. Check repo root then home."""
    url = os.environ.get("CDSAPI_URL", "").strip()
    key = os.environ.get("CDSAPI_KEY", "").strip()
    if url and key:
        return url, key
    candidates = [
        Path(__file__).resolve().parents[2] / ".cdsapirc",  # repo root
        Path.home() / ".cdsapirc",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            text = path.read_text().strip()
            url = key = None
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("url:"):
                    url = line.split(":", 1)[1].strip()
                elif line.startswith("key:"):
                    key = line.split(":", 1)[1].strip()
            if url and key:
                return url, key
        except Exception:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch ERA5 Nordic (10m wind) from CDS.")
    parser.add_argument("--start", required=True, help="Start month YYYY-MM")
    parser.add_argument("--end", required=True, help="End month YYYY-MM")
    parser.add_argument("--output-dir", type=Path, default=Path("era5_nordic"), help="Local output directory")
    parser.add_argument("--upload-bronze", action="store_true", help="Upload each NetCDF to Bronze (S3)")
    args = parser.parse_args()

    try:
        import cdsapi
    except ImportError:
        print("Install cdsapi: pip install cdsapi", file=sys.stderr)
        return 1

    rc = _load_cdsapirc()
    if not rc:
        print(
            "Missing CDS credentials. Set CDSAPI_URL and CDSAPI_KEY in .env, or put a .cdsapirc file "
            "(with 'url:' and 'key:' lines) in repo root or ~/.cdsapirc. Get key: https://cds.climate.copernicus.eu/profile",
            file=sys.stderr,
        )
        return 1
    url, key = rc
    cds = cdsapi.Client(url=url, key=key)
    ok = 0
    for year, month in month_range(args.start, args.end):
        out_name = f"era5_nordic_10m_wind_{year}_{month:02d}.nc"
        out_path = args.output_dir / out_name
        if out_path.exists():
            print(f"Skip (exists): {out_path}")
            ok += 1
            if args.upload_bronze:
                upload_to_bronze(out_path, f"era5/single-levels/{out_name}")
            continue
        if fetch_month(cds, year, month, out_path):
            ok += 1
            if args.upload_bronze:
                upload_to_bronze(out_path, f"era5/single-levels/{out_name}")
    print(f"Fetched {ok} months.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
