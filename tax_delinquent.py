"""Fetch Florida county delinquent-property-tax lists for the lead scorer.

Most FL counties run their annual tax-certificate sale through LienHub, which
exposes the advertised delinquent list as a public CSV (no login). This script
pulls that CSV for every county that's on LienHub, normalizes the parcel IDs,
and writes ``data/tax_delinquent.parquet``. ``score_leads.py`` joins it onto
the NAL parcels to flag tax-delinquent leads — the strongest motivated-seller
signal there is.

Coverage notes:
  - LienHub covers ~37 counties. Counties not on LienHub are skipped (Manatee
    uses a Pacific Blue S3 .xls; ~13 counties use RealAuction — both TODO).
  - The tax-collector "Account No." only equals the NAL PARCEL_ID for some
    counties (e.g. Pasco). Where it differs (Hillsborough/Pinellas use a
    separate folio) the join simply yields nothing — harmless, just no flag.
  - Advertised lists are live ~mid-April through the ~June 1 sale; run then.

Usage:  python tax_delinquent.py
"""

from __future__ import annotations

import csv as _csv
import io
import re
import time
from pathlib import Path

import polars as pl
import requests

from score_leads import COUNTIES

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
OUTPUT = Path(__file__).parent / "data" / "tax_delinquent.parquet"


def norm_parcel(x: object) -> str:
    """Strip punctuation/whitespace and upper-case — the parcel join key."""
    return re.sub(r"[^0-9A-Za-z]", "", str(x or "")).upper()


def _money(x: object) -> float:
    cleaned = re.sub(r"[^0-9.]", "", str(x or ""))
    try:
        return round(float(cleaned), 2) if cleaned else 0.0
    except ValueError:
        return 0.0


def lienhub_slug(county_name: str) -> str:
    return county_name.strip().lower().replace(" ", "-")


def fetch_lienhub(county_name: str) -> list[dict] | None:
    """Pull one county's advertised delinquent list from LienHub.

    Returns a list of {parcel_norm, tax_amount_owed, tax_homestead} dicts, or
    None if the county isn't on LienHub / the list isn't currently published.
    """
    base = f"https://lienhub.com/county/{lienhub_slug(county_name)}/certsale/main"
    s = requests.Session()
    s.headers.update(UA)
    try:
        page = s.get(base, timeout=30)
    except requests.RequestException:
        return None
    if page.status_code != 200 or "certsale" not in page.text.lower():
        return None
    m = re.search(r"unique_id=([A-Za-z0-9]+)", page.text)
    uid = m.group(1) if m else ""
    try:
        resp = s.get(
            f"{base}?unique_id={uid}&use_this=download_advertised_list", timeout=120
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200 or "csv" not in resp.headers.get("content-type", "").lower():
        return None
    rows = list(_csv.DictReader(io.StringIO(resp.text)))
    if not rows or "Account No." not in rows[0]:
        return None
    out: list[dict] = []
    for r in rows:
        pn = norm_parcel(r.get("Account No."))
        if not pn:
            continue
        out.append(
            {
                "parcel_norm": pn,
                "tax_amount_owed": _money(r.get("Face Amount")),
                "tax_homestead": str(r.get("Homestead", "")).strip() in ("1", "Y", "y"),
            }
        )
    return out or None


def main() -> None:
    frames: list[pl.DataFrame] = []
    hit, miss = [], []
    for name in COUNTIES.values():
        rows = fetch_lienhub(name)
        if rows:
            df = (
                pl.DataFrame(rows)
                .with_columns(pl.lit(name).alias("county_name"))
                # one row per parcel — keep the largest amount owed
                .sort("tax_amount_owed", descending=True)
                .unique(subset=["county_name", "parcel_norm"], keep="first")
            )
            frames.append(df)
            hit.append(name)
            print(f"  {name}: {df.height:,} delinquent parcels")
        else:
            miss.append(name)
        time.sleep(0.3)

    if not frames:
        raise SystemExit("no LienHub counties returned a list — sale window may be closed")

    combined = pl.concat(frames, how="vertical_relaxed")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(OUTPUT, compression="zstd")
    print(
        f"\n{len(hit)} counties on LienHub, {combined.height:,} delinquent parcels "
        f"-> {OUTPUT}"
    )
    print(f"not on LienHub (skipped): {', '.join(miss)}")


if __name__ == "__main__":
    main()
