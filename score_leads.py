"""Score Florida NAL parcels for motivated-seller signals.

Reads parsed county parquets from `fl-llc-properties/data/nal_parsed/` and
emits `data/leads.csv` for the dashboard.

Usage:
    python score_leads.py                                 # default 6 metro counties
    python score_leads.py --counties 39 23 58             # custom (Hillsborough, Dade, Orange)
    python score_leads.py --counties all                  # statewide
    python score_leads.py --input <dir> --output <file>   # custom paths
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import polars as pl

# ── County code → name (FL DOR official) ──────────────────────────────────────
COUNTIES: dict[int, str] = {
    11: "Alachua", 12: "Baker", 13: "Bay", 14: "Bradford", 15: "Brevard",
    16: "Broward", 17: "Calhoun", 18: "Charlotte", 19: "Citrus", 20: "Clay",
    21: "Collier", 22: "Columbia", 23: "Dade", 24: "Desoto", 25: "Dixie",
    26: "Duval", 27: "Escambia", 28: "Flagler", 29: "Franklin", 30: "Gadsden",
    31: "Gilchrist", 32: "Glades", 33: "Gulf", 34: "Hamilton", 35: "Hardee",
    36: "Hendry", 37: "Hernando", 38: "Highlands", 39: "Hillsborough", 40: "Holmes",
    41: "Indian River", 42: "Jackson", 43: "Jefferson", 44: "Lafayette", 45: "Lake",
    46: "Lee", 47: "Leon", 48: "Levy", 49: "Liberty", 50: "Madison",
    51: "Manatee", 52: "Marion", 53: "Martin", 54: "Monroe", 55: "Nassau",
    56: "Okaloosa", 57: "Okeechobee", 58: "Orange", 59: "Osceola", 60: "Palm Beach",
    61: "Pasco", 62: "Pinellas", 63: "Polk", 64: "Putnam", 65: "Saint Johns",
    66: "Saint Lucie", 67: "Santa Rosa", 68: "Sarasota", 69: "Seminole", 70: "Sumter",
    71: "Suwannee", 72: "Taylor", 73: "Union", 74: "Volusia", 75: "Wakulla",
    76: "Walton", 77: "Washington",
}

DEFAULT_COUNTIES = list(range(11, 78))  # All 67 FL counties (11-77 inclusive)
RESIDENTIAL_DOR = {"001", "002", "004", "005", "006", "008"}
MULTIFAMILY_DOR = {"003", "008", "009"}
CURRENT_YEAR = 2026
LONG_HELD_THRESHOLD = CURRENT_YEAR - 25

TRUST_ESTATE_RE = r"\b(TRUST|TRUSTEE|ESTATE|FAMILY|HEIRS|LIVING\s+TR|REV\s+TR|TR\s+OF)\b"
PO_BOX_RE = r"^\s*(P\.?\s*O\.?\s*BOX|POST\s+OFFICE\s+BOX|PO\s+BOX)\b"
BANK_TRUSTEE_RE = (
    r"\b(BANK|REO|N\s*A|NATIONAL\s+ASSOCIATION|FANNIE\s+MAE|FREDDIE\s+MAC|"
    r"FEDERAL\s+NATIONAL|FEDERAL\s+HOME|HUD|VA\s+SECRETARY|MERS|"
    r"DEUTSCHE\s+BANK|WELLS\s+FARGO|US\s+BANK|JPMORGAN|CITIBANK|"
    r"BANK\s+OF\s+AMERICA|MORTGAGE\s+TRUST)\b"
)
SALE_LOW_RATIO = 0.50    # last_sale_price < 50% of JV = bought distressed / inherited / fire sale
SALE_HIGH_RATIO = 1.50   # last_sale_price > 150% of JV = peak overpayer
HIGH_EQUITY_RATIO = 0.40 # last_sale_price < 40% of JV when sale exists = strong equity proxy

# Entity-detection keywords. Mirrors crm/lib/owner-normalize.ts so
# multi_property_owner counts here match what the CRM thinks.
ENTITY_KEYWORDS_RE = (
    r"\b(LLC|LLP|LP|INC|INCORPORATED|CORP|CORPORATION|LTD|LIMITED|CO|"
    r"COMPANY|PA|PLLC|TRUST|TRUSTEE|TRUSTEES|ESTATE|FOUNDATION|"
    r"ASSOCIATION|PARTNERSHIP|PARTNERS|PROPERTIES|INVESTMENTS|HOLDINGS|"
    r"VENTURES|GROUP|REALTY|DEVELOPMENT|DEVELOPERS|BUILDERS|MANAGEMENT|"
    r"ENTERPRISES|CAPITAL|CHURCH|MINISTRIES|TEMPLE|SCHOOL|ACADEMY|"
    r"UNIVERSITY|COLLEGE|HOSPITAL|DEPARTMENT|AUTHORITY|DISTRICT|BANK|"
    r"MORTGAGE|INSURANCE|HOA|COA|CITY OF|COUNTY OF|TOWN OF|STATE OF)\b"
)


def name_norm_expr(col: str) -> pl.Expr:
    """Normalize an owner name with pure polars expressions (streaming-safe).

    Matches crm/lib/owner-normalize.ts behavior:
      uppercase → strip periods → punct→space → &→AND → collapse spaces → trim.
    """
    return (
        pl.col(col)
        .fill_null("")
        .str.to_uppercase()
        .str.replace_all(r"\.", "")
        .str.replace_all(r"[,/\\]", " ")
        .str.replace_all(r"&", " AND ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def discover_parquets(input_dir: Path, county_codes: list[int]) -> list[Path]:
    files = []
    for code in county_codes:
        matches = list(input_dir.glob(f"{code:02d}_*.parquet"))
        if not matches:
            print(f"  warn: no parquet for county code {code} ({COUNTIES.get(code, '?')})")
            continue
        files.extend(matches)
    return files


def score(df: pl.DataFrame) -> pl.DataFrame:
    own_addr_up = pl.col("OWN_ADDR1").fill_null("").str.to_uppercase()
    own_name_up = pl.col("OWN_NAME").fill_null("").str.to_uppercase()

    own_zip5 = pl.col("OWN_ZIPCD").fill_null("").str.slice(0, 5)
    phy_zip5 = pl.col("PHY_ZIPCD").fill_null("").str.slice(0, 5)

    sale_yr = pl.col("SALE_YR1").fill_null(0)
    act_yr = pl.col("ACT_YR_BLT").fill_null(0)

    out_of_state = (pl.col("OWN_STATE").fill_null("") != "FL") & (pl.col("OWN_STATE").fill_null("") != "")
    out_of_zip = (own_zip5 != phy_zip5) & (own_zip5 != "") & (phy_zip5 != "")
    po_box = own_addr_up.str.contains(PO_BOX_RE)
    long_held = (sale_yr > 0) & (sale_yr <= LONG_HELD_THRESHOLD) | (
        (sale_yr == 0) & (act_yr > 0) & (act_yr <= LONG_HELD_THRESHOLD)
    )
    trust_estate = own_name_up.str.contains(TRUST_ESTATE_RE)
    bank_trustee = own_name_up.str.contains(BANK_TRUSTEE_RE)

    sale_price = pl.col("SALE_PRC1").fill_null(0)
    jv = pl.col("JV").fill_null(0)
    has_sale = (sale_price > 0) & (jv > 0)
    sale_low = has_sale & (sale_price < jv * SALE_LOW_RATIO)
    sale_high = has_sale & (sale_price > jv * SALE_HIGH_RATIO) & (sale_yr >= 2019)
    sale_anomaly = sale_low | sale_high
    high_equity = (
        has_sale & (sale_price < jv * HIGH_EQUITY_RATIO)
    ) | (
        # No recorded sale + old building + decent value = inherited high-equity
        (sale_price == 0) & (act_yr > 0) & (act_yr <= LONG_HELD_THRESHOLD) & (jv >= 100_000)
    )

    df = df.with_columns(
        name_norm_expr("OWN_NAME").alias("owner_norm"),
        out_of_state.alias("f_out_of_state"),
        out_of_zip.alias("f_out_of_zip"),
        po_box.alias("f_po_box"),
        long_held.alias("f_long_held_25y"),
        trust_estate.alias("f_trust_estate_name"),
        bank_trustee.alias("f_bank_trustee"),
        sale_anomaly.alias("f_sale_anomaly"),
        high_equity.alias("f_high_equity_proxy"),
    )

    # Entity vs individual: lightweight heuristic — any entity keyword present.
    df = df.with_columns(
        pl.col("owner_norm").str.contains(ENTITY_KEYWORDS_RE).alias("is_entity")
    )

    # multi_property_owner: group by NORMALIZED owner name so "ABC LLC",
    # "ABC, LLC", "ABC L.L.C." all collapse to one entity.
    multi = df.group_by(["CO_NO", "owner_norm"]).len().rename({"len": "owner_parcel_count"})
    df = df.join(multi, on=["CO_NO", "owner_norm"], how="left").with_columns(
        (pl.col("owner_parcel_count") >= 2).alias("f_multi_property_owner")
    )

    # score: 25 points each for out_of_state-or-zip, po_box, long_held, trust_estate
    df = df.with_columns(
        (
            25 * (pl.col("f_out_of_state") | pl.col("f_out_of_zip")).cast(pl.Int32)
            + 25 * pl.col("f_po_box").cast(pl.Int32)
            + 25 * pl.col("f_long_held_25y").cast(pl.Int32)
            + 25 * pl.col("f_trust_estate_name").cast(pl.Int32)
        ).alias("score")
    )

    # Only keep parcels with at least one flag firing
    df = df.filter(
        pl.col("f_out_of_state")
        | pl.col("f_out_of_zip")
        | pl.col("f_po_box")
        | pl.col("f_long_held_25y")
        | pl.col("f_trust_estate_name")
        | pl.col("f_multi_property_owner")
    )

    # Build flags string
    flag_pairs = [
        (pl.col("f_out_of_state"), "out_of_state"),
        (pl.col("f_out_of_zip"), "out_of_zip"),
        (pl.col("f_po_box"), "po_box"),
        (pl.col("f_long_held_25y"), "long_held_25y"),
        (pl.col("f_trust_estate_name"), "trust_estate_name"),
        (pl.col("f_multi_property_owner"), "multi_property_owner"),
        (pl.col("f_bank_trustee"), "bank_trustee"),
        (pl.col("f_sale_anomaly"), "sale_anomaly"),
        (pl.col("f_high_equity_proxy"), "high_equity_proxy"),
    ]
    flag_exprs = [pl.when(cond).then(pl.lit(name)).otherwise(pl.lit("")) for cond, name in flag_pairs]
    df = df.with_columns(
        pl.concat_list(flag_exprs)
        .list.eval(pl.element().filter(pl.element() != ""))
        .list.join(",")
        .alias("flags")
    )

    return df


def shape_output(df: pl.DataFrame) -> pl.DataFrame:
    co_to_name = pl.DataFrame(
        {"CO_NO": [f"{k:02d}" for k in COUNTIES], "county_name": list(COUNTIES.values())}
    )
    df = df.join(co_to_name, on="CO_NO", how="left")

    own_mailing = (
        pl.col("OWN_ADDR1").fill_null("")
        + pl.lit(" / ")
        + pl.col("OWN_CITY").fill_null("")
        + pl.lit(" / ")
        + pl.col("OWN_STATE").fill_null("")
        + pl.lit(" / ")
        + pl.col("OWN_ZIPCD").fill_null("")
    )

    # ── Deal math ──────────────────────────────────────────────────────────
    # est_equity = appraised value − last arm's-length sale price.
    # A sale price ≤ $1,000 is treated as a non-arm's-length transfer
    # (quitclaim / nominal deed), so equity is left unknown for those.
    _arms_length = pl.col("SALE_PRC1") > 1000
    est_equity = (
        pl.when(_arms_length)
        .then(pl.col("JV") - pl.col("SALE_PRC1"))
        .alias("est_equity")
    )
    equity_pct = (
        pl.when(_arms_length & (pl.col("JV") > 0))
        .then((pl.col("JV") - pl.col("SALE_PRC1")) / pl.col("JV"))
        .alias("equity_pct")
    )

    shaped = df.select(
        pl.col("PARCEL_ID").alias("parcel_id"),
        pl.col("CO_NO").alias("county_code"),
        pl.col("county_name"),
        pl.col("PHY_ADDR1").alias("situs_address"),
        pl.col("PHY_CITY").alias("situs_city"),
        pl.col("PHY_ZIPCD").alias("situs_zip"),
        pl.col("OWN_NAME").alias("owner_name"),
        pl.col("owner_norm"),
        pl.col("is_entity"),
        own_mailing.alias("owner_mailing"),
        pl.col("OWN_ADDR1").alias("owner_mailing_street"),
        pl.col("OWN_CITY").alias("owner_mailing_city"),
        pl.col("OWN_STATE").alias("owner_state"),
        pl.col("OWN_ZIPCD").alias("owner_mailing_zip"),
        pl.col("JV").alias("just_value"),
        pl.col("SALE_YR1").alias("last_sale_year"),
        pl.col("SALE_PRC1").alias("last_sale_price"),
        pl.lit("absentee_equity").alias("signal_type"),
        pl.col("score"),
        pl.col("flags"),
        pl.lit("").alias("evidence_url"),
        pl.col("DOR_UC").alias("dor_uc"),
        pl.col("NO_RES_UNTS").alias("residential_units"),
        pl.col("ACT_YR_BLT").alias("year_built"),
        pl.col("TOT_LVG_AREA").alias("living_area"),
        pl.col("SALE_PRC2").alias("prior_sale_price"),
        pl.col("SALE_YR2").alias("prior_sale_year"),
        est_equity,
        equity_pct,
    )

    # opportunity_score (0–100): blends the motivation score with an equity
    # signal. equity_signal = the known equity % when a real sale price
    # exists; else 80 when the high_equity_proxy flag fired (long-held / no
    # mortgage data — almost certainly high equity); else a neutral 50.
    # opportunity_score = 0.6·score + 0.4·equity_signal.
    equity_signal = (
        pl.when(pl.col("equity_pct").is_not_null())
        .then((pl.col("equity_pct") * 100).clip(0, 100))
        .when(pl.col("flags").str.contains("high_equity_proxy"))
        .then(pl.lit(80.0))
        .otherwise(pl.lit(50.0))
    )
    return shaped.with_columns(
        (0.6 * pl.col("score") + 0.4 * equity_signal)
        .round(0)
        .cast(pl.Int64)
        .alias("opportunity_score")
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--counties",
        nargs="*",
        default=None,
        help="County codes (ints) or 'all'. Default: 37 51 61 62 63 68 (metro 6).",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(r"C:\Users\noaho\fl-llc-properties\data\nal_parsed"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "leads.parquet",
        help="Output path. Writes parquet if .parquet, else CSV.",
    )
    p.add_argument("--min-score", type=int, default=75)
    args = p.parse_args()

    if args.counties is None:
        codes = DEFAULT_COUNTIES
    elif len(args.counties) == 1 and args.counties[0].lower() == "all":
        codes = sorted(COUNTIES.keys())
    else:
        codes = [int(c) for c in args.counties]

    print(f"counties: {[(c, COUNTIES.get(c, '?')) for c in codes]}")
    files = discover_parquets(args.input, codes)
    print(f"reading {len(files)} parquet file(s) from {args.input}")
    if not files:
        raise SystemExit("no input files found")

    # Process one county at a time — keeps peak RAM low.
    # multi_property_owner is already intra-county, so per-county is correct.
    per_county_out: list[pl.DataFrame] = []
    grand_loaded = 0
    grand_flagged = 0

    expected_cols = {
        "TOT_LVG_AREA": pl.Int64, "NO_RES_UNTS": pl.Int64,
        "ACT_YR_BLT": pl.Int64, "EFF_YR_BLT": pl.Int64,
        "SALE_PRC1": pl.Int64, "SALE_YR1": pl.Int64,
        "SALE_PRC2": pl.Int64, "SALE_YR2": pl.Int64,
        "OWN_ADDR2": pl.Utf8, "OWN_CITY": pl.Utf8, "OWN_STATE": pl.Utf8,
    }

    for f in files:
        df = pl.read_parquet(f)
        loaded = df.height
        grand_loaded += loaded

        # Some county parquets lack optional columns — backfill with nulls so
        # downstream score()/shape_output() can always select them.
        for col, dtype in expected_cols.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))

        df = df.filter(pl.col("DOR_UC").is_in(list(RESIDENTIAL_DOR)))
        scored = score(df)
        del df
        flagged = scored.height
        grand_flagged += flagged

        out_county = (
            shape_output(scored)
            .filter(pl.col("score") >= args.min_score)
        )
        del scored
        per_county_out.append(out_county)
        print(f"  {f.name}: {loaded:,} -> {flagged:,} flagged -> {out_county.height:,} >={args.min_score}")

    print(f"totals: loaded {grand_loaded:,} -> flagged {grand_flagged:,}")

    out = pl.concat(per_county_out, how="diagonal_relaxed").sort("score", descending=True)
    print(f"  combined output: {out.height:,} rows")

    # ── New-lead detection ─────────────────────────────────────────────────
    # Track the first run each parcel_id appeared in. `first_seen` is joined
    # onto the output so the dashboard can surface fresh inventory; parcels
    # that drop out of the lead set stay remembered in the snapshot.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen_path = args.output.parent / "seen_parcels.parquet"
    today = date.today().isoformat()
    if seen_path.exists():
        seen = pl.read_parquet(seen_path)
    else:
        seen = pl.DataFrame(schema={"parcel_id": pl.Utf8, "first_seen": pl.Utf8})

    out = out.with_columns(pl.col("parcel_id").cast(pl.Utf8))
    out = out.join(seen, on="parcel_id", how="left").with_columns(
        pl.col("first_seen").fill_null(today)
    )
    new_count = out.filter(pl.col("first_seen") == today).height
    updated_seen = pl.concat(
        [seen, out.select("parcel_id", "first_seen")], how="vertical_relaxed"
    ).unique(subset="parcel_id", keep="first")
    updated_seen.write_parquet(seen_path, compression="zstd")
    print(
        f"  new leads this run: {new_count:,}  "
        f"(seen-parcel snapshot: {updated_seen.height:,})"
    )

    if args.output.suffix.lower() == ".parquet":
        out.write_parquet(args.output, compression="zstd")
    else:
        out.write_csv(args.output)
    size_mb = args.output.stat().st_size / 1_000_000
    print(f"wrote {args.output} ({size_mb:.1f} MB, {out.height:,} leads)")


if __name__ == "__main__":
    main()
