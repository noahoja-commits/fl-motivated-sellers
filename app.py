"""Florida Motivated-Sellers Dashboard."""

from pathlib import Path

import polars as pl
import streamlit as st

st.set_page_config(
    page_title="FL Motivated Sellers",
    page_icon="🏚️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
LEADS_CSV = DATA_DIR / "leads.csv"
BY_OWNER = DATA_DIR / "by_owner.parquet"
BY_ADDR = DATA_DIR / "by_address.parquet"

CURRENT_YEAR = 2026

FLAG_DESCRIPTIONS = {
    "out_of_state": "Owner mailing address outside FL",
    "out_of_zip": "Owner mailing zip ≠ property zip",
    "po_box": "Mailing addr is a PO Box (often investor / absentee)",
    "long_held_25y": "Owned 25+ years (high equity, often tired landlord)",
    "trust_estate_name": "Owner is a trust or estate (inheritance / probate hint)",
    "multi_property_owner": "Same owner has multiple parcels",
}

CRM_COLUMNS = [
    "parcel_id",
    "owner_name",
    "owner_mailing",
    "situs_address",
    "situs_city",
    "situs_zip",
    "county_name",
    "just_value",
    "year_built",
    "living_area",
    "last_sale_year",
    "last_sale_price",
    "signal_type",
    "score",
    "flags",
]


@st.cache_data
def load_leads() -> pl.DataFrame:
    return pl.read_csv(LEADS_CSV)


@st.cache_data
def load_by_owner() -> pl.DataFrame:
    return pl.read_parquet(BY_OWNER)


@st.cache_data
def load_by_address() -> pl.DataFrame:
    return pl.read_parquet(BY_ADDR)


st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stMetric { background: #111; padding: 0.6rem 1rem; border-radius: 6px; border: 1px solid #222; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
    .dataframe td, .dataframe th { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
    .parcel-card { background: #111; border: 1px solid #2a2a2a; border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
    .parcel-card h4 { margin-top: 0; }
    .parcel-card .flagchip { display: inline-block; background: #1f3a1f; color: #9ee29e; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 11px; margin-right: 4px; font-family: monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏚️ Florida Motivated Sellers")
st.caption(
    "Pre-scored leads from the FL Department of Revenue NAL (Name-Address-Legal) statewide parcel file."
)

leads = load_leads()
by_owner = load_by_owner()
by_addr = load_by_address()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    counties_all = sorted(leads["county_name"].unique().to_list())
    selected_counties = st.multiselect(
        "Counties", counties_all, default=[], help="Empty = all counties"
    )

    min_value = int(leads["just_value"].min() or 0)
    max_value = int(leads["just_value"].max() or 0)
    value_range = st.slider(
        "Just value range ($)",
        min_value=min_value,
        max_value=max_value,
        value=(min_value, max_value),
        step=10_000,
    )

    min_score = st.slider("Min score", 0, 100, 75, step=25)

    st.markdown("**Required flags** (lead must have ALL)")
    require_flags = []
    for flag, desc in FLAG_DESCRIPTIONS.items():
        if st.checkbox(flag, key=f"req_{flag}", help=desc):
            require_flags.append(flag)

    st.markdown("**Exclude flags** (lead must have NONE)")
    exclude_flags = []
    for flag, desc in FLAG_DESCRIPTIONS.items():
        if st.checkbox(f"not {flag}", key=f"exc_{flag}", help=desc):
            exclude_flags.append(flag)

    st.divider()
    st.markdown("**Property characteristics**")

    multifamily_only = st.checkbox(
        "Multifamily only",
        help="Residential units ≥ 2 (DOR codes 003 / 008 / 009 also included)",
    )
    recent_buyer_stranded = st.checkbox(
        "Recent buyer underwater",
        help="Bought in the last 4 years at or above current just value — paid peak prices.",
    )
    pre_1980 = st.checkbox(
        "Pre-1980 build",
        help="Built before 1980 + no sale in last 25 years (deferred-maintenance proxy)",
    )


def apply_filters(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    if selected_counties:
        out = out.filter(pl.col("county_name").is_in(selected_counties))
    out = out.filter(
        (pl.col("just_value") >= value_range[0])
        & (pl.col("just_value") <= value_range[1])
        & (pl.col("score") >= min_score)
    )
    for f in require_flags:
        out = out.filter(pl.col("flags").str.contains(f))
    for f in exclude_flags:
        out = out.filter(~pl.col("flags").str.contains(f))

    if multifamily_only:
        out = out.filter(
            (pl.col("residential_units").fill_null(0) >= 2)
            | (pl.col("dor_uc").is_in(["003", "008", "009"]))
        )
    if recent_buyer_stranded:
        out = out.filter(
            (pl.col("last_sale_year") >= CURRENT_YEAR - 4)
            & (pl.col("last_sale_price").fill_null(0) >= pl.col("just_value"))
            & (pl.col("last_sale_price").fill_null(0) > 0)
        )
    if pre_1980:
        out = out.filter(
            (pl.col("year_built").fill_null(0) > 0)
            & (pl.col("year_built") < 1980)
            & (
                pl.col("last_sale_year").fill_null(0) < CURRENT_YEAR - 25
            )
        )
    return out


filtered = apply_filters(leads)

# ── KPI strip ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Leads (filtered)", f"{filtered.height:,}")
k2.metric("Total just-value", f"${int(filtered['just_value'].sum() or 0):,}")
k3.metric("Avg just-value", f"${int(filtered['just_value'].mean() or 0):,}")
k4.metric("Counties touched", f"{filtered['county_name'].n_unique()}")

# ── Tabs ───────────────────────────────────────────────────────────────────────
t_leads, t_owners, t_addr, t_lookup, t_about = st.tabs(
    [
        "📋 Scored leads",
        "👤 Multi-property owners",
        "🏢 Address clusters",
        "🔍 Parcel lookup",
        "ℹ️ About",
    ]
)

with t_leads:
    st.subheader("Absentee-equity leads")
    st.caption(
        f"Showing {filtered.height:,} of {leads.height:,} scored leads."
    )

    display = filtered.select(
        [
            "score",
            "county_name",
            "situs_address",
            "situs_city",
            "situs_zip",
            "owner_name",
            "owner_mailing",
            "owner_state",
            "just_value",
            "year_built",
            "residential_units",
            "last_sale_year",
            "flags",
            "parcel_id",
        ]
    ).sort("score", descending=True)

    st.dataframe(display, use_container_width=True, height=520, hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Download filtered (full columns)",
        display.write_csv(),
        file_name="motivated_sellers_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
    crm_ready = (
        filtered.select([c for c in CRM_COLUMNS if c in filtered.columns])
        .sort("score", descending=True)
    )
    d2.download_button(
        "⬇️ CRM-ready CSV (acquisitions-crm import format)",
        crm_ready.write_csv(),
        file_name="acquisitions_crm_import.csv",
        mime="text/csv",
        use_container_width=True,
    )

with t_owners:
    st.subheader("Owners holding multiple parcels")
    st.caption(
        f"{by_owner.height:,} owner entities holding 5+ FL parcels each. Statewide."
    )

    min_parcels = st.slider(
        "Min parcels per owner",
        int(by_owner["parcel_count"].min()),
        int(by_owner["parcel_count"].max()),
        5,
    )
    owner_view = (
        by_owner.filter(pl.col("parcel_count") >= min_parcels)
        .select(
            [
                "owner_name_example",
                "parcel_count",
                "total_just_value",
                "county_count",
                "mailing_addr_example",
            ]
        )
        .rename(
            {
                "owner_name_example": "owner_name",
                "mailing_addr_example": "mailing_addr",
            }
        )
        .sort("parcel_count", descending=True)
    )

    st.dataframe(owner_view, use_container_width=True, height=500, hide_index=True)
    st.download_button(
        "⬇️ Download owner list",
        owner_view.write_csv(),
        file_name="multi_property_owners.csv",
        mime="text/csv",
    )

with t_addr:
    st.subheader("Mailing-address clusters")
    st.caption(
        f"{by_addr.height:,} mailing addresses that show up on 5+ parcels each — property managers, large landlords, multifamily ownership entities."
    )

    min_a_parcels = st.slider(
        "Min parcels per address",
        int(by_addr["parcel_count"].min()),
        int(by_addr["parcel_count"].max()),
        5,
        key="addr_min",
    )
    addr_view = (
        by_addr.filter(pl.col("parcel_count") >= min_a_parcels)
        .select(
            [
                "addr_norm",
                "parcel_count",
                "distinct_owner_names",
                "total_just_value",
                "county_count",
            ]
        )
        .sort("parcel_count", descending=True)
    )

    st.dataframe(addr_view, use_container_width=True, height=500, hide_index=True)
    st.download_button(
        "⬇️ Download address clusters",
        addr_view.write_csv(),
        file_name="address_clusters.csv",
        mime="text/csv",
    )

with t_lookup:
    st.subheader("Look up a parcel")
    st.caption("Paste a parcel ID from the Scored Leads table.")

    pid = st.text_input("Parcel ID", value="", placeholder="e.g. 0258080015")
    if pid.strip():
        match = leads.filter(pl.col("parcel_id") == pid.strip())
        if match.is_empty():
            st.warning("No scored lead found for that parcel ID. (Only flagged parcels are in the dashboard.)")
        else:
            row = match.row(0, named=True)
            flag_html = "".join(f'<span class="flagchip">{f}</span>' for f in row["flags"].split(",") if f)

            st.markdown(
                f"""
                <div class="parcel-card">
                <h4>{row['situs_address']}, {row['situs_city']} {row['situs_zip']}</h4>
                <div>{flag_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score", row["score"])
            c2.metric("Just value", f"${int(row['just_value'] or 0):,}")
            c3.metric("Year built", row["year_built"] or "—")
            c4.metric("Living area", f"{int(row['living_area'] or 0):,} sqft" if row["living_area"] else "—")

            st.markdown("**Owner**")
            st.write(f"`{row['owner_name']}`  \n{row['owner_mailing']}")

            st.markdown("**Sale history**")
            sale_rows = []
            if row.get("last_sale_year"):
                sale_rows.append(
                    {
                        "year": row["last_sale_year"],
                        "price": f"${int(row['last_sale_price'] or 0):,}" if row.get("last_sale_price") else "—",
                        "rank": "most recent",
                    }
                )
            if row.get("prior_sale_year"):
                sale_rows.append(
                    {
                        "year": row["prior_sale_year"],
                        "price": f"${int(row['prior_sale_price'] or 0):,}" if row.get("prior_sale_price") else "—",
                        "rank": "prior",
                    }
                )
            if sale_rows:
                st.dataframe(pl.DataFrame(sale_rows), hide_index=True, use_container_width=True)
            else:
                st.caption("No sale history recorded.")

            st.markdown("**Property details**")
            details = {
                "Parcel ID": row["parcel_id"],
                "County": row["county_name"],
                "DOR use code": row.get("dor_uc") or "—",
                "Residential units": row.get("residential_units") or "—",
                "Owner state": row["owner_state"],
                "Signal type": row["signal_type"],
            }
            st.dataframe(
                pl.DataFrame({"field": list(details.keys()), "value": [str(v) for v in details.values()]}),
                hide_index=True,
                use_container_width=True,
            )

with t_about:
    st.markdown(
        """
### What this is

A Streamlit dashboard over Florida's statewide parcel data (NAL — Name, Address, Legal).
Every residential property in Florida appears in the upstream data; this dashboard
surfaces the ones whose owner/sale patterns match motivated-seller signals.

### What's been done before you see it

1. Downloaded all 67 FL county NAL files from FL DOR
2. Parsed and normalized owners/addresses
3. Filtered to residential DOR codes
4. Scored against the signal flags below
5. Joined into the aggregated owner and address files

### Signal flags

| Flag | Meaning |
|---|---|
"""
        + "\n".join(f"| `{f}` | {d} |" for f, d in FLAG_DESCRIPTIONS.items())
        + """

### Score

Each of the four main flags (out_of_state-or-out_of_zip, po_box, long_held_25y,
trust_estate_name) is worth 25 points — so a parcel with all four = 100. Only
parcels scoring ≥ 75 are shipped in the dashboard data file.
`multi_property_owner` is informational and doesn't move the score.

### Property-characteristic filters

- **Multifamily only** — units ≥ 2 or DOR use code 003/008/009
- **Recent buyer underwater** — bought in last 4 yrs at ≥ today's just value
- **Pre-1980 build** — built before 1980, no sale in 25+ years (deferred-maintenance proxy)
"""
    )
