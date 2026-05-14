"""
Florida Motivated-Sellers Dashboard
Streamlit UI over pre-computed NAL lead aggregates.
"""

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

FLAG_DESCRIPTIONS = {
    "out_of_state": "Owner mailing address outside FL",
    "out_of_zip": "Owner mailing zip ≠ property zip",
    "po_box": "Mailing addr is a PO Box (often investor / absentee)",
    "long_held_25y": "Owned 25+ years (high equity, often tired landlord)",
    "trust_estate_name": "Owner is a trust or estate (inheritance / probate hint)",
    "multi_property_owner": "Same owner has multiple parcels",
}


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
        "Counties",
        counties_all,
        default=[],
        help="Empty = all counties",
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

    min_score = st.slider("Min score", 0, 100, 0, step=5)

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
    return out


filtered = apply_filters(leads)

# ── KPI strip ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Leads (filtered)", f"{filtered.height:,}")
k2.metric(
    "Total just-value",
    f"${int(filtered['just_value'].sum() or 0):,}",
)
k3.metric(
    "Avg just-value",
    f"${int(filtered['just_value'].mean() or 0):,}",
)
k4.metric("Counties touched", f"{filtered['county_name'].n_unique()}")

# ── Tabs ───────────────────────────────────────────────────────────────────────
t_leads, t_owners, t_addr, t_about = st.tabs(
    ["📋 Scored leads", "👤 Multi-property owners", "🏢 Address clusters", "ℹ️ About"]
)

with t_leads:
    st.subheader("Absentee-equity leads")
    st.caption(
        f"Showing {filtered.height:,} of {leads.height:,} scored leads. Sort & filter by clicking column headers."
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
            "flags",
            "parcel_id",
        ]
    ).sort("score", descending=True)

    st.dataframe(display, use_container_width=True, height=500, hide_index=True)

    csv = display.write_csv()
    st.download_button(
        "⬇️ Download filtered leads as CSV",
        csv,
        file_name="motivated_sellers_filtered.csv",
        mime="text/csv",
    )

with t_owners:
    st.subheader("Owners holding multiple parcels")
    st.caption(
        f"{by_owner.height:,} owner entities holding 5+ FL parcels each. Sort by parcel count or total value."
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

with t_about:
    st.markdown(
        """
### What this is

A Streamlit dashboard over Florida's statewide parcel data (NAL — Name, Address, Legal).
Every property in Florida appears here, owners + assessed values included. The state
publishes it as free annual bulk files.

### What's been done before you see it

The 67 county NAL files have been:
1. Downloaded from FL Department of Revenue
2. Parsed and normalized (owner names, addresses)
3. Scored against motivated-seller signals
4. Aggregated by owner name and mailing address

This dashboard reads those pre-computed outputs.

### Signal flags

| Flag | Meaning |
|---|---|
"""
        + "\n".join(f"| `{f}` | {d} |" for f, d in FLAG_DESCRIPTIONS.items())
        + """

### What "score" means

100 = strong motivated-seller signal stack (multiple flags overlap, e.g. out-of-state
+ long-held + trust). Lower scores have fewer overlapping signals. Use the
**Required flags** sidebar to drill into specific lead profiles.
"""
    )
