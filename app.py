"""Florida Motivated-Sellers Dashboard."""

from pathlib import Path
from urllib.parse import quote_plus

import polars as pl
import streamlit as st

import crm_push
import skip_trace

st.set_page_config(
    page_title="FL Motivated Sellers",
    page_icon="🏚️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
LEADS_PATH = DATA_DIR / "leads.parquet"
BY_OWNER = DATA_DIR / "by_owner.parquet"
BY_ADDR = DATA_DIR / "by_address.parquet"

CURRENT_YEAR = 2026

FLAG_DESCRIPTIONS = {
    "out_of_state": "Owner mailing address outside FL",
    "out_of_zip": "Owner mailing zip ≠ property zip",
    "po_box": "Mailing addr is a PO Box (often investor / absentee)",
    "long_held_25y": "Owned 25+ years (high equity, often tired landlord)",
    "trust_estate_name": "Owner is a trust or estate (inheritance / probate hint)",
    "multi_property_owner": "Same normalized owner has multiple parcels",
    "bank_trustee": "Owner is a bank, REO, or mortgage trustee (foreclosed assets)",
    "sale_anomaly": "Last sale was <50% or >150% of just value (distressed buy or peak overpayer)",
    "high_equity_proxy": "Bought for <40% of current value, or no sale + 25+ yr old building",
}

CRM_COLUMNS = [
    "parcel_id", "owner_name", "owner_mailing", "situs_address", "situs_city",
    "situs_zip", "county_name", "just_value", "year_built", "living_area",
    "last_sale_year", "last_sale_price", "signal_type", "score", "flags",
]


@st.cache_data
def load_leads() -> pl.DataFrame:
    return pl.read_parquet(LEADS_PATH)


@st.cache_data
def load_by_owner() -> pl.DataFrame:
    return pl.read_parquet(BY_OWNER)


@st.cache_data
def load_by_address() -> pl.DataFrame:
    return pl.read_parquet(BY_ADDR)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_sunbiz_search(name: str) -> str:
    return skip_trace.sunbiz_search_by_name(name)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_sunbiz_detail(cor_number: str) -> dict:
    return skip_trace.sunbiz_detail(cor_number)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_ddg(query: str) -> dict:
    return skip_trace.duckduckgo_lookup(query)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px; }
    h1, h2, h3, h4 { font-family: 'Inter', system-ui, sans-serif; letter-spacing: -0.01em; }
    h1 { font-weight: 600; margin-bottom: 0.2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 14px; }
    .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 16px 0 24px; }
    .kpi-card { background: #141414; border: 1px solid #232323; border-radius: 8px; padding: 14px 16px; }
    .kpi-label { color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }
    .kpi-value { color: #f4f4f4; font-size: 22px; font-weight: 600; margin-top: 4px; }
    .kpi-sub { color: #6e6e6e; font-size: 11px; margin-top: 2px; }
    .flagchip { display: inline-block; background: #1f3a1f; color: #9ee29e; padding: 2px 8px;
                border-radius: 4px; font-size: 11px; margin: 0 4px 4px 0; font-family: monospace; }
    .parcel-header { background: #141414; border: 1px solid #232323; border-radius: 8px;
                     padding: 16px 20px; margin-bottom: 18px; }
    .parcel-header .addr { font-size: 18px; font-weight: 600; color: #f4f4f4; }
    .parcel-header .meta { color: #888; font-size: 13px; margin-top: 4px; }
    .skip-section { background: #0e0e0e; border: 1px solid #1f1f1f; border-radius: 8px;
                    padding: 14px 18px; margin-top: 12px; }
    .skip-section .label { color: #888; font-size: 11px; text-transform: uppercase;
                           letter-spacing: 0.06em; margin-bottom: 6px; }
    .skip-result-key { color: #888; font-size: 12px; }
    .skip-result-val { color: #e6e6e6; font-family: monospace; font-size: 13px; }
    a.link-pill { display: inline-block; background: #1a1a1a; color: #9ee29e;
                  padding: 6px 12px; border-radius: 6px; text-decoration: none;
                  margin: 0 6px 6px 0; font-size: 13px; border: 1px solid #2a2a2a; }
    a.link-pill:hover { background: #1f3a1f; color: #cfe; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Florida Motivated Sellers")
st.caption("Pre-scored leads from the FL Department of Revenue NAL parcel file.")

leads = load_leads()
by_owner = load_by_owner()
by_addr = load_by_address()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")

    with st.expander("Counties & value", expanded=True):
        counties_all = sorted(leads["county_name"].unique().to_list())
        selected_counties = st.multiselect(
            "Counties", counties_all, default=[],
            placeholder="All counties",
        )
        min_value = int(leads["just_value"].min() or 0)
        max_value = int(leads["just_value"].max() or 0)
        value_range = st.slider(
            "Just value ($)",
            min_value=min_value, max_value=max_value,
            value=(min_value, max_value), step=10_000,
            format="$%d",
        )
        min_score = st.slider("Min score", 0, 100, 75, step=25)

    with st.expander("Flag requirements"):
        st.caption("Require ALL checked flags")
        require_flags = [
            flag for flag, desc in FLAG_DESCRIPTIONS.items()
            if st.checkbox(flag, key=f"req_{flag}", help=desc)
        ]
        st.caption("Exclude any of these flags")
        exclude_flags = [
            flag for flag, desc in FLAG_DESCRIPTIONS.items()
            if st.checkbox(f"not {flag}", key=f"exc_{flag}", help=desc)
        ]

    with st.expander("Property characteristics"):
        multifamily_only = st.checkbox(
            "Multifamily only",
            help="Residential units ≥ 2 (DOR codes 003 / 008 / 009 also included)",
        )
        recent_buyer_stranded = st.checkbox(
            "Recent buyer underwater",
            help="Bought in last 4 yrs at ≥ current just value",
        )
        pre_1980 = st.checkbox(
            "Pre-1980 build + 25+ yr held",
            help="Deferred-maintenance proxy",
        )
        entity_only = st.checkbox(
            "LLC / entity owners only",
            help="Owner is an LLC, trust, or other entity (not an individual).",
        )

    st.markdown("---")
    st.markdown("### CRM push")
    default_token = ""
    default_url = crm_push.DEFAULT_BASE_URL
    try:
        default_token = st.secrets["crm"]["capture_token"]
    except Exception:
        pass
    try:
        default_url = st.secrets["crm"]["base_url"]
    except Exception:
        pass

    crm_base_url = st.text_input("CRM base URL", value=default_url, key="crm_base_url")
    crm_token = st.text_input(
        "CAPTURE_TOKEN",
        value=default_token,
        type="password",
        help="Bearer token for /api/capture. Set crm.capture_token in Streamlit secrets to persist.",
        key="crm_token",
    )
    crm_ready = bool(crm_token.strip())
    if not crm_ready:
        st.caption("Paste a token above to enable push-to-CRM buttons.")


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
            & (pl.col("last_sale_year").fill_null(0) < CURRENT_YEAR - 25)
        )
    if entity_only:
        out = out.filter(pl.col("is_entity"))
    return out


filtered = apply_filters(leads)

# ── KPI strip ──────────────────────────────────────────────────────────────────
def fmt_money(n: int | float) -> str:
    n = int(n or 0)
    if abs(n) >= 1_000_000_000:
        return f"${n / 1_000_000_000:.1f}B"
    if abs(n) >= 1_000_000:
        return f"${n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"${n / 1_000:.0f}K"
    return f"${n:,}"


kpi_html = f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">Leads</div>
    <div class="kpi-value">{filtered.height:,}</div>
    <div class="kpi-sub">filtered of {leads.height:,}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Total value</div>
    <div class="kpi-value">{fmt_money(filtered['just_value'].sum() or 0)}</div>
    <div class="kpi-sub">sum of just-values</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Avg value</div>
    <div class="kpi-value">{fmt_money(filtered['just_value'].mean() or 0)}</div>
    <div class="kpi-sub">per parcel</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Counties</div>
    <div class="kpi-value">{filtered['county_name'].n_unique()}</div>
    <div class="kpi-sub">touched by filters</div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
t_leads, t_owners, t_addr, t_lookup, t_analytics, t_about = st.tabs(
    ["Scored leads", "Multi-property owners", "Address clusters",
     "Parcel lookup", "Analytics", "About"]
)

LEAD_COL_CONFIG = {
    "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
    "county_name": st.column_config.TextColumn("County", width="small"),
    "situs_address": st.column_config.TextColumn("Address", width="medium"),
    "situs_city": st.column_config.TextColumn("City", width="small"),
    "situs_zip": st.column_config.TextColumn("Zip", width="small"),
    "owner_name": st.column_config.TextColumn("Owner", width="medium"),
    "owner_mailing": st.column_config.TextColumn("Mailing", width="medium"),
    "owner_state": st.column_config.TextColumn("State", width="small"),
    "just_value": st.column_config.NumberColumn("Just $", format="$%d"),
    "year_built": st.column_config.NumberColumn("Built", format="%d"),
    "residential_units": st.column_config.NumberColumn("Units", format="%d"),
    "last_sale_year": st.column_config.NumberColumn("Last sale", format="%d"),
    "last_sale_price": st.column_config.NumberColumn("Sale $", format="$%d"),
    "flags": st.column_config.TextColumn("Flags", width="medium"),
    "parcel_id": st.column_config.TextColumn("Parcel ID", width="small"),
}

with t_leads:
    st.markdown(f"##### {filtered.height:,} matching leads")

    if filtered.is_empty():
        st.info("No leads match these filters. Loosen the sidebar.")
    else:
        display = filtered.select([
            "score", "county_name", "situs_address", "situs_city", "situs_zip",
            "owner_name", "owner_mailing", "owner_state", "just_value",
            "year_built", "residential_units", "last_sale_year", "flags", "parcel_id",
        ]).sort("score", descending=True)

        st.dataframe(
            display,
            use_container_width=True,
            height=520,
            hide_index=True,
            column_config=LEAD_COL_CONFIG,
        )

        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "⬇️ Download filtered CSV",
            display.write_csv(),
            file_name="motivated_sellers_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
        crm_ready_csv = (
            filtered.select([c for c in CRM_COLUMNS if c in filtered.columns])
            .sort("score", descending=True)
        )
        d2.download_button(
            "⬇️ CRM-ready CSV",
            crm_ready_csv.write_csv(),
            file_name="acquisitions_crm_import.csv",
            mime="text/csv",
            use_container_width=True,
        )
        push_n = min(filtered.height, 200)
        push_clicked = d3.button(
            f"🚀 Push top {push_n} to CRM",
            disabled=not crm_ready or filtered.is_empty(),
            help="POST the highest-score filtered rows to the CRM /api/capture endpoint."
                 " Capped at 200/click for safety.",
            use_container_width=True,
        )
        if push_clicked and crm_ready:
            rows = filtered.sort("score", descending=True).head(push_n).to_dicts()
            progress = st.progress(0, text=f"pushing 0/{push_n} …")
            status_box = st.empty()

            def cb(done: int, total: int, last: dict) -> None:
                progress.progress(done / total, text=f"pushing {done}/{total} …")

            stats = crm_push.push_batch(rows, crm_base_url, crm_token, progress_cb=cb)
            progress.empty()
            if stats["failed"] == 0:
                status_box.success(
                    f"✅ Pushed {stats['total']} — created {stats['created']}, "
                    f"updated {stats['other_ok']}."
                )
            else:
                status_box.warning(
                    f"Pushed {stats['total']} — created {stats['created']}, "
                    f"updated {stats['other_ok']}, **failed {stats['failed']}**."
                )
                if stats["errors"]:
                    with st.expander("First few errors"):
                        for e in stats["errors"]:
                            st.code(e)

with t_owners:
    st.markdown(f"##### Owners holding multiple parcels")
    st.caption(f"{by_owner.height:,} entities holding 5+ FL parcels each. Statewide.")

    min_parcels = st.slider(
        "Min parcels per owner",
        int(by_owner["parcel_count"].min()),
        int(by_owner["parcel_count"].max()),
        5,
    )
    owner_view = (
        by_owner.filter(pl.col("parcel_count") >= min_parcels)
        .select([
            "owner_name_example", "parcel_count", "total_just_value",
            "county_count", "mailing_addr_example",
        ])
        .rename({
            "owner_name_example": "owner_name",
            "mailing_addr_example": "mailing_addr",
        })
        .sort("parcel_count", descending=True)
    )
    st.dataframe(
        owner_view,
        use_container_width=True,
        height=500,
        hide_index=True,
        column_config={
            "owner_name": st.column_config.TextColumn("Owner", width="medium"),
            "parcel_count": st.column_config.NumberColumn("Parcels", format="%d"),
            "total_just_value": st.column_config.NumberColumn("Total value", format="$%d"),
            "county_count": st.column_config.NumberColumn("Counties", format="%d"),
            "mailing_addr": st.column_config.TextColumn("Mailing addr", width="large"),
        },
    )
    st.download_button(
        "⬇️ Download owner list",
        owner_view.write_csv(),
        file_name="multi_property_owners.csv",
        mime="text/csv",
    )

with t_addr:
    st.markdown("##### Mailing-address clusters")
    st.caption(
        f"{by_addr.height:,} mailing addresses tied to 5+ parcels each — property managers, "
        f"big landlords, multifamily ownership entities."
    )
    min_a_parcels = st.slider(
        "Min parcels per address",
        int(by_addr["parcel_count"].min()),
        int(by_addr["parcel_count"].max()),
        5, key="addr_min",
    )
    addr_view = (
        by_addr.filter(pl.col("parcel_count") >= min_a_parcels)
        .select([
            "addr_norm", "parcel_count", "distinct_owner_names",
            "total_just_value", "county_count",
        ])
        .sort("parcel_count", descending=True)
    )
    st.dataframe(
        addr_view,
        use_container_width=True,
        height=500,
        hide_index=True,
        column_config={
            "addr_norm": st.column_config.TextColumn("Address", width="large"),
            "parcel_count": st.column_config.NumberColumn("Parcels", format="%d"),
            "distinct_owner_names": st.column_config.NumberColumn("Distinct owners", format="%d"),
            "total_just_value": st.column_config.NumberColumn("Total value", format="$%d"),
            "county_count": st.column_config.NumberColumn("Counties", format="%d"),
        },
    )
    st.download_button(
        "⬇️ Download address clusters",
        addr_view.write_csv(),
        file_name="address_clusters.csv",
        mime="text/csv",
    )


def render_link_pill(label: str, url: str) -> str:
    return f'<a class="link-pill" href="{url}" target="_blank" rel="noopener">{label}</a>'


with t_lookup:
    st.markdown("##### Parcel lookup")
    pid = st.text_input(
        "Parcel ID",
        value="",
        placeholder="paste a parcel ID from the Scored Leads tab",
        label_visibility="collapsed",
    )

    if not pid.strip():
        st.info("Enter a parcel ID above to see owner, sale history, and skip-trace tools.")
    else:
        match = leads.filter(pl.col("parcel_id") == pid.strip())
        if match.is_empty():
            st.warning("No scored lead found for that parcel ID. (Only flagged parcels are in the dashboard.)")
        else:
            row = match.row(0, named=True)
            flag_html = "".join(
                f'<span class="flagchip">{f}</span>'
                for f in (row["flags"] or "").split(",") if f
            )
            st.markdown(
                f"""
                <div class="parcel-header">
                  <div class="addr">{row['situs_address'] or '—'}</div>
                  <div class="meta">{row['situs_city']}, FL {row['situs_zip']} · {row['county_name']} County · parcel {row['parcel_id']}</div>
                  <div style="margin-top:8px">{flag_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score", row["score"])
            c2.metric("Just value", fmt_money(row["just_value"]))
            c3.metric("Year built", row["year_built"] or "—")
            c4.metric("Living area", f"{int(row['living_area'] or 0):,} sf" if row["living_area"] else "—")

            owner_col, sale_col = st.columns([1, 1])

            with owner_col:
                st.markdown("**Owner**")
                st.write(f"`{row['owner_name']}`")
                st.caption(row["owner_mailing"])
                if row["owner_state"]:
                    st.caption(f"State on file: {row['owner_state']}")

            with sale_col:
                st.markdown("**Sale history**")
                sales = []
                if row.get("last_sale_year"):
                    sales.append({
                        "year": int(row["last_sale_year"]),
                        "price": int(row["last_sale_price"] or 0),
                        "rank": "most recent",
                    })
                if row.get("prior_sale_year"):
                    sales.append({
                        "year": int(row["prior_sale_year"]),
                        "price": int(row["prior_sale_price"] or 0),
                        "rank": "prior",
                    })
                if sales:
                    st.dataframe(
                        pl.DataFrame(sales),
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "year": st.column_config.NumberColumn("Year", format="%d"),
                            "price": st.column_config.NumberColumn("Price", format="$%d"),
                            "rank": st.column_config.TextColumn("Which"),
                        },
                    )
                else:
                    st.caption("No sale history recorded.")

            # ── Skip trace ─────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("##### 🕵️ Skip trace")

            is_entity = bool(row.get("is_entity"))

            # Per-parcel push to CRM
            push_col, _ = st.columns([2, 5])
            push_one_clicked = push_col.button(
                "🚀 Push this lead to CRM",
                disabled=not crm_ready,
                help="POST this single parcel to /api/capture",
                use_container_width=True,
                key=f"push_one_{row['parcel_id']}",
            )
            if push_one_clicked and crm_ready:
                with st.spinner("Pushing to CRM…"):
                    res = crm_push.push_one(dict(row), crm_base_url, crm_token)
                if res.get("ok"):
                    pid = res.get("propertyId") or res.get("leadId") or ""
                    st.success(f"✅ pushed (HTTP {res.get('status')}) — id: {pid}")
                else:
                    st.error(
                        f"❌ failed: {res.get('error') or res.get('details') or res.get('status')}"
                    )

            colA, colB = st.columns(2)
            run_sunbiz = colA.button(
                "🏛️ Sunbiz live lookup",
                disabled=not is_entity,
                help=(
                    "Searches Sunbiz by owner name, then scrapes the entity detail page "
                    "for registered agent, email, phone, status."
                    if is_entity else
                    "Disabled — owner doesn't look like a business entity."
                ),
                use_container_width=True,
            )
            run_ddg = colB.button(
                "🔎 DuckDuckGo search",
                help="Free web search for owner + city. Pulls website, socials, email, phone from results.",
                use_container_width=True,
            )

            if run_sunbiz and is_entity:
                with st.spinner("Searching Sunbiz…"):
                    cor_number = cached_sunbiz_search(row["owner_name"])
                    if not cor_number:
                        st.warning("Sunbiz: no matching corporation found by that name.")
                    else:
                        detail = cached_sunbiz_detail(cor_number)
                        if not detail.get("page_found"):
                            st.warning("Sunbiz: entity number resolved but detail page didn't load.")
                        else:
                            st.markdown('<div class="skip-section">', unsafe_allow_html=True)
                            st.markdown(f'<div class="label">Sunbiz · {cor_number}</div>', unsafe_allow_html=True)
                            fields = {
                                "Status": detail.get("live_status") or "—",
                                "Last annual report": detail.get("last_report_year") or "—",
                                "Registered agent": detail.get("registered_agent") or "—",
                                "Email": detail.get("email") or "—",
                                "Phone": detail.get("phone") or "—",
                            }
                            for k, v in fields.items():
                                st.markdown(
                                    f'<div><span class="skip-result-key">{k}:</span> '
                                    f'<span class="skip-result-val">{v}</span></div>',
                                    unsafe_allow_html=True,
                                )
                            st.markdown("</div>", unsafe_allow_html=True)

            if run_ddg:
                with st.spinner("Searching the web…"):
                    q = f'"{row["owner_name"]}" {row["situs_city"]} FL'
                    res = cached_ddg(q)
                    st.markdown('<div class="skip-section">', unsafe_allow_html=True)
                    st.markdown(f'<div class="label">Web · "{q}"</div>', unsafe_allow_html=True)
                    fields = {
                        "Website": res.get("website") or "—",
                        "LinkedIn": res.get("linkedin") or "—",
                        "Facebook": res.get("facebook") or "—",
                        "Instagram": res.get("instagram") or "—",
                        "Email": res.get("email") or "—",
                        "Phone": res.get("phone") or "—",
                    }
                    for k, v in fields.items():
                        if v.startswith("http"):
                            st.markdown(
                                f'<div><span class="skip-result-key">{k}:</span> '
                                f'<a href="{v}" target="_blank" class="skip-result-val">{v}</a></div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div><span class="skip-result-key">{k}:</span> '
                                f'<span class="skip-result-val">{v}</span></div>',
                                unsafe_allow_html=True,
                            )
                    if res.get("snippets"):
                        st.markdown('<div class="label" style="margin-top:8px">Top snippets</div>',
                                    unsafe_allow_html=True)
                        for s in res["snippets"][:5]:
                            st.caption(f"• {s}")
                    st.markdown("</div>", unsafe_allow_html=True)

            # ── Manual lookup links ────────────────────────────────────────────
            st.markdown("---")
            st.markdown("##### 🔗 Quick links")
            owner_q = quote_plus(row["owner_name"] or "")
            city_q = quote_plus(row["situs_city"] or "")
            owner_name_q = quote_plus(f"{row['owner_name']} {row['situs_city']} FL")
            owner_first = quote_plus((row["owner_name"] or "").split()[0]) if row["owner_name"] else ""
            pa_q = quote_plus(f"{row['county_name']} county property appraiser")

            links = [
                ("Google: owner + city", f"https://www.google.com/search?q={owner_name_q}"),
                ("Sunbiz: name search", f"https://search.sunbiz.org/Inquiry/CorporationSearch/SearchResults?inquiryType=EntityName&inquiryDirective=StartsWith&searchNameOrder={owner_q}&searchTerm={owner_q}"),
                ("Property appraiser", f"https://www.google.com/search?q={pa_q}+{quote_plus(row['parcel_id'])}"),
                ("TruePeopleSearch", f"https://www.truepeoplesearch.com/results?name={owner_q}&citystatezip={city_q}%2C+FL"),
                ("FastPeopleSearch", f"https://www.fastpeoplesearch.com/name/{owner_q.replace('+', '-').lower()}_{city_q.lower()}-fl"),
                ("BeenVerified", f"https://www.beenverified.com/people/search/?fn={owner_first}&ln=&state=FL&city={city_q}"),
                ("Open in Maps", f"https://www.google.com/maps/search/{quote_plus(row['situs_address'] or '')}+{quote_plus(row['situs_city'] or '')}+FL+{quote_plus(row['situs_zip'] or '')}"),
            ]
            st.markdown(
                " ".join(render_link_pill(label, url) for label, url in links),
                unsafe_allow_html=True,
            )

with t_analytics:
    st.markdown("##### Lead distribution")
    st.caption(
        f"Charts reflect your current sidebar filters — {filtered.height:,} of "
        f"{leads.height:,} total leads."
    )

    if filtered.is_empty():
        st.info("No leads match the current filters.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Score breakdown**")
            score_dist = (
                filtered.group_by("score").agg(pl.len().alias("count"))
                .sort("score", descending=True)
                .to_pandas().set_index("score")
            )
            st.bar_chart(score_dist, horizontal=True, color="#9ee29e")

            st.markdown("**Top counties by lead count**")
            top_counties = (
                filtered.group_by("county_name").agg(pl.len().alias("count"))
                .sort("count", descending=True).head(15)
                .to_pandas().set_index("county_name")
            )
            st.bar_chart(top_counties, color="#9ee29e")

        with col_right:
            st.markdown("**Flag frequency**")
            flag_counts: dict[str, int] = {f: 0 for f in FLAG_DESCRIPTIONS}
            for s in filtered["flags"].to_list():
                for f in (s or "").split(","):
                    if f in flag_counts:
                        flag_counts[f] += 1
            flag_df = pl.DataFrame(
                {"flag": list(flag_counts.keys()),
                 "count": list(flag_counts.values())}
            ).sort("count", descending=True).to_pandas().set_index("flag")
            st.bar_chart(flag_df, horizontal=True, color="#9ee29e")

            st.markdown("**Just-value distribution**")
            jv_buckets = (
                filtered.with_columns(
                    pl.when(pl.col("just_value") < 100_000).then(pl.lit("< $100K"))
                    .when(pl.col("just_value") < 200_000).then(pl.lit("$100–200K"))
                    .when(pl.col("just_value") < 350_000).then(pl.lit("$200–350K"))
                    .when(pl.col("just_value") < 500_000).then(pl.lit("$350–500K"))
                    .when(pl.col("just_value") < 750_000).then(pl.lit("$500–750K"))
                    .when(pl.col("just_value") < 1_000_000).then(pl.lit("$750K–1M"))
                    .otherwise(pl.lit("$1M+"))
                    .alias("bucket")
                )
                .group_by("bucket").agg(pl.len().alias("count"))
                .to_pandas().set_index("bucket")
                .reindex(["< $100K", "$100–200K", "$200–350K", "$350–500K",
                          "$500–750K", "$750K–1M", "$1M+"])
            )
            st.bar_chart(jv_buckets, color="#9ee29e")

        st.markdown("---")
        st.markdown("**Top 25 owner entities by parcel count (in filtered set)**")
        top_owners = (
            filtered.filter(pl.col("is_entity"))
            .group_by("owner_norm")
            .agg(
                pl.len().alias("parcels"),
                pl.col("just_value").sum().alias("total_value"),
                pl.col("county_name").n_unique().alias("counties"),
                pl.col("owner_name").first().alias("example_owner"),
            )
            .sort("parcels", descending=True).head(25)
        )
        st.dataframe(
            top_owners,
            use_container_width=True,
            hide_index=True,
            column_config={
                "example_owner": st.column_config.TextColumn("Owner", width="large"),
                "parcels": st.column_config.NumberColumn("Parcels", format="%d"),
                "total_value": st.column_config.NumberColumn("Total value", format="$%d"),
                "counties": st.column_config.NumberColumn("Counties", format="%d"),
                "owner_norm": st.column_config.TextColumn("Normalized", width="medium"),
            },
        )


with t_about:
    st.markdown(
        """
### What this is

A dashboard over Florida's statewide parcel data (NAL — Name, Address, Legal).
Every residential property in Florida appears in the upstream data; this surfaces
the ones whose owner/sale patterns match motivated-seller signals.

### What's been done before you see it

1. Downloaded all 67 FL county NAL files from FL Department of Revenue
2. Parsed and normalized owners + mailing addresses
3. Filtered to residential DOR codes (001, 002, 004, 005, 006, 008)
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
parcels scoring ≥ 75 are shipped in the data file (~41k leads across six counties).
`multi_property_owner` is informational and doesn't move the score.

### Property-characteristic filters

- **Multifamily only** — units ≥ 2 or DOR use code 003 / 008 / 009
- **Recent buyer underwater** — bought in last 4 yrs at ≥ today's just value
- **Pre-1980 build + 25+ yr held** — deferred-maintenance proxy

### Skip trace

- **Sunbiz live lookup** — for entity-named owners (LLC / INC / TRUST etc.).
  Searches Sunbiz by name, fetches the matching corporation's detail page,
  pulls registered agent, status, email, phone. Free, no API key. ~3–8 s.
- **DuckDuckGo search** — free web search for owner + city. Returns website,
  socials, email, phone scraped from search results. ~3–6 s.
- **Quick links** — deep-linked queries on Google, Sunbiz, TruePeopleSearch,
  FastPeopleSearch, BeenVerified, county appraiser, and Google Maps.

Both live lookups cache for 24 hours so re-clicking is instant.

### Push to CRM

Paste your `CAPTURE_TOKEN` in the sidebar to enable two push paths:

- **Push top N to CRM** on Scored Leads — sends the highest-score filtered rows
  to `/api/capture`, capped at 200 per click. Idempotent (upserts by parcelId).
- **Push this lead to CRM** on Parcel Lookup — single row.

Set `crm.capture_token` in Streamlit secrets to skip the manual paste.

### Owner normalization

Owner names are normalized (uppercase, drop periods, &→AND, collapse spaces)
the same way the CRM does it. That means "ABC LLC", "ABC, LLC", and
"ABC L.L.C." count as one entity for `multi_property_owner`, and the CRM won't
create duplicate leads when you push the same owner twice.

### Analytics tab

Charts reflect whatever the sidebar is filtering to. Score and flag
distributions, top counties, just-value buckets, and a top-25-owner-entities
leaderboard for the current filter set.
"""
    )
