"""Florida Motivated-Sellers Dashboard."""

import json
import re
from pathlib import Path
from urllib.parse import quote_plus

import polars as pl
import streamlit as st

import crm_push
import outreach
import skip_trace
from fl_geo import FL_COUNTY_CENTROIDS

st.set_page_config(
    page_title="FL Motivated Sellers",
    page_icon="🏚️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _password_gate() -> None:
    """If `app.password` is set in Streamlit secrets, require it before render.

    Locally (no secrets file) the app is open. On Streamlit Cloud, set
    `app.password = "..."` in the app's Settings → Secrets to lock down.
    """
    expected = ""
    try:
        expected = st.secrets["app"]["password"]
    except Exception:
        pass
    if not expected:
        return
    if st.session_state.get("auth_ok"):
        return

    st.title("🔒 Locked")
    st.caption("Enter the dashboard password to continue.")
    pw = st.text_input("Password", type="password", label_visibility="collapsed")
    if pw and pw == expected:
        st.session_state["auth_ok"] = True
        st.rerun()
    elif pw:
        st.error("Wrong password.")
    st.stop()


_password_gate()

DATA_DIR = Path(__file__).parent / "data"
LEADS_PATH = DATA_DIR / "leads.parquet"
BY_OWNER = DATA_DIR / "by_owner.parquet"
BY_ADDR = DATA_DIR / "by_address.parquet"
SKIP_TRACE_CACHE = DATA_DIR / "skip_trace_cache.parquet"

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

# Mail-merge: owner address (where letter goes) + property address (subject of
# letter). Column order is friendly to Avery 5160 templates and Word mail merge.
MAIL_MERGE_COLUMNS = [
    "owner_name", "owner_mailing_street", "owner_mailing_city",
    "owner_state", "owner_mailing_zip",
    "situs_address", "situs_city", "situs_zip", "county_name",
    "score", "flags", "parcel_id",
]

# Polars panics serializing very large frames to CSV on this machine.
# Cap downloads at a safe size — fits Excel and any practical workflow.
CSV_EXPORT_CAP = 25_000


def safe_csv(df: pl.DataFrame) -> str:
    """write_csv with a row cap to avoid polars panics on huge frames."""
    if df.height > CSV_EXPORT_CAP:
        df = df.head(CSV_EXPORT_CAP)
    return df.write_csv()


def to_mail_merge_csv(df: pl.DataFrame) -> str:
    cleaned = (
        df.select([c for c in MAIL_MERGE_COLUMNS if c in df.columns])
        # Drop rows missing owner mailing — can't mail without the address.
        .filter(
            (pl.col("owner_mailing_street").is_not_null()) & (pl.col("owner_mailing_street") != "")
            & (pl.col("owner_mailing_city").is_not_null()) & (pl.col("owner_mailing_city") != "")
            & (pl.col("owner_mailing_zip").is_not_null()) & (pl.col("owner_mailing_zip") != "")
        )
    )
    return safe_csv(cleaned)


# Downcast map: Int64 columns that fit comfortably in narrower types. Values
# are unchanged — purely a memory representation change. Int16 holds 0..32767
# (years, score, unit counts); Int32 holds 0..2.1B (dollar amounts, sqft).
# just_value is deliberately left Int64 — it gets .sum()'d across 218k rows
# (KPI strip, Map, owner leaderboard) and Int32 could overflow that total.
_DOWNCAST = {
    "score": pl.Int16, "opportunity_score": pl.Int16,
    "last_sale_price": pl.Int32, "prior_sale_price": pl.Int32,
    "est_equity": pl.Int32, "owner_distance_mi": pl.Int16,
    "living_area": pl.Int32,
    "last_sale_year": pl.Int16, "prior_sale_year": pl.Int16,
    "year_built": pl.Int16, "residential_units": pl.Int16,
}
# Repeated-value string columns → Categorical (dictionary-encoded in memory).
# county_name has 67 distinct values across 218k rows, signal_type just 1.
_CATEGORICAL = ("county_name", "owner_state", "signal_type", "dor_uc")


@st.cache_data
def load_leads() -> pl.DataFrame:
    df = pl.read_parquet(LEADS_PATH)
    # evidence_url is always empty — drop it rather than carry 218k empty strings.
    if "evidence_url" in df.columns:
        df = df.drop("evidence_url")
    casts = [pl.col(c).cast(t, strict=False) for c, t in _DOWNCAST.items() if c in df.columns]
    casts += [pl.col(c).cast(pl.Categorical) for c in _CATEGORICAL if c in df.columns]
    return df.with_columns(casts) if casts else df


@st.cache_data
def load_by_owner() -> pl.DataFrame:
    return pl.read_parquet(BY_OWNER)


@st.cache_data
def load_by_address() -> pl.DataFrame:
    return pl.read_parquet(BY_ADDR)


# ttl=24h AND max_entries — bound both the age and the count of cached lookups
# so a long-running session can't grow these caches without limit.
@st.cache_data(ttl=86400, max_entries=512, show_spinner=False)
def cached_sunbiz_search(name: str) -> str:
    return skip_trace.sunbiz_search_by_name(name)


@st.cache_data(ttl=86400, max_entries=512, show_spinner=False)
def cached_sunbiz_detail(cor_number: str) -> dict:
    return skip_trace.sunbiz_detail(cor_number)


@st.cache_data(ttl=86400, max_entries=512, show_spinner=False)
def cached_ddg(query: str) -> dict:
    return skip_trace.duckduckgo_lookup(query)


def load_skip_cache() -> pl.DataFrame:
    return skip_trace.load_cache(SKIP_TRACE_CACHE)


def _anthropic_key() -> str:
    try:
        return st.secrets["anthropic"]["api_key"]
    except Exception:
        return ""


@st.cache_data(ttl=86400, max_entries=512, show_spinner=False)
def cached_opener(parcel_id: str, owner_name: str, flags: str, score: int,
                  city: str, year_built, just_value, last_sale_year, last_sale_price,
                  situs_address: str, situs_zip: str, county_name: str) -> dict:
    """Cached so re-clicking the same parcel within 24h is free."""
    row = {
        "parcel_id": parcel_id, "owner_name": owner_name, "flags": flags,
        "score": score, "situs_city": city, "year_built": year_built,
        "just_value": just_value, "last_sale_year": last_sale_year,
        "last_sale_price": last_sale_price, "situs_address": situs_address,
        "situs_zip": situs_zip, "county_name": county_name,
    }
    return outreach.draft_opener(row, _anthropic_key())


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

# ── Hydrate filter state from URL query params (once per session) ─────────────
qp = st.query_params


def _qp_csv(key: str) -> list[str]:
    raw = qp.get(key)
    return [x for x in raw.split(",") if x] if raw else []


def _qp_int(key: str, default: int) -> int:
    try:
        return int(qp.get(key, default))
    except (TypeError, ValueError):
        return default


def _qp_bool(key: str) -> bool:
    return str(qp.get(key, "")).lower() in {"1", "true", "yes"}


_VMIN_DEFAULT = int(leads["just_value"].min() or 0)
_VMAX_DEFAULT = int(leads["just_value"].max() or 0)

_STATE_DEFAULTS = {
    "f_counties": _qp_csv("counties"),
    "f_value_range": (_qp_int("vmin", _VMIN_DEFAULT), _qp_int("vmax", _VMAX_DEFAULT)),
    "f_min_score": _qp_int("min_score", 75),
    "f_multifamily": _qp_bool("multifamily"),
    "f_recent_buyer": _qp_bool("recent_buyer"),
    "f_pre_1980": _qp_bool("pre_1980"),
    "f_entity_only": _qp_bool("entity_only"),
}
for k, v in _STATE_DEFAULTS.items():
    st.session_state.setdefault(k, v)

# Star/shortlist state — session-only, exportable as CSV
st.session_state.setdefault("starred_parcels", set())

# Saved searches — session-only, exportable as JSON
st.session_state.setdefault("saved_searches", {})
# Deal-math / freshness / dedup widget state
st.session_state.setdefault("f_repair_est", 30_000)
st.session_state.setdefault("f_new_only", False)
st.session_state.setdefault("f_hide_crm", False)
st.session_state.setdefault("f_sort", "Opportunity")
st.session_state.setdefault("f_min_distance", 0)

_require_from_url = set(_qp_csv("require"))
_exclude_from_url = set(_qp_csv("exclude"))
for flag in FLAG_DESCRIPTIONS:
    st.session_state.setdefault(f"req_{flag}", flag in _require_from_url)
    st.session_state.setdefault(f"exc_{flag}", flag in _exclude_from_url)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")

    with st.expander("Counties & value", expanded=True):
        counties_all = sorted(leads["county_name"].unique().to_list())
        selected_counties = st.multiselect(
            "Counties", counties_all, placeholder="All counties", key="f_counties",
        )
        value_range = st.slider(
            "Just value ($)",
            min_value=_VMIN_DEFAULT, max_value=_VMAX_DEFAULT,
            step=10_000, format="$%d", key="f_value_range",
        )
        min_score = st.slider("Min score", 0, 100, step=25, key="f_min_score")

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
            "Multifamily only", key="f_multifamily",
            help="Residential units ≥ 2 (DOR codes 003 / 008 / 009 also included)",
        )
        recent_buyer_stranded = st.checkbox(
            "Recent buyer underwater", key="f_recent_buyer",
            help="Bought in last 4 yrs at ≥ current just value",
        )
        pre_1980 = st.checkbox(
            "Pre-1980 build + 25+ yr held", key="f_pre_1980",
            help="Deferred-maintenance proxy",
        )
        entity_only = st.checkbox(
            "LLC / entity owners only", key="f_entity_only",
            help="Owner is an LLC, trust, or other entity (not an individual).",
        )
        min_distance = st.number_input(
            "Owner ≥ N miles away", min_value=0, max_value=3000, step=25,
            key="f_min_distance",
            help="Distance from the owner's mailing address to the property. "
                 "Far-away absentee owners are markedly more motivated. 0 = no limit.",
        )

    with st.expander("💰 Deal math"):
        repair_est = st.number_input(
            "Repair estimate ($)", min_value=0, max_value=500_000, step=5_000,
            key="f_repair_est",
            help="Subtracted from the 70%-of-value offer ceiling. Tune to your typical rehab.",
        )
        st.caption(
            "Offer ceiling = 70% × just-value − repair estimate. A starting "
            "max-offer reference for absentee/equity deals — FL just-value "
            "approximates market, not ARV, so treat it as a ballpark."
        )

    with st.expander("🆕 Freshness & dedup"):
        new_only = st.checkbox(
            "New since last refresh", key="f_new_only",
            help="Show only parcels first seen in the most recent data refresh.",
        )
        hide_crm_dupes = st.checkbox(
            "Hide leads already in CRM", key="f_hide_crm",
            help="Cross-checks acquisitions-crm and hides parcels already captured. "
                 "Needs the CAPTURE_TOKEN set below.",
        )

    st.markdown("---")

    # ── Shareable filter URL ──────────────────────────────────────────────────
    def _current_qp() -> dict[str, str]:
        out: dict[str, str] = {}
        if st.session_state["f_counties"]:
            out["counties"] = ",".join(st.session_state["f_counties"])
        if st.session_state["f_min_score"] != 75:
            out["min_score"] = str(st.session_state["f_min_score"])
        vmin, vmax = st.session_state["f_value_range"]
        if vmin != _VMIN_DEFAULT:
            out["vmin"] = str(vmin)
        if vmax != _VMAX_DEFAULT:
            out["vmax"] = str(vmax)
        req = [f for f in FLAG_DESCRIPTIONS if st.session_state.get(f"req_{f}")]
        if req:
            out["require"] = ",".join(req)
        exc = [f for f in FLAG_DESCRIPTIONS if st.session_state.get(f"exc_{f}")]
        if exc:
            out["exclude"] = ",".join(exc)
        for sk, qk in [
            ("f_multifamily", "multifamily"), ("f_recent_buyer", "recent_buyer"),
            ("f_pre_1980", "pre_1980"), ("f_entity_only", "entity_only"),
        ]:
            if st.session_state.get(sk):
                out[qk] = "1"
        return out

    _now_qp = _current_qp()
    with st.expander("📎 Share this filter"):
        if not _now_qp:
            st.caption("No filters set. Pick something above and the URL will appear here.")
        else:
            from urllib.parse import urlencode
            qs = urlencode(_now_qp)
            st.caption("Append this to the dashboard URL — opens with the same filters preloaded:")
            st.code(f"?{qs}", language=None)

    # ── Saved searches ────────────────────────────────────────────────────────
    def _apply_saved_search(payload: dict) -> None:
        """on_click callback — write a saved filter dict back into widget state."""
        st.session_state["f_counties"] = (
            payload.get("counties", "").split(",") if payload.get("counties") else []
        )
        st.session_state["f_min_score"] = int(payload.get("min_score", 75))
        st.session_state["f_value_range"] = (
            int(payload.get("vmin", _VMIN_DEFAULT)),
            int(payload.get("vmax", _VMAX_DEFAULT)),
        )
        for sk, qk in [
            ("f_multifamily", "multifamily"), ("f_recent_buyer", "recent_buyer"),
            ("f_pre_1980", "pre_1980"), ("f_entity_only", "entity_only"),
        ]:
            st.session_state[sk] = qk in payload
        _req = set((payload.get("require") or "").split(","))
        _exc = set((payload.get("exclude") or "").split(","))
        for _flag in FLAG_DESCRIPTIONS:
            st.session_state[f"req_{_flag}"] = _flag in _req
            st.session_state[f"exc_{_flag}"] = _flag in _exc

    with st.expander("💾 Saved searches"):
        _saved = st.session_state["saved_searches"]
        _new_name = st.text_input(
            "Name this search", key="_save_name",
            placeholder="e.g. Out-of-state Polk pre-1980",
        )
        if st.button(
            "Save current filters", use_container_width=True,
            disabled=not _new_name.strip(),
        ):
            _saved[_new_name.strip()] = _current_qp()
            st.success(f"Saved “{_new_name.strip()}”.")
        if _saved:
            st.caption("Click a name to apply it:")
            for _name, _payload in list(_saved.items()):
                _ca, _cd = st.columns([4, 1])
                _ca.button(
                    _name, key=f"_apply_{_name}", use_container_width=True,
                    on_click=_apply_saved_search, args=(_payload,),
                )
                if _cd.button("🗑", key=f"_del_{_name}", help=f"Delete “{_name}”"):
                    del _saved[_name]
                    st.rerun()
            st.download_button(
                "⬇️ Export saved searches",
                json.dumps(_saved, indent=2),
                file_name="saved_searches.json", mime="application/json",
                use_container_width=True,
            )
        _ss_upload = st.file_uploader("Restore from JSON", type="json", key="_ss_upload")
        if _ss_upload is not None:
            try:
                _loaded = json.loads(_ss_upload.getvalue())
                if isinstance(_loaded, dict):
                    _saved.update(_loaded)
                    st.success(f"Loaded {len(_loaded)} saved search(es).")
                else:
                    st.error("That file isn't a saved-searches export.")
            except Exception as _e:
                st.error(f"Couldn't read that file: {_e}")

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


# ── Derived filter inputs (freshness + CRM dedup) ───────────────────────────
_latest_seen = leads["first_seen"].max() if "first_seen" in leads.columns else None


@st.cache_data(ttl=600, show_spinner="Checking CRM for duplicates…")
def cached_known_parcels(base_url: str, token: str) -> frozenset:
    """Cached set of parcelIds already in the CRM (refreshes every 10 min)."""
    res = crm_push.fetch_known_parcels(base_url, token)
    return frozenset(res["parcels"]) if res.get("ok") else frozenset()


_crm_known: frozenset = frozenset()
if hide_crm_dupes and crm_token.strip():
    _crm_known = cached_known_parcels(crm_base_url, crm_token)
    if not _crm_known:
        st.sidebar.caption("⚠️ Couldn't reach the CRM — dedup skipped.")


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
    if min_distance > 0 and "owner_distance_mi" in out.columns:
        out = out.filter(pl.col("owner_distance_mi") >= min_distance)
    if new_only and _latest_seen is not None and "first_seen" in out.columns:
        out = out.filter(pl.col("first_seen") == _latest_seen)
    if hide_crm_dupes and _crm_known:
        out = out.filter(~pl.col("parcel_id").is_in(list(_crm_known)))
    return out


filtered = apply_filters(leads)
# Live deal math — offer ceiling = 70% of just-value minus the repair estimate.
filtered = filtered.with_columns(
    (0.70 * pl.col("just_value") - repair_est)
    .round(0)
    .clip(lower_bound=0)
    .cast(pl.Int64)
    .alias("offer_estimate")
)

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


_new_count = (
    filtered.filter(pl.col("first_seen") == _latest_seen).height
    if _latest_seen is not None and "first_seen" in filtered.columns
    else 0
)

kpi_html = f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">Leads</div>
    <div class="kpi-value">{filtered.height:,}</div>
    <div class="kpi-sub">filtered of {leads.height:,} · 🆕 {_new_count:,} new</div>
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
_starred_count = len(st.session_state["starred_parcels"])
_star_tab_label = f"⭐ Starred ({_starred_count})" if _starred_count else "⭐ Starred"

t_leads, t_owners, t_addr, t_lookup, t_starred, t_analytics, t_map, t_about = st.tabs(
    ["Scored leads", "Multi-property owners", "Address clusters",
     "Parcel lookup", _star_tab_label, "Analytics", "🗺 Map", "About"]
)

LEAD_COL_CONFIG = {
    "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
    "opportunity_score": st.column_config.ProgressColumn(
        "Opportunity", min_value=0, max_value=100, format="%d",
        help="Blend of motivation score + equity signal.",
    ),
    "est_equity": st.column_config.NumberColumn(
        "Est. equity", format="$%d", help="Just-value − last sale price (when a sale exists).",
    ),
    "offer_estimate": st.column_config.NumberColumn(
        "Offer (est.)", format="$%d", help="70% of just-value − your repair estimate.",
    ),
    "first_seen": st.column_config.TextColumn("First seen", width="small"),
    "owner_distance_mi": st.column_config.NumberColumn(
        "Owner mi away", format="%d mi",
        help="Distance from the owner's mailing address to the property.",
    ),
    "lead_summary": st.column_config.TextColumn(
        "Why this lead", width="large", help="Plain-English read on the motivation signals.",
    ),
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
        _sort_opts = {
            "Opportunity": "opportunity_score",
            "Motivation score": "score",
            "Offer (est.)": "offer_estimate",
            "Just value": "just_value",
            "Est. equity": "est_equity",
            "Distance from owner": "owner_distance_mi",
        }
        _sort_label = st.selectbox("Sort by", list(_sort_opts), key="f_sort")
        display = filtered.select([
            "opportunity_score", "score", "lead_summary", "county_name", "situs_address",
            "situs_city", "situs_zip", "owner_name", "owner_mailing", "owner_state",
            "owner_distance_mi", "just_value", "est_equity", "offer_estimate", "year_built",
            "residential_units", "last_sale_year", "flags", "first_seen", "parcel_id",
        ]).sort(_sort_opts[_sort_label], descending=True, nulls_last=True)

        st.dataframe(
            display,
            use_container_width=True,
            height=520,
            hide_index=True,
            column_config=LEAD_COL_CONFIG,
        )

        d1, d2, d3, d4 = st.columns(4)
        d1.download_button(
            "⬇️ Filtered CSV",
            safe_csv(display),
            file_name="motivated_sellers_filtered.csv",
            mime="text/csv",
            use_container_width=True,
            help="Full columns of the filtered table.",
        )
        crm_ready_csv = (
            filtered.select([c for c in CRM_COLUMNS if c in filtered.columns])
            .sort("score", descending=True)
        )
        d2.download_button(
            "⬇️ CRM CSV",
            safe_csv(crm_ready_csv),
            file_name="acquisitions_crm_import.csv",
            mime="text/csv",
            use_container_width=True,
            help="Column shape matching acquisitions-crm's import script.",
        )
        d3.download_button(
            "⬇️ Mail-merge CSV",
            to_mail_merge_csv(filtered.sort("score", descending=True)),
            file_name="mail_merge.csv",
            mime="text/csv",
            use_container_width=True,
            help="Avery / Word mail-merge format: owner mailing + property address. "
                 "Drops rows missing owner mailing.",
        )
        push_n = min(filtered.height, 200)
        push_clicked = d4.button(
            f"🚀 Push top {push_n} to CRM",
            disabled=not crm_ready or filtered.is_empty(),
            help="POST the highest-score filtered rows to the CRM /api/capture endpoint."
                 " Capped at 200/click for safety.",
            use_container_width=True,
        )

        # Bulk skip-trace
        st.markdown("---")
        skip_cache_df = load_skip_cache()
        entity_filtered = filtered.filter(pl.col("is_entity"))
        unique_entities = entity_filtered.unique(subset=["owner_norm"]).height
        already_cached = (
            entity_filtered.filter(
                pl.col("owner_norm").is_in(skip_cache_df["owner_norm"].to_list())
            ).unique(subset=["owner_norm"]).height
            if not skip_cache_df.is_empty() else 0
        )
        to_run = unique_entities - already_cached
        bulk_n = min(to_run, 100)

        st1, st2 = st.columns([3, 2])
        st1.caption(
            f"Skip-trace cache: {skip_cache_df.height:,} entities. "
            f"In current filter: {unique_entities:,} unique entity owners "
            f"({already_cached:,} already cached, {to_run:,} new)."
        )
        bulk_clicked = st2.button(
            f"🕵️ Bulk skip-trace next {bulk_n}",
            disabled=bulk_n <= 0,
            help="Sunbiz live lookups for entity owners not yet in the cache. "
                 "~2 sec each (so up to ~4 min for 100 owners). Free, no API key.",
            use_container_width=True,
        )
        if bulk_clicked and bulk_n > 0:
            targets = (
                entity_filtered
                .filter(~pl.col("owner_norm").is_in(skip_cache_df["owner_norm"].to_list()))
                .unique(subset=["owner_norm"])
                .sort("score", descending=True)
                .head(bulk_n)
                .select(["owner_norm", "owner_name"])
                .to_dicts()
            )
            progress = st.progress(0, text=f"skip-tracing 0/{len(targets)} …")
            status_box = st.empty()

            def cb(done: int, total: int, last: dict) -> None:
                label = last.get("owner", "")[:60]
                progress.progress(done / total, text=f"skip-tracing {done}/{total} — {label}")

            stats = skip_trace.bulk_sunbiz_trace(
                targets, SKIP_TRACE_CACHE, delay_sec=2.0, on_progress=cb,
            )
            progress.empty()
            load_skip_cache.clear() if hasattr(load_skip_cache, "clear") else None
            status_box.success(
                f"✅ done — found {stats['found']}, no Sunbiz match {stats['no_match']}, "
                f"errors {stats['errors']}. Cache now has {stats['cache_size']:,} entities."
            )

        if not skip_cache_df.is_empty():
            cache_d1, cache_d2 = st.columns([1, 1])
            cache_d1.download_button(
                "⬇️ Download skip-trace cache",
                skip_cache_df.write_csv(),
                file_name="skip_trace_cache.csv",
                mime="text/csv",
                use_container_width=True,
                help="Keep a copy — Streamlit Cloud filesystem is ephemeral.",
            )
            uploaded = cache_d2.file_uploader(
                "Upload cache (.csv)",
                type=["csv"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                try:
                    incoming = pl.read_csv(uploaded)
                    merged = pl.concat([skip_cache_df, incoming], how="diagonal_relaxed")
                    merged = merged.unique(subset=["owner_norm"], keep="last")
                    skip_trace.save_cache(merged, SKIP_TRACE_CACHE)
                    st.success(f"Merged {incoming.height:,} rows. Cache now {merged.height:,}.")
                except Exception as e:
                    st.error(f"Failed to merge cache: {e}")
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
        safe_csv(owner_view),
        file_name="multi_property_owners.csv",
        mime="text/csv",
    )

    # ── Portfolio equity leaders (live, from the current filtered leads) ──────
    st.markdown("---")
    st.markdown("##### 💰 Portfolio equity leaders")
    st.caption(
        "Owners with the most total estimated equity across the *currently filtered* "
        "leads — prime candidates for a single bulk / portfolio offer."
    )
    portfolio = (
        filtered.group_by("owner_norm")
        .agg(
            pl.col("owner_name").first().alias("owner"),
            pl.len().alias("parcels"),
            pl.col("county_name").n_unique().alias("counties"),
            pl.col("just_value").sum().alias("total_value"),
            pl.col("est_equity").sum().alias("total_equity"),
            pl.col("opportunity_score").mean().round(0).cast(pl.Int64).alias("avg_opportunity"),
        )
        .filter(pl.col("parcels") >= 2)
        .sort("total_equity", descending=True, nulls_last=True)
        .drop("owner_norm")
    )
    if portfolio.is_empty():
        st.info("No owner holds 2+ parcels in the current filter — loosen the sidebar.")
    else:
        st.dataframe(
            portfolio.head(200),
            use_container_width=True, height=420, hide_index=True,
            column_config={
                "owner": st.column_config.TextColumn("Owner", width="medium"),
                "parcels": st.column_config.NumberColumn("Parcels", format="%d"),
                "counties": st.column_config.NumberColumn("Counties", format="%d"),
                "total_value": st.column_config.NumberColumn("Total value", format="$%d"),
                "total_equity": st.column_config.NumberColumn("Total equity", format="$%d"),
                "avg_opportunity": st.column_config.ProgressColumn(
                    "Avg opportunity", min_value=0, max_value=100, format="%d"),
            },
        )
        st.download_button(
            "⬇️ Download portfolio leaders",
            safe_csv(portfolio),
            file_name="portfolio_equity_leaders.csv", mime="text/csv",
        )

with t_addr:
    st.markdown("##### Mailing-address clusters")
    st.caption(
        f"{by_addr.height:,} mailing addresses tied to 5+ parcels each — property managers, "
        f"big landlords, and (when distinct owners > 1) likely shell-company clusters."
    )

    addr_c1, addr_c2 = st.columns([1, 1])
    min_a_parcels = addr_c1.slider(
        "Min parcels per address",
        int(by_addr["parcel_count"].min()),
        int(by_addr["parcel_count"].max()),
        5, key="addr_min",
    )
    shell_only = addr_c2.checkbox(
        "Shell-company candidates only (≥ 2 distinct owner names)",
        help="Surface addresses where multiple different entity names share the same mailing addr.",
        key="addr_shell_only",
    )

    addr_view = by_addr.filter(pl.col("parcel_count") >= min_a_parcels)
    if shell_only:
        addr_view = addr_view.filter(pl.col("distinct_owner_names") >= 2)
    addr_view = (
        addr_view.with_columns(
            pl.col("owner_names").list.unique().list.head(5).list.join(" · ").alias("owners_preview")
        )
        .select([
            "addr_norm", "parcel_count", "distinct_owner_names",
            "total_just_value", "county_count", "owners_preview",
        ])
        .sort("parcel_count", descending=True)
    )

    st.dataframe(
        addr_view,
        use_container_width=True,
        height=500,
        hide_index=True,
        column_config={
            "addr_norm": st.column_config.TextColumn("Address", width="medium"),
            "parcel_count": st.column_config.NumberColumn("Parcels", format="%d"),
            "distinct_owner_names": st.column_config.NumberColumn("Distinct owners", format="%d"),
            "total_just_value": st.column_config.NumberColumn("Total value", format="$%d"),
            "county_count": st.column_config.NumberColumn("Counties", format="%d"),
            "owners_preview": st.column_config.TextColumn("Owners (first 5)", width="large"),
        },
    )
    st.download_button(
        "⬇️ Download address clusters",
        safe_csv(addr_view),
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

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Opportunity", row.get("opportunity_score") or row["score"])
            _eq = row.get("est_equity")
            e2.metric("Est. equity", fmt_money(_eq) if _eq is not None else "—",
                      help="Just-value − last sale price (blank when no sale on file).")
            _offer = max(0, round(0.70 * (row["just_value"] or 0) - repair_est))
            e3.metric("Offer ceiling", fmt_money(_offer),
                      help="70% of just-value − your repair estimate.")
            _dist = row.get("owner_distance_mi")
            e4.metric("Owner distance", f"{int(_dist):,} mi" if _dist is not None else "—",
                      help="Owner mailing address → property.")

            if row.get("lead_summary"):
                st.info(f"**Why this lead** — {row['lead_summary']}")

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

            # Show cached result if we have one
            cache_hit = skip_trace.lookup_by_owner_norm(load_skip_cache(), row.get("owner_norm", ""))
            if cache_hit:
                phone = cache_hit.get("phone") or ""
                email = cache_hit.get("email") or ""
                phone_html = (
                    f'<a class="skip-result-val" href="tel:{re.sub(r"[^0-9+]", "", phone)}">{phone}</a>'
                    if phone else "—"
                )
                email_html = (
                    f'<a class="skip-result-val" href="mailto:{email}">{email}</a>'
                    if email else "—"
                )
                st.markdown(
                    f'<div class="skip-section">'
                    f'<div class="label">Cached · {cache_hit.get("cor_number") or "—"} · '
                    f'scraped {cache_hit.get("scraped_at", "")[:10]}</div>',
                    unsafe_allow_html=True,
                )
                cache_fields_html = {
                    "Status": cache_hit.get("status") or "—",
                    "Last annual report": cache_hit.get("last_report_year") or "—",
                    "Registered agent": cache_hit.get("ra_name") or "—",
                    "Email": email_html,
                    "Phone": phone_html,
                }
                for k, v in cache_fields_html.items():
                    st.markdown(
                        f'<div><span class="skip-result-key">{k}:</span> '
                        f'<span class="skip-result-val">{v}</span></div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            # Star + push to CRM
            star_col, push_col, _ = st.columns([1, 2, 4])
            parcel_id_str = str(row["parcel_id"])
            is_starred = parcel_id_str in st.session_state["starred_parcels"]
            star_label = "⭐ Starred" if is_starred else "☆ Star"
            star_clicked = star_col.button(
                star_label,
                use_container_width=True,
                key=f"star_{parcel_id_str}",
                help="Bookmark this parcel for the Starred tab.",
            )
            if star_clicked:
                if is_starred:
                    st.session_state["starred_parcels"].discard(parcel_id_str)
                else:
                    st.session_state["starred_parcels"].add(parcel_id_str)
                st.rerun()

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

            anthropic_ready = bool(_anthropic_key())

            colA, colB, colC = st.columns(3)
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
            run_opener = colC.button(
                "✍️ Draft opener (AI)",
                disabled=not anthropic_ready,
                help=(
                    "Use Claude Haiku to draft a 2–3 sentence custom opener tuned to this "
                    "lead's flag set. ~$0.001/call."
                    if anthropic_ready else
                    "Disabled — set anthropic.api_key in Streamlit secrets to enable."
                ),
                use_container_width=True,
                key=f"opener_{row['parcel_id']}",
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

            if run_opener and anthropic_ready:
                with st.spinner("Drafting opener…"):
                    res = cached_opener(
                        row["parcel_id"], row["owner_name"] or "", row["flags"] or "",
                        int(row["score"] or 0), row["situs_city"] or "",
                        row.get("year_built"), row.get("just_value"),
                        row.get("last_sale_year"), row.get("last_sale_price"),
                        row["situs_address"] or "", row["situs_zip"] or "",
                        row["county_name"] or "",
                    )
                if res.get("ok"):
                    st.markdown('<div class="skip-section">', unsafe_allow_html=True)
                    st.markdown('<div class="label">Suggested opener (Claude Haiku)</div>',
                                unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="color:#e6e6e6; font-size:15px; line-height:1.5; '
                        f'padding:8px 0;">{res["text"]}</div>',
                        unsafe_allow_html=True,
                    )
                    cache_note = ""
                    if res.get("cached_tokens"):
                        cache_note = f" · {res['cached_tokens']} cached prompt tokens"
                    st.caption(
                        f"in: {res.get('input_tokens', '?')} tok · "
                        f"out: {res.get('output_tokens', '?')} tok{cache_note}"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(f"Opener failed: {res.get('error', 'unknown')}")

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

            # ── Connected parcels (owner network) ─────────────────────────────
            st.markdown("---")
            st.markdown("##### 🕸️ Connected parcels")

            owner_norm_val = (row.get("owner_norm") or "").strip()
            mail_street_val = (row.get("owner_mailing_street") or "").strip()
            mail_city_val = (row.get("owner_mailing_city") or "").strip()

            same_owner = (
                leads.filter(
                    (pl.col("owner_norm") == owner_norm_val)
                    & (pl.col("parcel_id") != row["parcel_id"])
                )
                if owner_norm_val else pl.DataFrame()
            )
            same_addr = (
                leads.filter(
                    (pl.col("owner_mailing_street") == mail_street_val)
                    & (pl.col("owner_mailing_city") == mail_city_val)
                    & (pl.col("owner_norm") != owner_norm_val)
                )
                if mail_street_val and mail_city_val else pl.DataFrame()
            )

            net_col1, net_col2 = st.columns(2)
            with net_col1:
                st.markdown(f"**Same owner — {same_owner.height} other parcel(s)**")
                if same_owner.is_empty():
                    st.caption("None in the dashboard's lead set.")
                else:
                    st.dataframe(
                        same_owner.select([
                            "score", "county_name", "situs_address", "situs_city",
                            "just_value", "flags", "parcel_id",
                        ]).sort("score", descending=True).head(50),
                        use_container_width=True,
                        hide_index=True,
                        height=240,
                        column_config={
                            "score": st.column_config.ProgressColumn(
                                "Score", min_value=0, max_value=100, format="%d"
                            ),
                            "just_value": st.column_config.NumberColumn("Just $", format="$%d"),
                            "county_name": st.column_config.TextColumn("County", width="small"),
                            "situs_city": st.column_config.TextColumn("City", width="small"),
                        },
                    )
            with net_col2:
                st.markdown(
                    f"**Same mailing addr, different owner name — {same_addr.height} parcel(s)**"
                )
                st.caption("Possible shell-company / property-manager links.")
                if same_addr.is_empty():
                    st.caption("None.")
                else:
                    st.dataframe(
                        same_addr.select([
                            "owner_name", "score", "county_name", "situs_address",
                            "just_value", "parcel_id",
                        ]).sort("score", descending=True).head(50),
                        use_container_width=True,
                        hide_index=True,
                        height=240,
                        column_config={
                            "owner_name": st.column_config.TextColumn("Owner", width="medium"),
                            "score": st.column_config.ProgressColumn(
                                "Score", min_value=0, max_value=100, format="%d"
                            ),
                            "just_value": st.column_config.NumberColumn("Just $", format="$%d"),
                            "county_name": st.column_config.TextColumn("County", width="small"),
                        },
                    )

with t_starred:
    starred = st.session_state["starred_parcels"]
    st.markdown(f"##### Starred parcels ({len(starred):,})")
    st.caption(
        "Session-scoped bookmarks. Streamlit Cloud filesystem is ephemeral — "
        "use download / upload below to persist between sessions."
    )

    if not starred:
        st.info("No starred parcels yet. Open one in Parcel Lookup and tap ☆ Star.")
    else:
        starred_df = leads.filter(pl.col("parcel_id").is_in(list(starred))).select([
            "score", "county_name", "situs_address", "situs_city", "situs_zip",
            "owner_name", "owner_state", "just_value", "year_built", "flags", "parcel_id",
        ]).sort("score", descending=True)
        st.dataframe(
            starred_df,
            use_container_width=True,
            hide_index=True,
            column_config=LEAD_COL_CONFIG,
            height=440,
        )

        sa, sb, sc, sd = st.columns(4)
        sa.download_button(
            "⬇️ Parcel IDs",
            "\n".join(sorted(starred)),
            file_name="starred_parcels.txt",
            mime="text/plain",
            use_container_width=True,
        )
        starred_full = leads.filter(pl.col("parcel_id").is_in(list(starred)))
        sb.download_button(
            "⬇️ Mail-merge CSV",
            to_mail_merge_csv(starred_full.sort("score", descending=True)),
            file_name="starred_mail_merge.csv",
            mime="text/csv",
            use_container_width=True,
        )
        starred_push_n = min(starred_df.height, 200)
        push_starred = sc.button(
            f"🚀 Push all {starred_push_n} to CRM",
            disabled=not crm_ready,
            use_container_width=True,
            key="push_starred_batch",
        )
        clear_clicked = sd.button(
            "🗑 Clear list",
            use_container_width=True,
            key="clear_starred",
        )
        if clear_clicked:
            st.session_state["starred_parcels"] = set()
            st.rerun()

        if push_starred and crm_ready:
            rows = starred_df.head(starred_push_n).to_dicts()
            progress = st.progress(0, text=f"pushing 0/{starred_push_n} …")
            status_box = st.empty()

            def _cb(done: int, total: int, last: dict) -> None:
                progress.progress(done / total, text=f"pushing {done}/{total} …")

            stats = crm_push.push_batch(rows, crm_base_url, crm_token, progress_cb=_cb)
            progress.empty()
            if stats["failed"] == 0:
                status_box.success(
                    f"✅ pushed {stats['total']} — created {stats['created']}, updated {stats['other_ok']}"
                )
            else:
                status_box.warning(
                    f"pushed {stats['total']} — created {stats['created']}, "
                    f"updated {stats['other_ok']}, failed {stats['failed']}"
                )

        st.markdown("---")
        st.markdown("**Restore from file**")
        uploaded = st.file_uploader(
            "Upload parcel-ID list (one ID per line)",
            type=["txt", "csv"],
            label_visibility="collapsed",
            key="starred_upload",
        )
        if uploaded is not None:
            try:
                content = uploaded.read().decode("utf-8", errors="replace")
                ids = [line.strip() for line in content.splitlines() if line.strip()]
                # If CSV, first column is parcel_id
                ids = [i.split(",")[0].strip() for i in ids]
                added = 0
                for pid in ids:
                    if pid and pid not in st.session_state["starred_parcels"]:
                        st.session_state["starred_parcels"].add(pid)
                        added += 1
                if added:
                    st.success(f"Added {added} parcels. Tap any other tab and back to refresh.")
                else:
                    st.caption("No new parcels added.")
            except Exception as e:
                st.error(f"Failed to read upload: {e}")


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
        st.markdown("**Top 50 owner entities by parcel count (in filtered set)**")
        st.caption(
            "Includes cached skip-trace contacts (phone/email tappable). "
            "Run Bulk skip-trace on Scored Leads tab to fill the cache."
        )
        skip_cache_for_owners = load_skip_cache()
        top_owners = (
            filtered.filter(pl.col("is_entity"))
            .group_by("owner_norm")
            .agg(
                pl.len().alias("parcels"),
                pl.col("just_value").sum().alias("total_value"),
                pl.col("county_name").n_unique().alias("counties"),
                pl.col("score").mean().alias("avg_score"),
                pl.col("owner_name").first().alias("example_owner"),
            )
            .sort("parcels", descending=True).head(50)
        )
        if not skip_cache_for_owners.is_empty():
            top_owners = top_owners.join(
                skip_cache_for_owners.select([
                    "owner_norm",
                    pl.col("status").alias("sb_status"),
                    pl.col("ra_name").alias("sb_ra_name"),
                    pl.col("phone").alias("sb_phone"),
                    pl.col("email").alias("sb_email"),
                ]),
                on="owner_norm",
                how="left",
            )
            top_owners = top_owners.with_columns(
                pl.when(pl.col("sb_phone").is_not_null() & (pl.col("sb_phone") != ""))
                .then(pl.lit("tel:") + pl.col("sb_phone").str.replace_all(r"[^\d+]", ""))
                .otherwise(pl.lit(None))
                .alias("phone_link"),
                pl.when(pl.col("sb_email").is_not_null() & (pl.col("sb_email") != ""))
                .then(pl.lit("mailto:") + pl.col("sb_email"))
                .otherwise(pl.lit(None))
                .alias("email_link"),
            )
        else:
            for c in ["sb_status", "sb_ra_name", "sb_phone", "sb_email", "phone_link", "email_link"]:
                top_owners = top_owners.with_columns(pl.lit(None).alias(c))

        st.dataframe(
            top_owners.select([
                "example_owner", "parcels", "total_value", "counties", "avg_score",
                "sb_status", "sb_ra_name", "phone_link", "email_link",
            ]).rename({
                "example_owner": "owner_name",
                "phone_link": "phone",
                "email_link": "email",
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "owner_name": st.column_config.TextColumn("Owner", width="large"),
                "parcels": st.column_config.NumberColumn("Parcels", format="%d"),
                "total_value": st.column_config.NumberColumn("Total value", format="$%d"),
                "counties": st.column_config.NumberColumn("Counties", format="%d"),
                "avg_score": st.column_config.NumberColumn("Avg score", format="%d"),
                "sb_status": st.column_config.TextColumn("Sunbiz status", width="small"),
                "sb_ra_name": st.column_config.TextColumn("Reg. agent", width="medium"),
                "phone": st.column_config.LinkColumn("Phone", display_text=r".*tel:(.*)"),
                "email": st.column_config.LinkColumn("Email", display_text=r"mailto:(.*)"),
            },
        )


with t_map:
    import math
    import pydeck as pdk

    st.markdown("##### Florida lead density by county")
    st.caption(
        f"Bubble size = lead count, color intensity = total just-value. "
        f"Current filter: {filtered.height:,} leads across "
        f"{filtered['county_name'].n_unique()} counties."
    )

    if filtered.is_empty():
        st.info("No leads match the current filters.")
    else:
        county_agg = (
            filtered.group_by("county_name").agg(
                pl.len().alias("count"),
                pl.col("just_value").sum().alias("total_value"),
                pl.col("score").mean().alias("avg_score"),
            )
            .sort("count", descending=True)
            .to_dicts()
        )
        max_count = max(r["count"] for r in county_agg) or 1
        max_val = max(r["total_value"] or 0 for r in county_agg) or 1
        map_rows = []
        for r in county_agg:
            centroid = FL_COUNTY_CENTROIDS.get(r["county_name"])
            if not centroid:
                continue
            lat, lon = centroid
            radius = max(2500.0, math.sqrt(r["count"] / max_count) * 32_000.0)
            intensity = (r["total_value"] or 0) / max_val
            green = int(120 + intensity * 135)
            map_rows.append({
                "lat": lat, "lon": lon,
                "county": r["county_name"],
                "count": r["count"],
                "total_value": r["total_value"] or 0,
                "avg_score": round(r["avg_score"] or 0, 1),
                "radius": radius,
                "color": [40, green, 60, 200],
            })

        if not map_rows:
            st.warning("No centroids matched the filtered counties.")
        else:
            layer = pdk.Layer(
                "ScatterplotLayer",
                map_rows,
                get_position=["lon", "lat"],
                get_radius="radius",
                get_fill_color="color",
                pickable=True,
                stroked=True,
                get_line_color=[20, 20, 20, 220],
                line_width_min_pixels=1,
            )
            tooltip = {
                "html": (
                    "<b>{county} County</b><br>"
                    "Leads: {count}<br>"
                    "Total value: ${total_value}<br>"
                    "Avg score: {avg_score}"
                ),
                "style": {"backgroundColor": "#141414", "color": "#e6e6e6", "fontSize": "12px"},
            }
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(
                    latitude=28.5, longitude=-82.5, zoom=5.7, bearing=0, pitch=0,
                ),
                map_style="mapbox://styles/mapbox/dark-v10",
                tooltip=tooltip,
            )
            st.pydeck_chart(deck, use_container_width=True)

            with st.expander("Counties shown (full table)"):
                table = pl.DataFrame(map_rows).select(
                    "county", "count", "total_value", "avg_score"
                ).sort("count", descending=True)
                st.dataframe(
                    table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "county": st.column_config.TextColumn("County"),
                        "count": st.column_config.NumberColumn("Leads", format="%d"),
                        "total_value": st.column_config.NumberColumn("Total value", format="$%d"),
                        "avg_score": st.column_config.NumberColumn("Avg score", format="%.1f"),
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

### Deal math

- **Est. equity** — just-value minus the last arm's-length sale price. Blank when
  there's no sale on file (most long-held leads) — equity is then unknown but
  usually high.
- **Offer (est.)** — `70% × just-value − repair estimate`. A starting max-offer
  ceiling; set the repair estimate in the sidebar. FL just-value approximates
  market value (not ARV), so treat it as a ballpark.
- **Opportunity** — `0.6 × motivation score + 0.4 × equity signal`, where the
  equity signal is the known equity % (when a sale exists), 80 when the
  `high_equity_proxy` flag fired, else a neutral 50. Sort the leads table by it
  to surface the leads most likely to pencil as deals.

### Owner distance & "why this lead"

- **Owner mi away** — straight-line miles from the owner's mailing address to the
  property (zip-centroid haversine). A far-away owner is markedly more motivated;
  filter with "Owner ≥ N miles away" in the sidebar.
- **Why this lead** — a plain-English read on each parcel's signals, e.g.
  *"Out-of-state owner (NY), held 25+ years, PO-box mailing — likely high equity."*
- **Portfolio equity leaders** (Multi-property owners tab) — owners ranked by
  total estimated equity across the current filtered leads, to spot bulk-offer
  candidates.

### Freshness & dedup

- **First seen** — the data refresh a parcel first appeared in. "New since last
  refresh" filters to the most recent batch.
- **Hide leads already in CRM** — cross-checks `acquisitions-crm` (via
  `GET /api/parcels`) and hides parcels you've already captured, so you only
  work fresh leads. Needs the CAPTURE_TOKEN set.

### Saved searches

Name and save the current filter set, then re-apply it in one click. Export all
saved searches to a JSON file and re-upload it to restore them on another device.

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

### Shareable filter URLs

Every filter change updates the URL. Copy the address bar (or the
`📎 Share this filter` snippet in the sidebar) to bookmark or share a
filter combo — opening it preloads the same state.

### AI opener (Claude Haiku)

The Parcel Lookup tab has a `✍️ Draft opener (AI)` button that generates a
2–3 sentence custom intro tuned to the lead's flag set. Cost is ~$0.001
per call (system prompt is cached).

To enable, add to your Streamlit Cloud Settings → Secrets:
```toml
[anthropic]
api_key = "sk-ant-..."
```

### Password gate

Set this in Streamlit Cloud Settings → Secrets to lock the dashboard:
```toml
[app]
password = "your-secret"
```

Locally (no `secrets.toml`), the app stays open for development.
"""
    )

# ── Sync current filter state back to URL query params ───────────────────────
_qp_target = _current_qp()
_qp_existing = dict(st.query_params)
if _qp_target != _qp_existing:
    st.query_params.clear()
    for k, v in _qp_target.items():
        st.query_params[k] = v
