"""Push leads from the dashboard directly to acquisitions-crm.

Targets the /api/capture endpoint at the CRM, which expects:
  POST <base>/api/capture
  Authorization: Bearer <CAPTURE_TOKEN>
  JSON body matching crm/app/api/capture/route.ts::CaptureSchema

Idempotent on the CRM side: parcelId is the upsert key, so re-pushing a row
refreshes motivationScore/signals without overwriting UI edits.
"""

from __future__ import annotations

import time
from typing import Any

import requests

DEFAULT_BASE_URL = "https://acquisitions-crm-three.vercel.app"

_FLAG_PHRASES = {
    "out_of_state": "absentee owner (out of state)",
    "out_of_zip": "absentee owner (out of area)",
    "po_box": "PO box mailing",
    "long_held_25y": "long-held 25+ yrs",
    "trust_estate_name": "trust/estate-named owner",
    "multi_property_owner": "owns multiple in zip",
}


def row_to_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Translate one leads.parquet row into the CRM /api/capture body."""
    flags = [f for f in str(row.get("flags") or "").split(",") if f]
    motivation = " | ".join(_FLAG_PHRASES.get(f, f) for f in flags)

    return {
        # Lead (owner)
        "firstName": (row.get("owner_norm") or row.get("owner_name") or "(unknown)").strip() or "(unknown)",
        "motivation": motivation or None,
        "source": "OTHER",
        "sourceDetail": f"motivated-sellers absentee_equity (score {row.get('score')})",
        # Property
        "parcelId": str(row.get("parcel_id") or "") or None,
        "motivationScore": int(row.get("score") or 0),
        "streetAddress": (row.get("situs_address") or "").strip(),
        "city": (row.get("situs_city") or "").strip(),
        "state": "FL",
        "zip": (row.get("situs_zip") or "").strip(),
        "county": (row.get("county_name") or "").strip() or None,
        "yearBuilt": int(row["year_built"]) if row.get("year_built") else None,
        "sqft": int(row["living_area"]) if row.get("living_area") else None,
        "arv": int(row["just_value"]) if row.get("just_value") else None,
        # Owner mailing
        "ownerName": row.get("owner_name") or None,
        "ownerMailingStreet": row.get("owner_mailing_street") or None,
        "ownerMailingCity": row.get("owner_mailing_city") or None,
        "ownerMailingState": (row.get("owner_state") or "")[:2] or None,
        "ownerMailingZip": row.get("owner_mailing_zip") or None,
        "signals": flags,
    }


def push_one(row: dict[str, Any], base_url: str, token: str, timeout: int = 15) -> dict[str, Any]:
    """Push a single row to /api/capture. Returns {'ok': bool, 'status': int, ...}."""
    if not token:
        return {"ok": False, "status": 0, "error": "no token"}

    payload = row_to_payload(row)
    if not payload["streetAddress"] or not payload["city"] or not payload["zip"]:
        return {"ok": False, "status": 0, "error": "missing required address fields"}

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/capture",
            json=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=timeout,
        )
        out: dict[str, Any] = {"ok": resp.ok, "status": resp.status_code}
        try:
            body = resp.json()
            out.update({k: body.get(k) for k in ("leadId", "propertyId", "error", "details")})
        except Exception:
            out["raw"] = resp.text[:500]
        return out
    except requests.exceptions.Timeout:
        return {"ok": False, "status": 0, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "status": 0, "error": f"{type(e).__name__}: {e}"}


def push_batch(
    rows: list[dict[str, Any]],
    base_url: str,
    token: str,
    delay_sec: float = 0.15,
    progress_cb=None,
) -> dict[str, Any]:
    """Push many rows with a small delay between requests. Returns aggregate stats."""
    created = 0
    updated_or_other_ok = 0
    failed = 0
    errors: list[str] = []

    for i, row in enumerate(rows):
        res = push_one(row, base_url, token)
        if res.get("ok"):
            if res.get("status") == 201:
                created += 1
            else:
                updated_or_other_ok += 1
        else:
            failed += 1
            if len(errors) < 10:
                err = res.get("error") or f"HTTP {res.get('status')}"
                errors.append(f"{row.get('parcel_id', '?')}: {err}")
        if progress_cb:
            progress_cb(i + 1, len(rows), res)
        if delay_sec > 0 and i < len(rows) - 1:
            time.sleep(delay_sec)

    return {
        "total": len(rows),
        "created": created,
        "other_ok": updated_or_other_ok,
        "failed": failed,
        "errors": errors,
    }
