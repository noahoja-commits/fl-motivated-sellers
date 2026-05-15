"""AI-written outreach opener for a single parcel lead.

Uses Anthropic Claude Haiku 4.5 via the standard SDK. The system prompt is
marked for ephemeral prompt caching so repeated calls within a session pay
the cached rate (~10x cheaper input tokens).

Cost reference: claude-haiku-4-5 priced ~$1/MTok input, $5/MTok output.
This prompt + a ~120-token output ≈ $0.001 per call. With prompt caching on
the system block, subsequent calls land closer to $0.0003.
"""

from __future__ import annotations

import anthropic

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You write 2 to 3 short opening sentences for cold-call or
direct-mail outreach by a Florida real-estate investor to a property owner.

Rules:
- Tone: warm, specific, professional. No sales pitch, no buzzwords,
  no "just reaching out," no "I came across your property online."
- Mention 1 or 2 specific facts about the situation (long-held? out-of-state?
  trust/estate? owns multiple? bank-owned?) so it doesn't sound generic or mass-produced.
- End with a low-pressure question or invitation to talk.
- Max 60 words total. No bullet points, no headings, no markdown.
- Plain text output only. No quoting, no commentary, no "Here is your opener."
"""

FLAG_HINTS = {
    "out_of_state": "owner mails to another state — likely absentee",
    "out_of_zip": "owner mails to a different zip than the property",
    "po_box": "owner uses a PO box (investor / absentee)",
    "long_held_25y": "owned 25+ years — high equity",
    "trust_estate_name": "owner is a trust or estate (likely inheritance / probate)",
    "multi_property_owner": "owns multiple parcels in the area",
    "bank_trustee": "bank or REO trustee — distressed asset",
    "sale_anomaly": "sale price was unusually low or unusually high vs today's value",
    "high_equity_proxy": "bought decades ago for a fraction of today's value",
}


def _build_user_message(row: dict) -> str:
    flags = [f for f in (row.get("flags") or "").split(",") if f]
    flag_lines = "\n".join(f"- {f}: {FLAG_HINTS.get(f, '')}" for f in flags) or "- (no flags)"
    last_sale = (
        f"{row.get('last_sale_year')} for ${int(row.get('last_sale_price') or 0):,}"
        if row.get("last_sale_year") and row.get("last_sale_price")
        else "no recorded sale"
    )
    return (
        f"Lead summary:\n"
        f"- Owner: {row.get('owner_name')}\n"
        f"- Property: {row.get('situs_address')}, {row.get('situs_city')}, "
        f"FL {row.get('situs_zip')} ({row.get('county_name')} County)\n"
        f"- Current just value: ${int(row.get('just_value') or 0):,}\n"
        f"- Year built: {row.get('year_built') or 'unknown'}\n"
        f"- Last sale: {last_sale}\n"
        f"- Score: {row.get('score')}\n"
        f"- Flags fired:\n{flag_lines}\n\n"
        f"Write the opener."
    )


def draft_opener(row: dict, api_key: str) -> dict:
    """Returns {'ok': bool, 'text': str, 'error': str, 'cached_tokens': int}."""
    if not api_key:
        return {"ok": False, "text": "", "error": "no anthropic api key configured", "cached_tokens": 0}
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=MODEL,
            max_tokens=200,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": _build_user_message(row)}],
        )
        text = ""
        if resp.content and getattr(resp.content[0], "type", None) == "text":
            text = resp.content[0].text.strip()
        usage = resp.usage
        return {
            "ok": True,
            "text": text,
            "error": "",
            "cached_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
    except Exception as e:
        return {
            "ok": False,
            "text": "",
            "error": f"{type(e).__name__}: {e}",
            "cached_tokens": 0,
        }
