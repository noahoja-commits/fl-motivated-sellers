"""Free skip-trace helpers — Sunbiz detail scrape + DuckDuckGo lookup.

Ported and trimmed from sunbizdashboard-main/app.py.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

SUNBIZ_DETAIL_URL = "https://search.sunbiz.org/Inquiry/CorporationSearch/SearchResultDetail"
SUNBIZ_NAME_SEARCH_URL = "https://search.sunbiz.org/Inquiry/CorporationSearch/SearchResults"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://search.sunbiz.org/",
}

_INACTIVE_STATUSES = {
    "inactive", "dissolved", "revoked", "cancelled", "withdrawn",
    "administratively dissolved", "voluntarily dissolved",
    "merged", "converted", "expired",
}

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}")
_SUNBIZ_SKIP_DOMAINS = {
    "sunbiz.org", "dos.myflorida.com", "floridados.gov",
    "myfloridacfo.com", "dor.myflorida.com",
}
_WEB_SKIP_DOMAINS = {
    "example.com", "yourdomain.com", "gmail.com", "wixpress.com",
    "squarespace.com", "wordpress.com", "sentry.io",
}


def sunbiz_detail(cor_number: str) -> dict:
    """Scrape the Sunbiz detail page for a known corporation number."""
    out = {
        "email": "", "phone": "", "registered_agent": "",
        "last_report_year": "", "live_status": "",
        "is_inactive": False, "page_found": True,
        "cor_number": cor_number,
    }
    if not cor_number:
        out["page_found"] = False
        return out

    try:
        resp = requests.get(
            SUNBIZ_DETAIL_URL,
            params={
                "inquirytype": "DocumentNumber",
                "inquiryDirective": "StartsWith",
                "inquiryValue": cor_number.strip(),
                "redirected": "true",
            },
            headers=_HEADERS,
            timeout=12,
        )
        if resp.status_code != 200:
            out["page_found"] = False
            return out

        soup = BeautifulSoup(resp.text, "html.parser")
        page_text = soup.get_text(" ")

        if "no records found" in page_text.lower() or len(page_text.strip()) < 200:
            out["page_found"] = False
            out["is_inactive"] = True
            return out

        m = re.search(r"Status[:\s]+([A-Za-z ]+?)(?:\n|<|\|)", page_text, re.IGNORECASE)
        if m:
            raw = m.group(1).strip().lower()
            out["live_status"] = raw.title()
            out["is_inactive"] = any(s in raw for s in _INACTIVE_STATUSES)

        yrs = re.findall(r"Annual Report.*?20(\d{2})", page_text, re.IGNORECASE)
        if yrs:
            out["last_report_year"] = "20" + yrs[-1]

        for em in _EMAIL_RE.findall(page_text):
            if em.split("@")[-1].lower() not in _SUNBIZ_SKIP_DOMAINS:
                out["email"] = em.lower()
                break

        phones = _PHONE_RE.findall(page_text)
        if phones:
            out["phone"] = phones[0].strip()

        for label in soup.find_all(string=re.compile("Registered Agent", re.I)):
            parent = label.parent
            if parent:
                nxt = parent.find_next_sibling()
                if nxt:
                    out["registered_agent"] = nxt.get_text(strip=True)[:80]
            break

    except requests.exceptions.Timeout:
        out["page_found"] = False
    except Exception:
        pass

    return out


def sunbiz_search_by_name(name: str) -> str:
    """Look up a corp number by entity name. Returns the first match's cor_number or ''."""
    if not name:
        return ""
    try:
        resp = requests.get(
            SUNBIZ_NAME_SEARCH_URL,
            params={
                "inquiryType": "EntityName",
                "inquiryDirective": "StartsWith",
                "searchNameOrder": name.upper().strip(),
                "searchTerm": name.upper().strip(),
            },
            headers=_HEADERS,
            timeout=12,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        for link in soup.select("a"):
            href = link.get("href", "")
            m = re.search(r"inquiryValue=([A-Z0-9]+)", href, re.IGNORECASE)
            if m:
                return m.group(1)
    except Exception:
        pass
    return ""


def duckduckgo_lookup(query: str) -> dict:
    """Free DuckDuckGo HTML scrape — returns website, socials, email, phone."""
    out = {
        "website": "", "linkedin": "", "instagram": "",
        "facebook": "", "email": "", "phone": "",
        "snippets": [], "query": query,
    }
    if not query.strip():
        return out

    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "b": "", "kl": "us-en"},
            headers={**_HEADERS, "Content-Type": "application/x-www-form-urlencoded"},
            timeout=12,
        )
        soup = BeautifulSoup(resp.text, "html.parser")

        for snip_tag in soup.select(".result__snippet, .result__body")[:8]:
            text = snip_tag.get_text(" ", strip=True)
            if text:
                out["snippets"].append(text[:220])

        for tag in soup.select("a.result__url, a.result__a"):
            href = tag.get("href", "")
            if "uddg=" in href:
                qs = parse_qs(urlparse(href).query)
                href = unquote(qs.get("uddg", [""])[0])
            url = href.lower()
            if not url.startswith("http"):
                continue

            if ("linkedin.com/in/" in url or "linkedin.com/company/" in url) and not out["linkedin"]:
                out["linkedin"] = href
            elif "instagram.com/" in url and len(url.split("instagram.com/")[-1]) > 1 and not out["instagram"]:
                out["instagram"] = href
            elif "facebook.com/" in url and len(url.split("facebook.com/")[-1]) > 1 and not out["facebook"]:
                out["facebook"] = href
            elif not out["website"] and not any(
                s in url for s in (
                    "sunbiz", "floridados", "linkedin", "instagram", "facebook",
                    "twitter", "x.com", "yelp", "bbb.org", "yellowpages", "mapquest",
                    "whitepages", "spokeo", "radaris", "bizapedia", "opencorporates",
                    "duckduckgo", "google.com",
                )
            ):
                out["website"] = href

        if out["website"]:
            try:
                wr = requests.get(out["website"], headers=_HEADERS, timeout=8, allow_redirects=True)
                page_text = BeautifulSoup(wr.text, "html.parser").get_text(" ")
                for em in _EMAIL_RE.findall(page_text):
                    if em.split("@")[-1].lower() not in _WEB_SKIP_DOMAINS:
                        out["email"] = em.lower()
                        break
                phones = _PHONE_RE.findall(page_text)
                if phones:
                    out["phone"] = phones[0].strip()
            except Exception:
                pass

    except Exception:
        pass

    return out
