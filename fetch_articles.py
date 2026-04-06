"""
fetch_articles.py  —  CrossBorder Copilot Phase 1
Fetches DHL article pages that are publicly accessible (not blocked).
Saves them as HTML files in html_pages/.

Usage:
    python fetch_articles.py
"""

import time
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HTML_DIR = Path("html_pages")
HTML_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

DELAY = 2.0
TIMEOUT = 20

# Pages confirmed accessible (dhl.com/discover and dhl.com/us-en/home/ship)
ARTICLES = [
    ("customs-clearance-faq",
     "https://www.dhl.com/us-en/home/ship/customs-clearance-and-customs-declaration-faq.html"),
    ("customs-clearance-documents",
     "https://www.dhl.com/us-en/home/express/shipping-and-tracking/customs/customs-clearance/customs-clearance-documents.html"),
    ("customs-clearance-tips-us",
     "https://www.dhl.com/discover/en-us/ship-with-dhl/import-with-dhl/customs-clearance-and-restrictions"),
    ("customs-clearance-tips-global",
     "https://www.dhl.com/discover/en-global/ship-with-dhl/import-with-dhl/customs-clearance-and-restrictions"),
    ("customs-clearance-tips-sg",
     "https://www.dhl.com/discover/en-sg/logistics-advice/import-export-advice/customs-clearance-tips-avoid-delays"),
    ("customs-first-time-shippers",
     "https://www.dhl.com/discover/en-us/global-logistics-advice/essential-guides/customs-advice-first-time-shippers"),
    ("international-shipping-steps",
     "https://www.dhl.com/discover/en-global/e-commerce-advice/e-commerce-trends/the-six-steps-of-the-International-shipping-process"),
    ("ecommerce-tracking-faq",
     "https://www.dhl.com/us-en/home/customer-service/ecommerce-tracking-faq.html"),
    ("customs-clearance-must-knows",
     "https://www.dhl.com/gb-en/home/global-forwarding/freight-forwarding-education-center/customs-clearance-the-ocean-freight-must-knows.html"),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

session = requests.Session()
session.headers.update(HEADERS)

saved = 0
for name, url in ARTICLES:
    out_path = HTML_DIR / f"{name}.html"
    if out_path.exists():
        log.info("  SKIP (already exists)  %s", name)
        continue
    try:
        resp = session.get(url, timeout=TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup.find_all(["nav", "footer", "script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        if len(text) < 200:
            log.info("  ✗  %-40s  too little content", name)
            time.sleep(DELAY)
            continue

        out_path.write_text(resp.text, encoding="utf-8", errors="ignore")
        log.info("  ✓  %-40s  %d chars", name, len(text))
        saved += 1
    except Exception as exc:
        log.info("  ✗  %-40s  %s", name, exc)
    time.sleep(DELAY)

print(f"\nDone. Saved {saved} new HTML files → {HTML_DIR}/")
print("Next step: python parse_local.py && python ingest.py --reset")
