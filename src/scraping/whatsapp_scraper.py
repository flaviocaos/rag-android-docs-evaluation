"""
whatsapp_scraper.py — Coleta WhatsApp Help Center via URLs diretas conhecidas
O faq.whatsapp.com bloqueia crawlers, então usamos lista de URLs fixas + headers especiais.
"""

import requests
import json
import time
import re
import logging
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# URLs diretas do WhatsApp Help Center (coletadas manualmente)
WHATSAPP_URLS = [
    "https://faq.whatsapp.com/1994340644141728",  # About WhatsApp
    "https://faq.whatsapp.com/539814344816423",   # End-to-end encryption
    "https://faq.whatsapp.com/820124435853543",   # Linked devices
    "https://faq.whatsapp.com/1049597165590932",  # View once
    "https://faq.whatsapp.com/1137271884026785",  # Create a group
    "https://faq.whatsapp.com/196737011380816",   # Restore chat backup Android
    "https://faq.whatsapp.com/988059992053596",   # Backup to Google Drive
    "https://faq.whatsapp.com/1160613664285661",  # Mute notifications
    "https://faq.whatsapp.com/713787026506464",   # Two-step verification
    "https://faq.whatsapp.com/252072643436707",   # Status privacy
    "https://faq.whatsapp.com/1207939939839576",  # Block contacts
    "https://faq.whatsapp.com/360047315031",      # disappearing messages
    "https://faq.whatsapp.com/282967159713864",   # Voice and video calls
    "https://faq.whatsapp.com/267027452948898",   # WhatsApp Web
    "https://faq.whatsapp.com/1674753932948762",  # Storage and data
    "https://faq.whatsapp.com/1012693262670134",  # Notifications
    "https://faq.whatsapp.com/196775899352115",   # Profile photo privacy
    "https://faq.whatsapp.com/1256977581803711",  # Last seen and online
    "https://faq.whatsapp.com/1095939084935962",  # Read receipts
    "https://faq.whatsapp.com/885901392214861",   # Communities
    "https://faq.whatsapp.com/392191799085809",   # Stickers
    "https://faq.whatsapp.com/1080414132677086",  # Polls
    "https://faq.whatsapp.com/1243813596163923",  # Business accounts
    "https://faq.whatsapp.com/1111234503557036",  # Payments
    "https://faq.whatsapp.com/732656773907890",   # Security notifications
]

# Headers que simulam o app móvel do WhatsApp
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.101 Mobile Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [l.strip() for l in text.splitlines()]
    noise = [
        r"^(Home|Menu|Search|Back|Share|Was this helpful|Yes|No).*$",
        r"^©.*$",
        r"^.{0,25}$",
    ]
    cleaned = [l for l in lines
               if l and not any(re.match(p, l, re.I) for p in noise)]
    return "\n".join(cleaned).strip()


def scrape_url(url: str, session: requests.Session) -> dict | None:
    try:
        resp = session.get(url, timeout=20, headers=HEADERS)
        if resp.status_code != 200:
            log.warning(f"Status {resp.status_code}: {url}")
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        content = ""
        for selector in ["article", "main", ".content", ".faq-content", "body"]:
            el = soup.select_one(selector)
            if el:
                content = clean_text(el.get_text(separator="\n", strip=True))
                if len(content) > 150:
                    break

        if len(content) < 100:
            return None

        title = ""
        el = soup.select_one("h1") or soup.select_one("title")
        if el:
            title = el.get_text(strip=True)

        return {
            "id": f"whatsapp_{abs(hash(url)) % 100000:05d}",
            "app": "whatsapp",
            "url": url,
            "title": title,
            "content": content,
            "collected_at": datetime.utcnow().isoformat(),
            "word_count": len(content.split()),
            "char_count": len(content),
        }

    except Exception as e:
        log.warning(f"Erro em {url}: {e}")
        return None


def run():
    session = requests.Session()
    docs = []

    log.info(f"[WHATSAPP] Coletando {len(WHATSAPP_URLS)} URLs diretas...")

    for i, url in enumerate(WHATSAPP_URLS, 1):
        log.info(f"  [{i}/{len(WHATSAPP_URLS)}] {url}")
        doc = scrape_url(url, session)
        if doc:
            docs.append(doc)
            log.info(f"    OK: {doc['title'][:50]} ({doc['word_count']} palavras)")
        else:
            log.warning(f"    FALHOU: {url}")
        time.sleep(2.0)

    # Salva
    out_path = OUTPUT_DIR / "whatsapp.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    log.info(f"\n[WHATSAPP] {len(docs)} páginas coletadas.")
    log.info(f"Salvo em {out_path}")

    if len(docs) < 10:
        log.warning("Poucos documentos coletados. Veja alternativa abaixo.")
        log.warning("Alternativa: acesse https://faq.whatsapp.com manualmente,")
        log.warning("copie o texto de 20+ artigos e salve em data/raw/whatsapp_manual.txt")


if __name__ == "__main__":
    run()
