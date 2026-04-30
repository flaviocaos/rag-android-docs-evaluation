"""
scraper_advanced.py — Versão robusta do scraper com anti-bloqueio
Estratégias: sitemap parsing, retry com backoff, headers realistas

Use este script se o scraper.py básico encontrar erros 403/429.
"""

import requests
import json
import time
import re
import logging
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Pool de User-Agents realistas
# ──────────────────────────────────────────────

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

# ──────────────────────────────────────────────
# Sitemaps de cada app (fallback ao crawl normal)
# ──────────────────────────────────────────────

SITEMAPS = {
    "spotify": [
        "https://support.spotify.com/sitemap.xml",
        "https://support.spotify.com/us/sitemap.xml",
    ],
    "whatsapp": [
        "https://faq.whatsapp.com/sitemap.xml",
    ],
    "telegram": [
        # Telegram não tem sitemap — usa seed list expandida
    ],
}

# Seed URLs expandidas para Telegram (sem sitemap)
TELEGRAM_SEEDS = [
    "https://telegram.org/faq",
    "https://telegram.org/tour",
    "https://telegram.org/blog/topics",
    "https://core.telegram.org/bots",
    "https://core.telegram.org/api",
]


def get_headers() -> dict:
    """Retorna headers aleatórios para parecer um browser real."""
    ua = random.choice(USER_AGENTS)
    return {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }


def fetch_with_retry(url: str, session: requests.Session,
                     max_retries: int = 3, base_delay: float = 2.0) -> requests.Response | None:
    """Faz request com retry exponencial em caso de rate limiting."""
    for attempt in range(max_retries):
        try:
            session.headers.update(get_headers())
            resp = session.get(url, timeout=20, allow_redirects=True)

            if resp.status_code == 429:  # Rate limited
                wait = base_delay * (2 ** attempt) + random.uniform(0, 1)
                log.warning(f"Rate limited em {url}. Aguardando {wait:.1f}s...")
                time.sleep(wait)
                continue

            if resp.status_code == 403:
                log.warning(f"403 Forbidden: {url} — pulando")
                return None

            resp.raise_for_status()
            return resp

        except requests.exceptions.Timeout:
            log.warning(f"Timeout em {url} (tentativa {attempt+1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            log.warning(f"Erro em {url}: {e} (tentativa {attempt+1}/{max_retries})")

        time.sleep(base_delay * (attempt + 1))

    return None


def parse_sitemap(sitemap_url: str, session: requests.Session,
                  domain_filter: str = "") -> list[str]:
    """Extrai URLs de um sitemap XML."""
    resp = fetch_with_retry(sitemap_url, session)
    if not resp:
        return []

    urls = []
    try:
        root = ET.fromstring(resp.content)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # Sitemap index (aponta para outros sitemaps)
        for sitemap in root.findall("sm:sitemap/sm:loc", ns):
            sub_urls = parse_sitemap(sitemap.text.strip(), session, domain_filter)
            urls.extend(sub_urls)
            time.sleep(0.5)

        # URLs diretas
        for url_el in root.findall("sm:url/sm:loc", ns):
            url = url_el.text.strip()
            if not domain_filter or domain_filter in url:
                urls.append(url)

    except ET.ParseError as e:
        log.warning(f"Erro ao parsear sitemap {sitemap_url}: {e}")

    return urls


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [l.strip() for l in text.splitlines()]
    noise_patterns = [
        r"^(Home|Menu|Search|Skip to|Back to|Was this helpful\?|Yes|No|Share).*$",
        r"^\d+\s*(results?|articles?).*$",
        r"^(Cookie|Privacy|Terms).*$",
        r"^©.*$",
        r"^.{0,20}$",  # Linhas muito curtas
    ]
    cleaned = [l for l in lines
               if l and not any(re.match(p, l, re.I) for p in noise_patterns)]
    return "\n".join(cleaned).strip()


def scrape_page(url: str, session: requests.Session, app: str) -> dict | None:
    """Scraping de uma única página."""
    resp = fetch_with_retry(url, session)
    if not resp:
        return None

    if "text/html" not in resp.headers.get("Content-Type", ""):
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove elementos de UI
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "button", "iframe", "noscript"]):
        tag.decompose()

    # Tenta extrair o conteúdo principal
    content = ""
    for selector in ["article", "main", "[role='main']", ".content",
                     ".article-body", ".help-article", "body"]:
        el = soup.select_one(selector)
        if el:
            content = clean_text(el.get_text(separator="\n", strip=True))
            if len(content) > 200:
                break

    if len(content) < 100:
        return None

    title = ""
    for sel in ["h1", "title"]:
        el = soup.select_one(sel)
        if el:
            title = el.get_text(strip=True)
            break

    return {
        "id": f"{app}_{hash(url) % 100000:05d}",
        "app": app,
        "url": url,
        "title": title,
        "content": content,
        "collected_at": datetime.utcnow().isoformat(),
        "word_count": len(content.split()),
        "char_count": len(content),
    }


def scrape_app(app: str, max_pages: int = 80) -> list[dict]:
    """Scraping completo de um app com sitemap ou crawl."""
    session = requests.Session()
    docs = []
    urls_to_fetch: list[str] = []

    # Tenta via sitemap primeiro
    for sitemap_url in SITEMAPS.get(app, []):
        log.info(f"[{app.upper()}] Tentando sitemap: {sitemap_url}")
        urls = parse_sitemap(sitemap_url, session)
        if urls:
            log.info(f"[{app.upper()}] {len(urls)} URLs encontradas no sitemap")
            urls_to_fetch = urls[:max_pages]
            break
        time.sleep(1)

    # Fallback: seed URLs
    if not urls_to_fetch:
        if app == "telegram":
            urls_to_fetch = TELEGRAM_SEEDS
        log.info(f"[{app.upper()}] Usando seed URLs ({len(urls_to_fetch)} URLs)")

    visited = set()
    queue = deque(urls_to_fetch)

    while queue and len(docs) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        log.info(f"  [{len(docs)+1}/{max_pages}] {url}")
        doc = scrape_page(url, session, app)

        if doc:
            docs.append(doc)

        # Delay humano
        time.sleep(random.uniform(1.0, 2.5))

    log.info(f"[{app.upper()}] {len(docs)} páginas coletadas.")
    return docs


def run(apps: list[str] | None = None):
    targets = apps or ["spotify", "whatsapp", "telegram"]
    all_stats = {}

    for app in targets:
        docs = scrape_app(app)

        out_path = OUTPUT_DIR / f"{app}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        all_stats[app] = {
            "pages_collected": len(docs),
            "total_words": sum(d["word_count"] for d in docs),
        }
        log.info(f"Salvo: {out_path}")

    stats_path = OUTPUT_DIR / "collection_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    return all_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--apps", nargs="+",
                        choices=["spotify", "whatsapp", "telegram"], default=None)
    args = parser.parse_args()
    run(apps=args.apps)
