"""
scraper.py — Coleta de documentação pública dos apps Android
Apps: Spotify, WhatsApp, Telegram
Saída: data/raw/{app}.jsonl — um documento por linha
"""

import requests
import json
import time
import re
import logging
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

# ──────────────────────────────────────────────
# Configurações por app
# ──────────────────────────────────────────────

APPS_CONFIG = {
    "spotify": {
        "seed_urls": [
            "https://support.spotify.com/us/",
            "https://support.spotify.com/us/article/listen-offline/",
            "https://support.spotify.com/us/article/spotify-free-vs-spotify-premium/",
            "https://support.spotify.com/us/article/account-privacy/",
        ],
        "allowed_domains": ["support.spotify.com"],
        "url_patterns": [r"/us/article/", r"/us/\?"],
        "max_pages": 80,
        "content_selector": "article",
        "fallback_selector": "main",
    },
    "whatsapp": {
        "seed_urls": [
            "https://faq.whatsapp.com/",
            "https://faq.whatsapp.com/general/",
            "https://faq.whatsapp.com/android/",
        ],
        "allowed_domains": ["faq.whatsapp.com"],
        "url_patterns": [r"/general/", r"/android/", r"/web/", r"/security/"],
        "max_pages": 80,
        "content_selector": "article",
        "fallback_selector": "div.faq-answer",
    },
    "telegram": {
        "seed_urls": [
            "https://telegram.org/faq",
            "https://telegram.org/faq#general-questions",
            "https://telegram.org/tour",
        ],
        "allowed_domains": ["telegram.org"],
        "url_patterns": [r"/faq", r"/tour", r"/blog"],
        "max_pages": 60,
        "content_selector": "div.tgme_page_description",
        "fallback_selector": "main",
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 13; Pixel 7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Mobile Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Limpeza de texto
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove ruído típico de páginas de suporte."""
    # Colapsa espaços e quebras de linha excessivas
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove linhas muito curtas (menus, breadcrumbs)
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if len(l) > 30 or l == ""]
    # Remove padrões comuns de UI
    noise_patterns = [
        r"^(Home|Menu|Search|Skip to|Back to|Was this helpful\?|Yes|No|Share).*$",
        r"^\d+\s*(results?|articles?).*$",
        r"^(Cookie|Privacy|Terms).*$",
        r"^©.*$",
    ]
    cleaned = []
    for line in lines:
        if not any(re.match(p, line, re.IGNORECASE) for p in noise_patterns):
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def extract_text(soup: BeautifulSoup, content_selector: str, fallback_selector: str) -> str:
    """Extrai texto do elemento mais relevante da página."""
    # Remove elementos de navegação
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "button", "iframe"]):
        tag.decompose()

    # Tenta o seletor principal
    container = soup.select_one(content_selector)
    if not container:
        container = soup.select_one(fallback_selector)
    if not container:
        container = soup.find("body")

    if container:
        return clean_text(container.get_text(separator="\n", strip=True))
    return ""


def extract_title(soup: BeautifulSoup) -> str:
    """Extrai o título mais relevante da página."""
    for selector in ["h1", "title", "meta[property='og:title']"]:
        el = soup.select_one(selector)
        if el:
            return (el.get("content") or el.get_text()).strip()
    return ""


def extract_description(soup: BeautifulSoup) -> str:
    """Extrai meta description."""
    meta = soup.select_one("meta[name='description'], meta[property='og:description']")
    if meta:
        return meta.get("content", "").strip()
    return ""


# ──────────────────────────────────────────────
# Crawler principal
# ──────────────────────────────────────────────

class DocCrawler:
    def __init__(self, app_name: str, config: dict):
        self.app = app_name
        self.config = config
        self.visited: set[str] = set()
        self.queue: deque[str] = deque(config["seed_urls"])
        self.docs: list[dict] = []
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _is_valid_url(self, url: str) -> bool:
        """Verifica se a URL pertence ao domínio e padrões permitidos."""
        parsed = urlparse(url)
        domain_ok = parsed.netloc in self.config["allowed_domains"]
        if not domain_ok:
            return False
        # Ao menos um padrão deve casar (ou aceita qualquer path do domínio)
        patterns = self.config.get("url_patterns", [])
        if not patterns:
            return True
        return any(re.search(p, parsed.path) for p in patterns)

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extrai links internos válidos."""
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0].split("?")[0]  # Remove âncoras e query
            full_url = urljoin(base_url, href)
            if full_url not in self.visited and self._is_valid_url(full_url):
                links.append(full_url)
        return links

    def _fetch(self, url: str) -> BeautifulSoup | None:
        """Faz o request e retorna BeautifulSoup ou None em caso de erro."""
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            if "text/html" not in resp.headers.get("Content-Type", ""):
                return None
            return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            log.warning(f"Erro ao buscar {url}: {e}")
            return None

    def crawl(self) -> list[dict]:
        """Executa o crawl respeitando o limite de páginas."""
        max_pages = self.config["max_pages"]
        log.info(f"[{self.app.upper()}] Iniciando crawl (máx {max_pages} páginas)...")

        while self.queue and len(self.visited) < max_pages:
            url = self.queue.popleft()
            if url in self.visited:
                continue

            self.visited.add(url)
            log.info(f"  [{len(self.visited)}/{max_pages}] {url}")

            soup = self._fetch(url)
            if not soup:
                continue

            # Extrai conteúdo
            text = extract_text(
                soup,
                self.config["content_selector"],
                self.config["fallback_selector"],
            )

            if len(text) < 100:  # Ignora páginas sem conteúdo útil
                continue

            doc = {
                "id": f"{self.app}_{len(self.docs):04d}",
                "app": self.app,
                "url": url,
                "title": extract_title(soup),
                "description": extract_description(soup),
                "content": text,
                "collected_at": datetime.utcnow().isoformat(),
                "char_count": len(text),
                "word_count": len(text.split()),
            }
            self.docs.append(doc)

            # Descobre novos links
            new_links = self._extract_links(soup, url)
            self.queue.extend(new_links)

            time.sleep(1.2)  # Respeita o servidor — não sobrecarrega

        log.info(f"[{self.app.upper()}] Concluído: {len(self.docs)} páginas coletadas.")
        return self.docs


# ──────────────────────────────────────────────
# Persistência
# ──────────────────────────────────────────────

def save_jsonl(docs: list[dict], app: str) -> Path:
    """Salva os documentos em JSONL (um JSON por linha)."""
    out_path = OUTPUT_DIR / f"{app}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    log.info(f"Salvo: {out_path} ({len(docs)} docs)")
    return out_path


def save_stats(all_stats: dict) -> None:
    """Salva um resumo da coleta para referência."""
    stats_path = OUTPUT_DIR / "collection_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    log.info(f"Estatísticas salvas em {stats_path}")


# ──────────────────────────────────────────────
# Entrada
# ──────────────────────────────────────────────

def run(apps: list[str] | None = None):
    """
    Executa o scraping para os apps especificados.
    Se apps=None, coleta todos.
    """
    targets = apps or list(APPS_CONFIG.keys())
    all_stats = {}

    for app_name in targets:
        config = APPS_CONFIG[app_name]
        crawler = DocCrawler(app_name, config)
        docs = crawler.crawl()
        save_jsonl(docs, app_name)

        all_stats[app_name] = {
            "pages_collected": len(docs),
            "total_words": sum(d["word_count"] for d in docs),
            "total_chars": sum(d["char_count"] for d in docs),
            "avg_words_per_doc": round(
                sum(d["word_count"] for d in docs) / max(len(docs), 1), 1
            ),
        }

    save_stats(all_stats)

    log.info("\n=== RESUMO DA COLETA ===")
    for app, stats in all_stats.items():
        log.info(
            f"{app.upper()}: {stats['pages_collected']} páginas | "
            f"{stats['total_words']:,} palavras | "
            f"média {stats['avg_words_per_doc']} palavras/doc"
        )

    return all_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Coleta documentação pública de apps Android")
    parser.add_argument(
        "--apps",
        nargs="+",
        choices=list(APPS_CONFIG.keys()),
        default=None,
        help="Apps a coletar (padrão: todos)",
    )
    args = parser.parse_args()
    run(apps=args.apps)
