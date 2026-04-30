"""
preprocessor.py — Limpeza, deduplicação e chunking do corpus coletado
Entrada:  data/raw/{app}.jsonl
Saída:    data/processed/corpus.jsonl  (chunks prontos para indexação)
          data/processed/corpus_stats.json
"""

import json
import re
import hashlib
import logging
from pathlib import Path
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Parâmetros de chunking
# ──────────────────────────────────────────────

CHUNK_SIZE_TOKENS = 512   # tamanho alvo em tokens (~palavras * 1.3)
CHUNK_OVERLAP = 50        # overlap entre chunks consecutivos (em tokens)
MIN_CHUNK_WORDS = 30      # chunks menores que isso são descartados


# ──────────────────────────────────────────────
# Utilitários
# ──────────────────────────────────────────────

def token_count(text: str) -> int:
    """Estimativa rápida de tokens (sem tokenizador externo)."""
    return int(len(text.split()) * 1.3)


def content_hash(text: str) -> str:
    """Hash MD5 do conteúdo para deduplicação."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    """Normalização final antes do chunking."""
    text = re.sub(r"\s+", " ", text)          # Colapsa whitespace
    text = re.sub(r"\.{3,}", "...", text)     # Reticências
    text = re.sub(r"-{3,}", "—", text)        # Hífens múltiplos
    return text.strip()


# ──────────────────────────────────────────────
# Chunking fixo (estratégia RAG-A)
# ──────────────────────────────────────────────

def chunk_fixed(text: str, chunk_size: int = CHUNK_SIZE_TOKENS,
                overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide o texto em chunks de tamanho fixo (em palavras),
    com sobreposição para preservar contexto entre chunks.
    """
    words = text.split()
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i: i + chunk_size]
        chunk = " ".join(chunk_words)
        if len(chunk_words) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
    return chunks


# ──────────────────────────────────────────────
# Chunking semântico (estratégia RAG-B e RAG-C)
# ──────────────────────────────────────────────

def chunk_semantic(text: str, max_tokens: int = CHUNK_SIZE_TOKENS) -> list[str]:
    """
    Divide respeitando fronteiras naturais (parágrafos, frases).
    Agrega parágrafos até atingir max_tokens, então fecha o chunk.
    Isso preserva a coerência semântica de cada bloco.
    """
    # Divide em parágrafos
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = token_count(para)

        # Parágrafo sozinho já excede o limite — divide por frases
        if para_tokens > max_tokens:
            # Fecha o chunk atual antes
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts, current_tokens = [], 0
            # Divide o parágrafo por frases
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_buf: list[str] = []
            sent_tokens = 0
            for sent in sentences:
                st = token_count(sent)
                if sent_tokens + st > max_tokens and sent_buf:
                    chunks.append(" ".join(sent_buf))
                    sent_buf, sent_tokens = [], 0
                sent_buf.append(sent)
                sent_tokens += st
            if sent_buf:
                chunks.append(" ".join(sent_buf))
            continue

        # Adiciona o parágrafo ao chunk atual se couber
        if current_tokens + para_tokens <= max_tokens:
            current_parts.append(para)
            current_tokens += para_tokens
        else:
            # Fecha o atual e começa um novo
            if current_parts:
                chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_tokens = para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    # Filtra chunks muito pequenos
    return [c for c in chunks if len(c.split()) >= MIN_CHUNK_WORDS]


# ──────────────────────────────────────────────
# Pipeline de processamento
# ──────────────────────────────────────────────

def load_raw_docs(app: str) -> Iterator[dict]:
    """Carrega documentos brutos do JSONL."""
    path = RAW_DIR / f"{app}.jsonl"
    if not path.exists():
        log.warning(f"Arquivo não encontrado: {path}")
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def process_corpus(apps: list[str] | None = None) -> None:
    """
    Processa todos os apps e gera o corpus final com ambos os tipos de chunking.
    Cada chunk tem: id, app, doc_id, url, title, content, chunking_strategy, chunk_index
    """
    apps = apps or ["spotify", "whatsapp", "telegram"]

    all_chunks: list[dict] = []
    seen_hashes: set[str] = set()
    stats: dict[str, dict] = {}

    for app in apps:
        raw_docs = list(load_raw_docs(app))
        log.info(f"[{app.upper()}] {len(raw_docs)} documentos brutos carregados.")

        app_chunks_fixed = 0
        app_chunks_semantic = 0
        app_dupes = 0

        for doc in raw_docs:
            text = normalize(doc.get("content", ""))
            if not text:
                continue

            base_meta = {
                "app": doc["app"],
                "doc_id": doc["id"],
                "url": doc["url"],
                "title": doc.get("title", ""),
                "description": doc.get("description", ""),
                "collected_at": doc.get("collected_at", ""),
            }

            # Gera ambos os tipos de chunks (usados em estratégias diferentes)
            for strategy, chunks in [
                ("fixed", chunk_fixed(text)),
                ("semantic", chunk_semantic(text)),
            ]:
                for i, chunk_text in enumerate(chunks):
                    h = content_hash(chunk_text)
                    if h in seen_hashes:
                        app_dupes += 1
                        continue
                    seen_hashes.add(h)

                    chunk_id = f"{app}_{strategy}_{doc['id']}_{i:03d}"
                    all_chunks.append({
                        "id": chunk_id,
                        **base_meta,
                        "content": chunk_text,
                        "chunking_strategy": strategy,
                        "chunk_index": i,
                        "word_count": len(chunk_text.split()),
                        "token_estimate": token_count(chunk_text),
                        "content_hash": h,
                    })

                if strategy == "fixed":
                    app_chunks_fixed += len(chunks)
                else:
                    app_chunks_semantic += len(chunks)

        stats[app] = {
            "raw_docs": len(raw_docs),
            "chunks_fixed": app_chunks_fixed,
            "chunks_semantic": app_chunks_semantic,
            "duplicates_removed": app_dupes,
        }
        log.info(
            f"[{app.upper()}] Chunks fixo: {app_chunks_fixed} | "
            f"Semântico: {app_chunks_semantic} | Dupes removidos: {app_dupes}"
        )

    # Salva corpus completo
    corpus_path = PROCESSED_DIR / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Salva estatísticas
    stats["_total"] = {
        "total_chunks": len(all_chunks),
        "fixed_chunks": sum(s["chunks_fixed"] for s in stats.values() if "_" not in "".join(s.keys())),
        "semantic_chunks": sum(s.get("chunks_semantic", 0) for s in stats.values()),
    }
    stats_path = PROCESSED_DIR / "corpus_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log.info(f"\nCorpus salvo em {corpus_path}")
    log.info(f"Total de chunks gerados: {len(all_chunks)}")

    # Mostra exemplos
    log.info("\n=== EXEMPLO DE CHUNK (fixed) ===")
    for c in all_chunks[:1]:
        log.info(f"ID: {c['id']}")
        log.info(f"App: {c['app']} | URL: {c['url']}")
        log.info(f"Palavras: {c['word_count']} | Tokens estimados: {c['token_estimate']}")
        log.info(f"Conteúdo (200 chars): {c['content'][:200]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pré-processa e chunka o corpus coletado")
    parser.add_argument(
        "--apps",
        nargs="+",
        choices=["spotify", "whatsapp", "telegram"],
        default=None,
    )
    args = parser.parse_args()
    process_corpus(apps=args.apps)
