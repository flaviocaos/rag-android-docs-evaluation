"""
indexer.py — Gera embeddings e índices FAISS para as 3 estratégias RAG
Entrada:  data/processed/corpus.jsonl
Saída:    data/processed/index_rag_a/  (chunking fixo  + all-MiniLM-L6)
          data/processed/index_rag_b/  (chunking semântico + bge-small)
          data/processed/index_rag_c/  (chunking semântico + bge-large)
"""

import json
import pickle
import logging
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CORPUS_PATH = Path("data/processed/corpus.jsonl")
INDEX_DIR   = Path("data/processed")

# ──────────────────────────────────────────────
# Configuração das 3 estratégias
# ──────────────────────────────────────────────

STRATEGIES = {
    "rag_a": {
        "chunking":    "fixed",
        "model_name":  "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Baseline: chunking fixo + MiniLM",
    },
    "rag_b": {
        "chunking":    "semantic",
        "model_name":  "BAAI/bge-small-en-v1.5",
        "description": "Intermediário: chunking semântico + BGE-small + re-ranking",
    },
    "rag_c": {
        "chunking":    "semantic",
        "model_name":  "BAAI/bge-large-en-v1.5",
        "description": "Avançado: chunking semântico + BGE-large + re-ranking + prompt estruturado",
    },
}


# ──────────────────────────────────────────────
# Carrega corpus
# ──────────────────────────────────────────────

def load_corpus(chunking_strategy: str) -> list[dict]:
    """Filtra chunks pelo tipo de chunking."""
    chunks = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                if chunk["chunking_strategy"] == chunking_strategy:
                    chunks.append(chunk)
    log.info(f"  Chunks carregados ({chunking_strategy}): {len(chunks)}")
    return chunks


# ──────────────────────────────────────────────
# Geração de embeddings
# ──────────────────────────────────────────────

def generate_embeddings(chunks: list[dict], model: SentenceTransformer,
                        batch_size: int = 32) -> np.ndarray:
    """Gera embeddings para todos os chunks em batches."""
    texts = [c["content"] for c in chunks]
    log.info(f"  Gerando embeddings para {len(texts)} chunks...")

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Normaliza para cosine similarity
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    log.info(f"  Embeddings gerados em {elapsed:.1f}s | Shape: {embeddings.shape}")
    return embeddings.astype("float32")


# ──────────────────────────────────────────────
# Construção do índice FAISS
# ──────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Cria índice FAISS Inner Product (equivalente a cosine similarity
    quando vetores são normalizados).
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine para vetores normalizados
    index.add(embeddings)
    log.info(f"  Índice FAISS criado: {index.ntotal} vetores, dimensão {dim}")
    return index


# ──────────────────────────────────────────────
# Persistência
# ──────────────────────────────────────────────

def save_index(strategy_id: str, index: faiss.Index,
               chunks: list[dict], model_name: str) -> Path:
    """Salva índice FAISS + metadados dos chunks."""
    out_dir = INDEX_DIR / f"index_{strategy_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Salva índice FAISS
    faiss.write_index(index, str(out_dir / "index.faiss"))

    # Salva chunks (metadados + conteúdo)
    with open(out_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    # Salva configuração
    config = {
        "strategy_id": strategy_id,
        "model_name":  model_name,
        "num_chunks":  len(chunks),
        "embedding_dim": index.d,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info(f"  Índice salvo em {out_dir}")
    return out_dir


# ──────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────

def build_all_indexes(strategies: list[str] | None = None):
    targets = strategies or list(STRATEGIES.keys())

    # Carrega corpus uma vez para cada tipo de chunking
    corpus_cache: dict[str, list[dict]] = {}

    for strategy_id in targets:
        config = STRATEGIES[strategy_id]
        log.info(f"\n{'='*50}")
        log.info(f"Indexando: {strategy_id.upper()} — {config['description']}")
        log.info(f"{'='*50}")

        # Carrega chunks (usa cache se já carregou esse tipo)
        chunking = config["chunking"]
        if chunking not in corpus_cache:
            corpus_cache[chunking] = load_corpus(chunking)
        chunks = corpus_cache[chunking]

        # Carrega modelo de embedding
        log.info(f"  Carregando modelo: {config['model_name']}")
        log.info(f"  (Primeira vez pode demorar — download do modelo)")
        t0 = time.time()
        model = SentenceTransformer(config["model_name"])
        log.info(f"  Modelo carregado em {time.time()-t0:.1f}s")

        # Gera embeddings
        embeddings = generate_embeddings(chunks, model)

        # Constrói índice FAISS
        index = build_faiss_index(embeddings)

        # Salva
        save_index(strategy_id, index, chunks, config["model_name"])

        log.info(f"  {strategy_id.upper()} concluído!")

    log.info(f"\n{'='*50}")
    log.info("INDEXAÇÃO COMPLETA!")
    log.info(f"{'='*50}")
    for sid in targets:
        idx_path = INDEX_DIR / f"index_{sid}"
        log.info(f"  {sid}: {idx_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gera índices FAISS para as estratégias RAG")
    parser.add_argument("--strategies", nargs="+",
                        choices=list(STRATEGIES.keys()), default=None)
    args = parser.parse_args()
    build_all_indexes(strategies=args.strategies)
