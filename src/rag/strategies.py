"""
strategies.py — Implementação das 3 estratégias RAG
RAG-A: chunking fixo  + MiniLM      + top-k=3 + prompt simples
RAG-B: chunking sem.  + BGE-small   + top-k=5 + re-ranking + prompt simples
RAG-C: chunking sem.  + BGE-large   + top-k=5 + re-ranking + prompt estruturado
"""

import json
import pickle
import logging
import time
from pathlib import Path
from dataclasses import dataclass

import faiss
import numpy as np
import anthropic
from sentence_transformers import SentenceTransformer, CrossEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INDEX_DIR = Path("data/processed")

# ──────────────────────────────────────────────
# Estrutura de resultado
# ──────────────────────────────────────────────

@dataclass
class RAGResult:
    question:        str
    answer:          str
    retrieved_chunks: list[dict]
    strategy_id:     str
    retrieval_ms:    float
    generation_ms:   float
    total_ms:        float
    model_used:      str
    top_k:           int
    reranked:        bool


# ──────────────────────────────────────────────
# Carregador de índice
# ──────────────────────────────────────────────

def load_index(strategy_id: str) -> tuple[faiss.Index, list[dict], dict]:
    """Carrega índice FAISS, chunks e configuração."""
    idx_dir = INDEX_DIR / f"index_{strategy_id}"

    index  = faiss.read_index(str(idx_dir / "index.faiss"))
    with open(idx_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(idx_dir / "config.json") as f:
        config = json.load(f)

    log.info(f"Índice {strategy_id} carregado: {index.ntotal} vetores")
    return index, chunks, config


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

PROMPT_SIMPLE = """Based on the following context, answer the question concisely and accurately.
If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""


PROMPT_STRUCTURED = """You are a helpful assistant specialized in Android app documentation.
Your task is to answer user questions based ONLY on the provided context.

Instructions:
- Answer directly and concisely based on the context
- If the context does not contain the answer, explicitly state: "The provided documentation does not contain information about this topic."
- Do not add information beyond what is in the context
- Be specific and actionable in your response

Context:
{context}

Question: {question}

Answer (based strictly on the context above):"""


# ──────────────────────────────────────────────
# Classe base RAG
# ──────────────────────────────────────────────

class BaseRAG:
    def __init__(self, strategy_id: str, model_name: str,
                 top_k: int, use_reranking: bool, prompt_template: str):
        self.strategy_id     = strategy_id
        self.model_name      = model_name
        self.top_k           = top_k
        self.use_reranking   = use_reranking
        self.prompt_template = prompt_template

        # Carrega índice
        self.index, self.chunks, self.config = load_index(strategy_id)

        # Carrega modelo de embedding
        log.info(f"[{strategy_id}] Carregando embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)

        # Carrega re-ranker se necessário
        self.reranker = None
        if use_reranking:
            reranker_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            log.info(f"[{strategy_id}] Carregando re-ranker: {reranker_name}")
            self.reranker = CrossEncoder(reranker_name)

        # Cliente Anthropic
        self.client = anthropic.Anthropic()

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        """Recupera os top-k chunks mais relevantes."""
        # Gera embedding da pergunta
        q_emb = self.embedder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        # Busca no FAISS (busca por maiores inner products = cosine similarity)
        k_search = min(top_k * 3, len(self.chunks))  # Busca mais para re-ranker filtrar
        scores, indices = self.index.search(q_emb, k_search)

        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                chunk = self.chunks[idx].copy()
                chunk["retrieval_score"] = float(score)
                retrieved.append(chunk)

        return retrieved[:top_k] if not self.use_reranking else retrieved

    def rerank(self, question: str, chunks: list[dict], top_k: int) -> list[dict]:
        """Re-rankeia os chunks usando cross-encoder."""
        if not self.reranker or not chunks:
            return chunks[:top_k]

        pairs = [(question, c["content"]) for c in chunks]
        scores = self.reranker.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return chunks[:top_k]

    def build_context(self, chunks: list[dict]) -> str:
        """Monta o contexto a partir dos chunks recuperados."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}: {chunk['app'].title()} — {chunk['title'][:60]}]\n"
                f"{chunk['content'][:800]}"
            )
        return "\n\n---\n\n".join(parts)

    def generate(self, question: str, context: str) -> str:
        """Gera a resposta usando Claude."""
        prompt = self.prompt_template.format(
            context=context,
            question=question,
        )
        message = self.client.messages.create(
            model="claude-haiku-4-5-20251001",  # Rápido e barato para experimentos
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    def answer(self, question: str) -> RAGResult:
        """Pipeline completo: retrieve → (rerank) → generate."""
        t_start = time.time()

        # Retrieval
        t0 = time.time()
        candidates = self.retrieve(question, self.top_k)
        t_retrieval = (time.time() - t0) * 1000

        # Re-ranking (se habilitado)
        if self.use_reranking:
            chunks = self.rerank(question, candidates, self.top_k)
        else:
            chunks = candidates[:self.top_k]

        # Geração
        context = self.build_context(chunks)
        t0 = time.time()
        answer_text = self.generate(question, context)
        t_generation = (time.time() - t0) * 1000

        t_total = (time.time() - t_start) * 1000

        return RAGResult(
            question=question,
            answer=answer_text,
            retrieved_chunks=chunks,
            strategy_id=self.strategy_id,
            retrieval_ms=round(t_retrieval, 1),
            generation_ms=round(t_generation, 1),
            total_ms=round(t_total, 1),
            model_used=self.model_name,
            top_k=self.top_k,
            reranked=self.use_reranking,
        )


# ──────────────────────────────────────────────
# As 3 estratégias concretas
# ──────────────────────────────────────────────

class RAGA(BaseRAG):
    """
    RAG-A — Baseline
    Chunking fixo 512 tokens + all-MiniLM-L6-v2 + top-k=3 + prompt simples
    Hipótese: performance base sem otimizações
    """
    def __init__(self):
        super().__init__(
            strategy_id     = "rag_a",
            model_name      = "sentence-transformers/all-MiniLM-L6-v2",
            top_k           = 3,
            use_reranking   = False,
            prompt_template = PROMPT_SIMPLE,
        )


class RAGB(BaseRAG):
    """
    RAG-B — Intermediário
    Chunking semântico + BGE-small-en + top-k=5 + re-ranking + prompt simples
    Hipótese: chunking semântico + re-ranking melhoram faithfulness (H1 + H2)
    """
    def __init__(self):
        super().__init__(
            strategy_id     = "rag_b",
            model_name      = "BAAI/bge-small-en-v1.5",
            top_k           = 5,
            use_reranking   = True,
            prompt_template = PROMPT_SIMPLE,
        )


class RAGC(BaseRAG):
    """
    RAG-C — Avançado
    Chunking semântico + BGE-large + top-k=5 + re-ranking + prompt estruturado
    Hipótese: prompt estruturado reduz alucinações (H3)
    """
    def __init__(self):
        super().__init__(
            strategy_id     = "rag_c",
            model_name      = "BAAI/bge-large-en-v1.5",
            top_k           = 5,
            use_reranking   = True,
            prompt_template = PROMPT_STRUCTURED,
        )


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def get_strategy(strategy_id: str) -> BaseRAG:
    strategies = {
        "rag_a": RAGA,
        "rag_b": RAGB,
        "rag_c": RAGC,
    }
    if strategy_id not in strategies:
        raise ValueError(f"Estratégia desconhecida: {strategy_id}. Use: {list(strategies.keys())}")
    return strategies[strategy_id]()


# ──────────────────────────────────────────────
# Teste rápido
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERRO: defina a variável ANTHROPIC_API_KEY antes de rodar.")
        print("Windows: set ANTHROPIC_API_KEY=sua-chave-aqui")
        exit(1)

    question = "How do I download songs for offline listening on Spotify?"
    print(f"\nTeste rápido — Pergunta: {question}\n")

    for sid in ["rag_a", "rag_b", "rag_c"]:
        print(f"\n{'='*50}")
        print(f"Estratégia: {sid.upper()}")
        print(f"{'='*50}")
        try:
            rag = get_strategy(sid)
            result = rag.answer(question)
            print(f"Resposta: {result.answer[:300]}")
            print(f"Latência: {result.total_ms:.0f}ms "
                  f"(retrieval: {result.retrieval_ms:.0f}ms | "
                  f"geração: {result.generation_ms:.0f}ms)")
            print(f"Chunks usados: {len(result.retrieved_chunks)}")
        except Exception as e:
            print(f"Erro: {e}")
