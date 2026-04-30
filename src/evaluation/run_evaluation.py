"""
run_evaluation.py — Pipeline completo de avaliação das 3 estratégias RAG
Entrada:  data/processed/ground_truth.csv
          data/processed/index_rag_{a,b,c}/
Saída:    results/predictions/{strategy}.jsonl
          results/metrics_summary.json
          results/metrics_table.csv
"""

import json
import csv
import logging
import time
import os
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

GROUND_TRUTH_PATH = Path("data/processed/ground_truth.csv")
RESULTS_DIR       = Path("results/predictions")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Tokenização simples por palavras."""
    import re
    return re.findall(r"\b\w+\b", text.lower())


def compute_f1(prediction: str, ground_truth: str) -> float:
    """F1 token-level entre predição e ground truth."""
    pred_tokens = set(tokenize(prediction))
    gt_tokens   = set(tokenize(ground_truth))

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match normalizado (case-insensitive, strip)."""
    return float(prediction.strip().lower() == ground_truth.strip().lower())


def compute_semantic_similarity(prediction: str, ground_truth: str,
                                 model: SentenceTransformer) -> float:
    """Similaridade de cosseno entre embeddings da predição e ground truth."""
    embs = model.encode(
        [prediction, ground_truth],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    similarity = float(np.dot(embs[0], embs[1]))
    return round(max(0.0, similarity), 4)


def compute_faithfulness(answer: str, retrieved_chunks: list[dict],
                          threshold: int = 4) -> float:
    """
    Faithfulness aproximada: verifica quantas n-gramas da resposta
    aparecem nos chunks recuperados.
    Valor entre 0 (sem suporte) e 1 (totalmente ancorada no contexto).
    """
    answer_tokens = tokenize(answer)
    context_text  = " ".join(c["content"] for c in retrieved_chunks)
    context_tokens = tokenize(context_text)
    context_set   = set(context_tokens)

    if not answer_tokens:
        return 0.0

    # Verifica n-gramas de tamanho `threshold`
    total_ngrams   = 0
    matched_ngrams = 0

    for i in range(len(answer_tokens) - threshold + 1):
        ngram = tuple(answer_tokens[i:i + threshold])
        total_ngrams += 1
        # Verifica se todos os tokens do n-grama estão no contexto
        if all(t in context_set for t in ngram):
            matched_ngrams += 1

    if total_ngrams == 0:
        return 0.0

    return round(matched_ngrams / total_ngrams, 4)


# ──────────────────────────────────────────────
# Carrega ground truth
# ──────────────────────────────────────────────

def load_ground_truth() -> list[dict]:
    questions = []
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)
    log.info(f"Ground truth carregado: {len(questions)} perguntas")
    return questions


# ──────────────────────────────────────────────
# Avaliação de uma estratégia
# ──────────────────────────────────────────────

def evaluate_strategy(strategy_id: str, questions: list[dict],
                       sem_model: SentenceTransformer) -> dict:
    """Roda todas as perguntas em uma estratégia e calcula métricas."""
    from src.rag.strategies import get_strategy

    log.info(f"\n{'='*55}")
    log.info(f"Avaliando: {strategy_id.upper()}")
    log.info(f"{'='*55}")

    rag = get_strategy(strategy_id)

    results     = []
    f1_scores   = []
    em_scores   = []
    sem_scores  = []
    faith_scores = []
    latencies   = []

    pred_path = RESULTS_DIR / f"{strategy_id}.jsonl"

    with open(pred_path, "w", encoding="utf-8") as pred_file:
        for i, q in enumerate(questions, 1):
            question   = q["question"]
            ground_truth = q["ground_truth_answer"]
            app        = q["app"]
            difficulty = q["difficulty"]

            log.info(f"  [{i}/{len(questions)}] {app} | {difficulty} | {question[:60]}...")

            try:
                result = rag.answer(question)
                answer = result.answer

                # Calcula métricas
                f1   = compute_f1(answer, ground_truth)
                em   = compute_exact_match(answer, ground_truth)
                sem  = compute_semantic_similarity(answer, ground_truth, sem_model)
                faith = compute_faithfulness(answer, result.retrieved_chunks)
                lat  = result.total_ms

                f1_scores.append(f1)
                em_scores.append(em)
                sem_scores.append(sem)
                faith_scores.append(faith)
                latencies.append(lat)

                record = {
                    "question":       question,
                    "ground_truth":   ground_truth,
                    "prediction":     answer,
                    "app":            app,
                    "difficulty":     difficulty,
                    "f1":             f1,
                    "exact_match":    em,
                    "semantic_sim":   sem,
                    "faithfulness":   faith,
                    "latency_ms":     lat,
                    "retrieval_ms":   result.retrieval_ms,
                    "generation_ms":  result.generation_ms,
                    "num_chunks":     len(result.retrieved_chunks),
                    "strategy_id":    strategy_id,
                }
                pred_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.append(record)

                log.info(f"    F1={f1:.3f} | Sem={sem:.3f} | Faith={faith:.3f} | {lat:.0f}ms")

            except Exception as e:
                log.error(f"    ERRO: {e}")
                # Registra falha
                record = {
                    "question": question, "ground_truth": ground_truth,
                    "prediction": "", "app": app, "difficulty": difficulty,
                    "f1": 0.0, "exact_match": 0.0, "semantic_sim": 0.0,
                    "faithfulness": 0.0, "latency_ms": 0.0,
                    "retrieval_ms": 0.0, "generation_ms": 0.0,
                    "num_chunks": 0, "strategy_id": strategy_id, "error": str(e),
                }
                pred_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.append(record)

            time.sleep(0.5)  # Evita rate limit da API

    # Agrega métricas
    def safe_mean(lst):
        return round(float(np.mean(lst)), 4) if lst else 0.0

    summary = {
        "strategy_id":       strategy_id,
        "num_questions":     len(questions),
        "avg_f1":            safe_mean(f1_scores),
        "avg_exact_match":   safe_mean(em_scores),
        "avg_semantic_sim":  safe_mean(sem_scores),
        "avg_faithfulness":  safe_mean(faith_scores),
        "avg_latency_ms":    safe_mean(latencies),
        "median_latency_ms": round(float(np.median(latencies)), 1) if latencies else 0.0,
        "by_app": {},
        "by_difficulty": {},
    }

    # Breakdown por app
    for app in ["spotify", "whatsapp", "telegram"]:
        app_results = [r for r in results if r["app"] == app]
        if app_results:
            summary["by_app"][app] = {
                "n":              len(app_results),
                "avg_f1":         safe_mean([r["f1"] for r in app_results]),
                "avg_semantic":   safe_mean([r["semantic_sim"] for r in app_results]),
                "avg_faithfulness": safe_mean([r["faithfulness"] for r in app_results]),
            }

    # Breakdown por dificuldade
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            summary["by_difficulty"][diff] = {
                "n":            len(diff_results),
                "avg_f1":       safe_mean([r["f1"] for r in diff_results]),
                "avg_semantic": safe_mean([r["semantic_sim"] for r in diff_results]),
            }

    log.info(f"\n  RESULTADO {strategy_id.upper()}:")
    log.info(f"  F1={summary['avg_f1']:.3f} | "
             f"Sem={summary['avg_semantic_sim']:.3f} | "
             f"Faith={summary['avg_faithfulness']:.3f} | "
             f"Lat={summary['avg_latency_ms']:.0f}ms")

    return summary


# ──────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────

def run_evaluation(strategies: list[str] | None = None):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nERRO: variável ANTHROPIC_API_KEY não definida!")
        print("Windows: set ANTHROPIC_API_KEY=sua-chave-aqui")
        print("Depois rode novamente este script.")
        return

    targets = strategies or ["rag_a", "rag_b", "rag_c"]
    questions = load_ground_truth()

    # Modelo semântico para avaliação (compartilhado entre estratégias)
    log.info("Carregando modelo de avaliação semântica...")
    sem_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    all_summaries = {}

    for strategy_id in targets:
        summary = evaluate_strategy(strategy_id, questions, sem_model)
        all_summaries[strategy_id] = summary

    # Salva resultados finais
    out_path = Path("results/metrics_summary.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    # Tabela CSV para o artigo
    csv_path = Path("results/metrics_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Strategy", "F1", "Semantic Sim", "Faithfulness",
                         "Exact Match", "Avg Latency (ms)"])
        for sid, s in all_summaries.items():
            writer.writerow([
                sid.upper(),
                f"{s['avg_f1']:.3f}",
                f"{s['avg_semantic_sim']:.3f}",
                f"{s['avg_faithfulness']:.3f}",
                f"{s['avg_exact_match']:.3f}",
                f"{s['avg_latency_ms']:.0f}",
            ])

    # Imprime tabela final
    print("\n" + "="*65)
    print("TABELA COMPARATIVA FINAL")
    print("="*65)
    print(f"{'Strategy':<12} {'F1':>6} {'Sem.Sim':>8} {'Faith.':>8} {'EM':>6} {'Lat(ms)':>9}")
    print("-"*65)
    for sid, s in all_summaries.items():
        print(f"{sid.upper():<12} "
              f"{s['avg_f1']:>6.3f} "
              f"{s['avg_semantic_sim']:>8.3f} "
              f"{s['avg_faithfulness']:>8.3f} "
              f"{s['avg_exact_match']:>6.3f} "
              f"{s['avg_latency_ms']:>9.0f}")
    print("="*65)
    print(f"\nResultados salvos em:")
    print(f"  {out_path}")
    print(f"  {csv_path}")
    print(f"  {RESULTS_DIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Avalia as estratégias RAG")
    parser.add_argument("--strategies", nargs="+",
                        choices=["rag_a", "rag_b", "rag_c"], default=None)
    args = parser.parse_args()
    run_evaluation(strategies=args.strategies)
