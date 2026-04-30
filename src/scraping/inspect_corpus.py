"""
inspect_corpus.py — Valida e exibe estatísticas do corpus processado
Execute após o preprocessor.py para verificar a qualidade dos dados.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def inspect_raw():
    print("\n" + "=" * 60)
    print("CORPUS BRUTO (data/raw/)")
    print("=" * 60)
    for app in ["spotify", "whatsapp", "telegram"]:
        path = RAW_DIR / f"{app}.jsonl"
        if not path.exists():
            print(f"  {app}: ARQUIVO NÃO ENCONTRADO")
            continue
        docs = load_jsonl(path)
        words = [d.get("word_count", 0) for d in docs]
        print(f"\n  {app.upper()}")
        print(f"    Documentos:        {len(docs)}")
        print(f"    Total de palavras: {sum(words):,}")
        print(f"    Média palavras/doc:{sum(words)/max(len(words),1):.0f}")
        print(f"    Min / Max:         {min(words, default=0)} / {max(words, default=0)}")

        # Mostra 3 títulos de exemplo
        titles = [d.get("title", "sem título") for d in docs[:3]]
        print(f"    Exemplos de títulos:")
        for t in titles:
            print(f"      • {t[:70]}")


def inspect_processed():
    path = PROCESSED_DIR / "corpus.jsonl"
    if not path.exists():
        print("\nCorpus processado não encontrado. Rode o preprocessor.py primeiro.")
        return

    chunks = load_jsonl(path)

    print("\n" + "=" * 60)
    print("CORPUS PROCESSADO (data/processed/corpus.jsonl)")
    print("=" * 60)

    # Por app e estratégia
    by_app_strategy = defaultdict(list)
    for c in chunks:
        key = (c["app"], c["chunking_strategy"])
        by_app_strategy[key].append(c["word_count"])

    print(f"\n  Total de chunks: {len(chunks)}")
    print()

    for (app, strategy), word_counts in sorted(by_app_strategy.items()):
        avg = sum(word_counts) / max(len(word_counts), 1)
        print(f"  {app.upper()} [{strategy}]")
        print(f"    Chunks: {len(word_counts):>5}  |  "
              f"Média: {avg:>5.0f} palavras  |  "
              f"Min: {min(word_counts)}  Max: {max(word_counts)}")

    # Verifica qualidade mínima
    print("\n" + "-" * 40)
    print("VERIFICAÇÃO DE QUALIDADE")
    print("-" * 40)

    apps_found = set(c["app"] for c in chunks)
    expected = {"spotify", "whatsapp", "telegram"}
    missing = expected - apps_found

    checks = [
        ("Apps presentes", len(apps_found) == 3, f"Encontrados: {apps_found}"),
        ("Apps faltando", len(missing) == 0, f"Faltando: {missing}" if missing else "Nenhum"),
        ("Total chunks >= 300", len(chunks) >= 300, f"Total: {len(chunks)}"),
        ("Chunks sem conteúdo", sum(1 for c in chunks if not c.get("content")), ""),
        ("Chunks com URL", sum(1 for c in chunks if c.get("url")), f"{sum(1 for c in chunks if c.get('url'))}/{len(chunks)}"),
    ]

    for name, result, detail in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {name}: {detail}")

    # Exemplo de chunk
    print("\n" + "-" * 40)
    print("EXEMPLO DE CHUNK (semantic, primeiro do corpus)")
    print("-" * 40)
    sem_chunks = [c for c in chunks if c["chunking_strategy"] == "semantic"]
    if sem_chunks:
        ex = sem_chunks[0]
        print(f"  ID:       {ex['id']}")
        print(f"  App:      {ex['app']}")
        print(f"  URL:      {ex['url']}")
        print(f"  Palavras: {ex['word_count']}")
        print(f"  Conteúdo:")
        print("  " + "\n  ".join(ex["content"][:400].split("\n")))
        print("  ...")


if __name__ == "__main__":
    inspect_raw()
    inspect_processed()
