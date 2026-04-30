# RAG Evaluation for Android App Documentation

**Desafio técnico — Especialista em IA/LLM/NLP | SIDIA**

Avaliação experimental de estratégias de Retrieval-Augmented Generation (RAG) para perguntas e respostas sobre documentação pública de aplicativos Android.

---

## Visão Geral

Este projeto investiga qual estratégia de RAG produz melhores respostas para perguntas sobre documentações públicas dos aplicativos **Spotify**, **WhatsApp** e **Telegram**.

**3 estratégias comparadas:**

| ID    | Chunking   | Embedding Model          | Top-K | Re-ranking | Prompt        |
|-------|------------|--------------------------|-------|------------|---------------|
| RAG-A | Fixo 512   | all-MiniLM-L6-v2         | 3     | Não        | Simples       |
| RAG-B | Semântico  | bge-small-en-v1.5        | 5     | Sim        | Simples       |
| RAG-C | Semântico  | bge-large-en-v1.5        | 5     | Sim        | Estruturado   |

**Hipóteses científicas testadas:**
- **H1**: Chunking semântico melhora o recall em relação ao chunking fixo
- **H2**: Re-ranking aumenta a faithfulness das respostas geradas
- **H3**: Prompts estruturados reduzem alucinações (maior faithfulness)

---

## Estrutura do Projeto

```
rag_challenge/
├── data/
│   ├── raw/                    # Documentos brutos (JSONL por app)
│   └── processed/
│       ├── corpus.jsonl        # Corpus com chunks (fixo + semântico)
│       ├── ground_truth.csv    # Dataset de avaliação (30-50 Q&A)
│       └── corpus_stats.json   # Estatísticas da coleta
├── src/
│   ├── scraping/
│   │   ├── scraper.py          # Coleta de dados (crawler)
│   │   ├── preprocessor.py     # Chunking e limpeza
│   │   └── inspect_corpus.py   # Validação do corpus
│   ├── rag/
│   │   ├── indexer.py          # Geração de embeddings + FAISS
│   │   ├── strategies.py       # Implementação das 3 estratégias
│   │   └── reranker.py         # Re-ranking cross-encoder
│   └── evaluation/
│       ├── metrics.py          # F1, similaridade semântica, faithfulness
│       └── run_evaluation.py   # Pipeline completo de avaliação
├── results/
│   ├── predictions/            # Respostas geradas por cada estratégia
│   └── metrics_summary.json    # Tabela comparativa final
├── notebooks/
│   └── analysis.ipynb          # Análise exploratória e gráficos
├── article/                    # Artigo IEEE (.tex)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Requisitos

- Python 3.10+
- ~4 GB de espaço em disco (modelos de embedding)
- Chave da API Anthropic (variável de ambiente `ANTHROPIC_API_KEY`)
- Conexão à internet para coleta e download de modelos

---

## Instalação

### Opção 1 — Ambiente local

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/rag-android-docs.git
cd rag-android-docs

# Crie o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Instale as dependências
pip install -r requirements.txt

# Configure a API key
export ANTHROPIC_API_KEY="sua-chave-aqui"
```

### Opção 2 — Docker (recomendado para reprodutibilidade)

```bash
docker build -t rag-challenge .
docker run -e ANTHROPIC_API_KEY=sua-chave -v $(pwd)/data:/app/data rag-challenge
```

---

## Execução Passo a Passo

### Passo 1 — Coleta de dados

```bash
# Coleta todos os apps (Spotify, WhatsApp, Telegram)
python src/scraping/scraper.py

# Ou coleta um app específico
python src/scraping/scraper.py --apps spotify whatsapp

# Inspeciona o resultado
python src/scraping/inspect_corpus.py
```

**Saída esperada:** `data/raw/spotify.jsonl`, `data/raw/whatsapp.jsonl`, `data/raw/telegram.jsonl`

### Passo 2 — Pré-processamento

```bash
python src/scraping/preprocessor.py
```

**Saída esperada:** `data/processed/corpus.jsonl` (~1000+ chunks)

### Passo 3 — Indexação (embeddings + FAISS)

```bash
# Cria os índices vetoriais para as 3 estratégias
python src/rag/indexer.py
```

**Saída esperada:** `data/processed/index_rag_a/`, `index_rag_b/`, `index_rag_c/`

### Passo 4 — Experimentos

```bash
# Roda as 3 estratégias no dataset de avaliação
python src/evaluation/run_evaluation.py
```

**Saída esperada:** `results/predictions/*.jsonl` e `results/metrics_summary.json`

### Passo 5 — Análise

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Dataset de Avaliação

O arquivo `data/processed/ground_truth.csv` contém **31 pares pergunta/resposta** distribuídos:

| App       | Fácil | Médio | Difícil | Total |
|-----------|-------|-------|---------|-------|
| Spotify   | 4     | 5     | 1       | 10    |
| WhatsApp  | 5     | 4     | 1       | 10    |
| Telegram  | 4     | 5     | 2       | 11    |
| **Total** | **13**| **14**| **4**   | **31**|

**Processo de construção do ground truth:**
1. Cada pergunta foi gerada após leitura direta do documento de suporte oficial
2. A resposta foi escrita manualmente com base apenas no conteúdo da URL registrada
3. Dificuldade foi classificada por complexidade de raciocínio e número de documentos necessários

---

## Métricas de Avaliação

| Métrica              | Justificativa                                           |
|----------------------|---------------------------------------------------------|
| F1 token-level       | Mede overlap léxico com o ground truth                  |
| Similaridade semântica | Captura qualidade semântica além do match exato        |
| Faithfulness         | Verifica ancoragem da resposta no contexto recuperado   |
| Latência (ms)        | Custo operacional de cada estratégia                    |

---

## Dependências Principais

```
anthropic>=0.25.0        # Geração de respostas (LLM)
sentence-transformers    # Modelos de embedding
faiss-cpu                # Banco vetorial
langchain                # Orquestração RAG
beautifulsoup4           # Web scraping
requests                 # HTTP
lxml                     # Parser HTML
tqdm                     # Progress bars
pandas                   # Manipulação de dados
scikit-learn             # Métricas
nltk                     # F1, BLEU
```

---

## Reprodutibilidade

- Todos os random seeds fixados em `42`
- Versões de dependências fixadas em `requirements.txt`
- Logs completos em `results/experiment_log.json`
- Docker garante ambiente idêntico ao original

---

## Contato

Candidato: [Seu Nome]
Vaga: Especialista Técnico em IA — SIDIA
