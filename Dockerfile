FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python primeiro (cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baixa modelos NLTK necessários para métricas
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copia o código
COPY src/ ./src/
COPY data/processed/ground_truth.csv ./data/processed/ground_truth.csv
COPY README.md .

# Cria diretórios de saída
RUN mkdir -p data/raw data/processed results/predictions

# Variável de ambiente para a API key (passada em runtime)
ENV ANTHROPIC_API_KEY=""
ENV PYTHONPATH=/app

# Ponto de entrada padrão: roda o pipeline completo
CMD ["bash", "-c", \
    "python src/scraping/scraper.py && \
     python src/scraping/preprocessor.py && \
     python src/rag/indexer.py && \
     python src/evaluation/run_evaluation.py"]
