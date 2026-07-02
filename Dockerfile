FROM python:3.11-slim

WORKDIR /app

# Install core dependencies only (no GPU, no sentence-transformers at runtime)
COPY requirements.txt .
RUN pip install --no-cache-dir numpy pandas scikit-learn tqdm xgboost

# Copy source code
COPY config.py .
COPY rank.py .
COPY src/ ./src/

# Copy pre-computed artifacts
COPY data/ ./data/

# Default command
ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "./candidates.jsonl", "--out", "./submission.csv"]
