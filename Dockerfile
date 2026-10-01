FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY analysis_functions.py .
COPY run_analysis.py .
COPY tests/data/sample_reviews.csv tests/data/sample_reviews.csv

CMD ["python", "run_analysis.py", "--data", "tests/data/sample_reviews.csv"]