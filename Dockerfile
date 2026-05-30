FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/raw/ data/raw/
COPY preprocess/ preprocess/
COPY training/ training/
RUN mkdir -p model data/train \
    && python preprocess/Preprocessor.py \
    && python training/Trainer.py \
    && rm -rf mlruns mlflow.db

COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.EmoDetectorAPI:app", "--host", "0.0.0.0", "--port", "8000"]
