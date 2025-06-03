# src/api.py

import faiss
import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from typing import List

app = FastAPI()

# Загрузка модели и индекса
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"  # быстрая и эффективная
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

index = faiss.read_index("models/faiss_index.idx")
metadata = pd.read_csv("models/ticket_metadata.csv")


class TicketRequest(BaseModel):
    text: str
    top_k: int = 3


def get_embedding(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
        embedding = output.last_hidden_state[:, 0, :]  # [CLS]
    return embedding.cpu().numpy()


@app.post("/predict")
def predict_category(request: TicketRequest):
    query_vec = get_embedding(request.text)
    distances, indices = index.search(query_vec, request.top_k)

    results = []
    for idx in indices[0]:
        entry = metadata.iloc[idx]
        results.append({
            "id": int(entry["id"]),
            "text": entry["text"],
            "category": entry["category"]
        })

    # Мажоритарный класс
    categories = [r["category"] for r in results]
    majority = max(set(categories), key=categories.count)

    return {
        "input": request.text,
        "predicted_category": majority,
        "top_matches": results
    }
