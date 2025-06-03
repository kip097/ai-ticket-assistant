import pandas as pd
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

def get_embeddings(texts, model, tokenizer, device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True, output_hidden_states=True)
        # Берём эмбеддинг CLS токена из последнего слоя
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = embeddings.cpu().numpy()
    return embeddings

def main():
    # 1. Загружаем данные
    df = pd.read_csv("data/sample_tickets.csv")

    # 2. Загружаем модель для эмбеддингов (без классификации!)
    model_name = "decapoda-research/llama-7b-hf"  # или другую подходящую модель
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # 3. Получаем эмбеддинги для всех текстов
    texts = df['text'].tolist()
    embeddings = get_embeddings(texts, model, tokenizer, device)

    # 4. Создаем FAISS индекс
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 5. Сохраняем индекс на диск
    faiss.write_index(index, "models/faiss_index.idx")

    # 6. Сохраняем тексты и метки для поиска
    df[['id', 'text', 'category']].to_csv("models/ticket_metadata.csv", index=False)

if __name__ == "__main__":
    main()
