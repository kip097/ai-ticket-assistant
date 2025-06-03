import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from peft import get_peft_model, LoraConfig, TaskType

# Пути
DATA_PATH = '../data/sample_tickets.csv'   # путь к датасету (относительно папки src)
MODEL_SAVE_PATH = '../models/ticket_classifier_model'

# 1. Загружаем данные
df = pd.read_csv(DATA_PATH)
print(f"Всего заявок: {len(df)}")

# 2. Преобразуем категории в числовые метки
labels = df['category'].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df['label'] = df['category'].map(label2id)

# 3. Делим на train/test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# 4. Загружаем токенизатор и модель (берём pretrained LLaMA, замените на нужную)
MODEL_NAME = 'bert-base-uncased'  # Замените на нужную модель, например 'meta-llama/Llama-2-7b-hf' (если доступна)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 5. Подготовка датасетов для transformers
def preprocess(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 6. Загружаем модель для классификации
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

# 7. Настройка LoRA (пример)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]  # для BERT, подберите для вашей модели
)

model = get_peft_model(model, lora_config)

# 8. Метрика для оценки
metric = load_metric('f1')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')

# 9. Параметры обучения
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
)

# 10. Создаём Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 11. Запускаем обучение
trainer.train()

# 12. Сохраняем модель
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print(f"Модель и токенизатор сохранены в {MODEL_SAVE_PATH}")

