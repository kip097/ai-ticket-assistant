import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

def prepare_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['text', 'category'])
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['category'])
    return df, label_encoder

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

def main():
    # 1. Загружаем данные
    df, label_encoder = prepare_data("data/sample_tickets.csv")

    # 2. Разбиваем на train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # 3. Преобразуем в Dataset для Hugging Face
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    # 4. Загружаем токенизатор и модель (берём LLaMA, например, 7b или меньшую для теста)
    model_name = "decapoda-research/llama-7b-hf"  # замените на доступную модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

    # 5. Токенизируем
    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # 6. Указываем колонки для обучения
    train_ds = train_ds.remove_columns(["text", "category", "__index_level_0__"])
    test_ds = test_ds.remove_columns(["text", "category", "__index_level_0__"])

    train_ds.set_format("torch")
    test_ds.set_format("torch")

    # 7. Добавляем LoRA для дообучения
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    # 8. Настраиваем Trainer
    training_args = TrainingArguments(
        output_dir="models/",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        num_train_epochs=3,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    # 9. Запускаем обучение
    trainer.train()

    # 10. Сохраняем модель и label encoder
    model.save_pretrained("models/classifier_lora")
    tokenizer.save_pretrained("models/classifier_lora")

    import joblib
    joblib.dump(label_encoder, "models/label_encoder.pkl")

if __name__ == "__main__":
    main()
