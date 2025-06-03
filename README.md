ai-ticket-assistant/
├── README.md
├── requirements.txt
├── data/
│   └── sample_tickets.csv
├── model/
│   └── lora_finetuned_model/       # дообученная модель
├── faiss_index/
│   └── response_templates.index    # индекс шаблонов
├── app/
│   ├── main.py                     # FastAPI сервер
│   ├── classifier.py               # классификация
│   ├── responder.py               # генерация ответа
│   └── utils.py                    # вспомогательные функции
├── n8n/
│   └── workflow.json               # экспорт сценария из n8n
├── prompts/
│   └── response_prompt.txt         # шаблон генерации
└── eval/
    └── eval_classification.ipynb  # метрики качества
