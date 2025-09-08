from datasets import load_dataset
dataset = load_dataset('imdb')
# 토크나이저 -모델
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
def preprocess_f(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
tokenized_datasets = dataset.map(preprocess_f, batched=True)

# 모델 설정
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# 학습 설정
import os
import torch
from transformers import TrainingArguments, Trainer

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)

tr_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    # --- 속도 향상을 위한 핵심 옵션들 ---
    fp16=True,  # 최신 GPU 성능 활용 (혼합 정밀도)
    dataloader_num_workers=4
)

trainer = Trainer(
    model=model,
    args=tr_args,
    train_dataset=tokenized_datasets['train'], 
    eval_dataset=tokenized_datasets['test']

)

trainer.train()