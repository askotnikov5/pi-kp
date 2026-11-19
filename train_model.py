"""Скрипт подготовки обучающего корпуса FactRuEval-2016 и дообучения модели RuBERT.

Объединяет:
  * парсинг исходных файлов FactRuEval-2016 (.tokens / .spans) в схему BIO;
  * генерацию синтетического расширения обучающей выборки;
  * дообучение трансформера DeepPavlov/rubert-base-cased и сохранение весов
    в каталог model/rubert_pii_ner.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import random

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_OUT_DIR = ROOT_DIR / "model" / "rubert_pii_ner"


def load_jsonl(file_path: Path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def parse_factrueval_dir(dir_path: str) -> list[dict]:
    dataset = []
    token_files = glob.glob(os.path.join(dir_path, "*.tokens"))

    for t_file in token_files:
        base_name = t_file[:-7]
        spans_file = base_name + ".spans"
        if not os.path.exists(spans_file):
            continue

        tokens_dict = {}
        tokens_list = []
        with open(t_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    t_id = parts[0]
                    t_text = " ".join(parts[3:])
                    tokens_dict[t_id] = t_text
                    tokens_list.append((t_id, t_text))

        token_tags = {t_id: "O" for t_id in tokens_dict}
        with open(spans_file, "r", encoding="utf-8") as f:
            for line in f:
                if "#" in line:
                    parts_before_comment = line.split("#")[0].strip().split()
                    if len(parts_before_comment) < 5:
                        continue
                    attr_type = parts_before_comment[1]
                    token_ids_in_span = parts_before_comment[4:]
                    if attr_type in ["name", "surname", "patronymic"]:
                        tag_prefix = "B-"
                        for t_id in token_ids_in_span:
                            if t_id in token_tags:
                                token_tags[t_id] = f"{tag_prefix}{attr_type.upper()}"
                                tag_prefix = "I-"

        doc_tokens = [t_text for _, t_text in tokens_list]
        doc_tags = [token_tags[t_id] for t_id, _ in tokens_list]
        if doc_tokens:
            dataset.append({"tokens": doc_tokens, "ner_tags": doc_tags})

    return dataset


def train() -> None:
    import evaluate
    import numpy as np
    from datasets import Dataset, DatasetDict
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    print("Загрузка данных FactRuEval и синтетического корпуса из data/...")
    train_data = load_jsonl(DATA_DIR / "factrueval_train.jsonl")
    synth_data = load_jsonl(DATA_DIR / "factrueval_synthetic.jsonl")
    test_data = load_jsonl(DATA_DIR / "factrueval_test.jsonl")

    combined_train_data = train_data + synth_data
    texts_train = [item["tokens"] for item in combined_train_data]
    tags_train = [item["ner_tags"] for item in combined_train_data]
    texts_test = [item["tokens"] for item in test_data]
    tags_test = [item["ner_tags"] for item in test_data]

    unique_tags = sorted(set(tag for sublist in tags_train for tag in sublist))
    label2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2label = {i: tag for tag, i in label2id.items()}

    numeric_tags_train = [[label2id[t] for t in sublist] for sublist in tags_train]
    numeric_tags_test = [[label2id[t] for t in sublist] for sublist in tags_test]

    dataset_dict = DatasetDict({
        "train": Dataset.from_dict({"tokens": texts_train, "ner_tags": numeric_tags_train}),
        "test": Dataset.from_dict({"tokens": texts_test, "ner_tags": numeric_tags_test}),
    })

    model_checkpoint = "DeepPavlov/rubert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            max_length=512,
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset_dict.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["tokens", "ner_tags"],
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(unique_tags),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [id2label[pred] for (pred, lbl) in zip(prediction, label) if lbl != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[lbl] for (pred, lbl) in zip(prediction, label) if lbl != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=str(ROOT_DIR / "model" / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=6,
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUT_DIR))
    print(f"Модель успешно сохранена в {MODEL_OUT_DIR}")


if __name__ == "__main__":
    train()
