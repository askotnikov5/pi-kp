"""Оценка уровня сохранения семантического содержания с помощью официальной библиотеки bert-score.

Сравнивает исходный тестовый корпус data/test_variant24.jsonl (150 документов)
с его синонимически обезличенной копией и с замаскированной символами '*' копией
с использованием базовой русскоязычной модели DeepPavlov/rubert-base-cased.
Результаты сохраняются в data/evaluation_bertscore.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from bert_score import BERTScorer

import anon as A

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
BERTSCORE_MODEL = "DeepPavlov/rubert-base-cased"


def _summarize(P, R, F1) -> dict[str, float]:
    p_list = P.tolist()
    r_list = R.tolist()
    f1_list = F1.tolist()
    return {
        "precision": round(sum(p_list) / len(p_list), 4),
        "recall": round(sum(r_list) / len(r_list), 4),
        "f1": round(sum(f1_list) / len(f1_list), 4),
        "min_f1": round(min(f1_list), 4),
        "max_f1": round(max(f1_list), 4),
    }


def evaluate_bertscore() -> dict:
    dataset_path = DATA_DIR / "test_variant24.jsonl"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Не найден {dataset_path}. Сначала запустите generate_dataset.py"
        )

    cases = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    originals: list[str] = []
    synonymous_texts: list[str] = []
    entity_masked_texts: list[str] = []
    full_masked_texts: list[str] = []

    for case in cases:
        text = case["text"]
        spans = case["expected"]
        originals.append(text)
        synonymous_texts.append(A.anon(text, spans))
        entity_masked_texts.append(A.mask_pii(text, spans, full_text=False))
        full_masked_texts.append(A.mask_pii(text, spans, full_text=True))

    print("=" * 78)
    print("ОЦЕНКА СОХРАНЕНИЯ СМЫСЛА (ОФИЦИАЛЬНАЯ БИБЛИОТЕКА bert-score)")
    print(f"Базовая модель: {BERTSCORE_MODEL} | Документов: {len(originals)}")
    print("=" * 78)

    scorer = BERTScorer(
        model_type=BERTSCORE_MODEL,
        num_layers=9,
        lang="ru",
        rescale_with_baseline=False,
        device="cpu",
    )
    # У DeepPavlov/rubert-base-cased model_max_length не задан явно (1e30),
    # ограничиваем стандартной длиной окна BERT (512 токенов)
    scorer._tokenizer.model_max_length = 512

    P_syn, R_syn, F1_syn = scorer.score(synonymous_texts, originals, batch_size=16)
    P_ent, R_ent, F1_ent = scorer.score(entity_masked_texts, originals, batch_size=16)
    P_full, R_full, F1_full = scorer.score(full_masked_texts, originals, batch_size=16)

    syn_metrics = _summarize(P_syn, R_syn, F1_syn)
    ent_mask_metrics = _summarize(P_ent, R_ent, F1_ent)
    full_mask_metrics = _summarize(P_full, R_full, F1_full)

    print(
        f"1. Оригинал vs Синонимическое обезличивание: "
        f"P={syn_metrics['precision']:.4f}, R={syn_metrics['recall']:.4f}, F1={syn_metrics['f1']:.4f}"
    )
    print(
        f"2. Оригинал vs Маскирование ПДн ('*'):       "
        f"P={ent_mask_metrics['precision']:.4f}, R={ent_mask_metrics['recall']:.4f}, F1={ent_mask_metrics['f1']:.4f}"
    )
    print(
        f"3. Оригинал vs Полное маскирование ('*'):    "
        f"P={full_mask_metrics['precision']:.4f}, R={full_mask_metrics['recall']:.4f}, F1={full_mask_metrics['f1']:.4f}"
    )
    print("-" * 78)
    delta_ent = round(syn_metrics["f1"] - ent_mask_metrics["f1"], 4)
    delta_full = round(syn_metrics["f1"] - full_mask_metrics["f1"], 4)
    print(f"Прирост F1 BERTScore синонимического метода над маскированием ПДн:  +{delta_ent:.4f}")
    print(f"Прирост F1 BERTScore синонимического метода над полным маскированием: +{delta_full:.4f}")
    print("=" * 78)

    examples = []
    for idx in range(min(3, len(originals))):
        examples.append(
            {
                "original": originals[idx],
                "synonymous": synonymous_texts[idx],
                "entity_masked": entity_masked_texts[idx],
                "bertscore_f1_synonymous": round(float(F1_syn[idx]), 4),
                "bertscore_f1_entity_masked": round(float(F1_ent[idx]), 4),
            }
        )

    result = {
        "num_documents": len(originals),
        "library": "bert-score",
        "model": f"{BERTSCORE_MODEL} (num_layers=9)",
        "synonymous": syn_metrics,
        "entity_masking": ent_mask_metrics,
        "full_masking": full_mask_metrics,
        "delta_f1_vs_entity_masking": delta_ent,
        "delta_f1_vs_full_masking": delta_full,
        "examples": examples,
    }

    out_path = DATA_DIR / "evaluation_bertscore.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Результаты BERTScore сохранены в {out_path}")
    return result


if __name__ == "__main__":
    evaluate_bertscore()
