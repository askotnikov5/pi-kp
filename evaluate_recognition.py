"""Оценка качества распознавания персональных данных (Token Classification BIO, Вариант 24).

Рассчитывает метрики Precision, Recall, F1 в разрезе каждого из 10 целевых
атрибутов Варианта 24, а также среднюю (Macro-F1) и средневзвешенную
(Weighted-F1) меру по всей программе в целом. Результаты сохраняются в
data/evaluation_recognition.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from recognizer import extract_pii

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

VARIANT24_ATTRIBUTES = [
    "SURNAME",
    "NAME",
    "INN",
    "PASSPORT",
    "BANK_CARD",
    "OGRNIP",
    "IP",
    "EMAIL",
    "BIRTH_CERT",
    "KLADR",
]


def load_jsonl(file_path: Path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    tokens = []
    for match in re.finditer(r"\S+", text):
        raw = match.group(0)
        s, e = match.start(), match.end()
        stripped = raw.strip(".,;:!?()[]{}\"'«»")
        if stripped:
            off = raw.find(stripped)
            s += off
            e = s + len(stripped)
            tokens.append((stripped, s, e))
    return tokens


def spans_to_bio_tags(
    tokens: list[tuple[str, int, int]],
    spans: list[list[object]],
) -> tuple[list[str], list[str]]:
    bio_tags: list[str] = []
    class_tags: list[str] = []
    active_span_idx: int | None = None

    sorted_spans = sorted(spans, key=lambda x: (int(x[1]), -(int(x[2]) - int(x[1]))))

    for _, t_start, t_end in tokens:
        matched_idx = None
        matched_attr = "O"
        for idx, (attr, s_start, s_end) in enumerate(sorted_spans):
            s_start_i, s_end_i = int(s_start), int(s_end)
            if not (t_end <= s_start_i or t_start >= s_end_i):
                matched_idx = idx
                matched_attr = str(attr)
                break

        if matched_idx is None:
            bio_tags.append("O")
            class_tags.append("O")
            active_span_idx = None
        else:
            prefix = "I-" if active_span_idx == matched_idx else "B-"
            bio_tags.append(f"{prefix}{matched_attr}")
            class_tags.append(matched_attr)
            active_span_idx = matched_idx

    return bio_tags, class_tags


def compute_metrics_dict(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate() -> dict:
    kp_file = DATA_DIR / "test_variant24.jsonl"
    if not kp_file.is_file():
        raise FileNotFoundError(
            f"Не найден {kp_file}. Сначала запустите generate_dataset.py"
        )

    kp_cases = load_jsonl(kp_file)

    token_stats = {
        attr: {"tp": 0, "fp": 0, "fn": 0} for attr in VARIANT24_ATTRIBUTES + ["O"]
    }
    entity_stats = {
        attr: {"tp": 0, "fp": 0, "fn": 0, "instances": 0}
        for attr in VARIANT24_ATTRIBUTES
    }

    total_tokens = 0
    for case in kp_cases:
        text = case["text"]
        expected = case["expected"]
        detected = extract_pii(text)

        exp_set = set((str(item[0]), int(item[1]), int(item[2])) for item in expected)
        det_set = set((str(item[0]), int(item[1]), int(item[2])) for item in detected)

        for attr, s, e in exp_set:
            if attr in entity_stats:
                entity_stats[attr]["instances"] += 1
                if (attr, s, e) in det_set:
                    entity_stats[attr]["tp"] += 1
                else:
                    entity_stats[attr]["fn"] += 1
        for attr, s, e in det_set - exp_set:
            if attr in entity_stats:
                entity_stats[attr]["fp"] += 1

        tokens = tokenize_with_spans(text)
        total_tokens += len(tokens)
        _, true_classes = spans_to_bio_tags(tokens, expected)
        _, pred_classes = spans_to_bio_tags(tokens, detected)

        for y_true, y_pred in zip(true_classes, pred_classes):
            if y_true == y_pred:
                if y_true in token_stats:
                    token_stats[y_true]["tp"] += 1
            else:
                if y_true in token_stats:
                    token_stats[y_true]["fn"] += 1
                if y_pred in token_stats:
                    token_stats[y_pred]["fp"] += 1

    per_file_entity_stats = {}
    for attr in VARIANT24_ATTRIBUTES:
        f_path = DATA_DIR / f"test_{attr.lower()}.jsonl"
        if not f_path.is_file():
            continue
        cases = load_jsonl(f_path)
        tp = fp = fn = 0
        for case in cases:
            text = case["text"]
            exp_set = set((str(x[0]), int(x[1]), int(x[2])) for x in case["expected"])
            det_set = set(
                (str(x[0]), int(x[1]), int(x[2]))
                for x in extract_pii(text)
                if str(x[0]) == attr
            )
            tp += len(exp_set & det_set)
            fp += len(det_set - exp_set)
            fn += len(exp_set - det_set)
        per_file_entity_stats[attr] = compute_metrics_dict(tp, fp, fn)

    print("=" * 78)
    print("ОЦЕНКА КАЧЕСТВА РАСПОЗНАВАНИЯ ПДн (ВАРИАНТ 24, TOKEN CLASSIFICATION BIO)")
    print(f"Корпус: data/{kp_file.name} | Документов: {len(kp_cases)} | Всего токенов: {total_tokens}")
    print("=" * 78)

    per_attribute_token_metrics = {}
    per_attribute_entity_metrics = {}

    macro_f1_sum = 0.0
    macro_p_sum = 0.0
    macro_r_sum = 0.0
    weighted_f1_num = 0.0
    weighted_p_num = 0.0
    weighted_r_num = 0.0
    pii_support_total = 0

    print(
        f"{'Атрибут':12s} | {'Экз.':>5s} | {'Токены':>6s} | {'Precision':>9s} | {'Recall':>8s} | {'F1-мера':>8s}"
    )
    print("-" * 78)

    for attr in VARIANT24_ATTRIBUTES:
        t_m = compute_metrics_dict(
            token_stats[attr]["tp"],
            token_stats[attr]["fp"],
            token_stats[attr]["fn"],
        )
        e_m = compute_metrics_dict(
            entity_stats[attr]["tp"],
            entity_stats[attr]["fp"],
            entity_stats[attr]["fn"],
        )
        e_m["instances"] = entity_stats[attr]["instances"]
        per_attribute_token_metrics[attr] = t_m
        per_attribute_entity_metrics[attr] = e_m

        supp = int(t_m["support"])
        macro_f1_sum += float(t_m["f1"])
        macro_p_sum += float(t_m["precision"])
        macro_r_sum += float(t_m["recall"])
        weighted_f1_num += supp * float(t_m["f1"])
        weighted_p_num += supp * float(t_m["precision"])
        weighted_r_num += supp * float(t_m["recall"])
        pii_support_total += supp

        print(
            f"{attr:12s} | {entity_stats[attr]['instances']:5d} | {supp:6d} | "
            f"{t_m['precision']:9.4f} | {t_m['recall']:8.4f} | {t_m['f1']:8.4f}"
        )

    o_metrics = compute_metrics_dict(
        token_stats["O"]["tp"],
        token_stats["O"]["fp"],
        token_stats["O"]["fn"],
    )
    print("-" * 78)
    print(
        f"{'O (фон)':12s} | {'-':>5s} | {int(o_metrics['support']):6d} | "
        f"{o_metrics['precision']:9.4f} | {o_metrics['recall']:8.4f} | {o_metrics['f1']:8.4f}"
    )

    n_classes = len(VARIANT24_ATTRIBUTES)
    macro_precision = round(macro_p_sum / n_classes, 4)
    macro_recall = round(macro_r_sum / n_classes, 4)
    macro_f1 = round(macro_f1_sum / n_classes, 4)

    weighted_precision = round(weighted_p_num / pii_support_total, 4) if pii_support_total else 0.0
    weighted_recall = round(weighted_r_num / pii_support_total, 4) if pii_support_total else 0.0
    weighted_f1 = round(weighted_f1_num / pii_support_total, 4) if pii_support_total else 0.0

    all_support = pii_support_total + int(o_metrics["support"])
    weighted_f1_with_o = round(
        (weighted_f1_num + int(o_metrics["support"]) * float(o_metrics["f1"])) / all_support,
        4,
    )
    macro_f1_with_o = round((macro_f1_sum + float(o_metrics["f1"])) / (n_classes + 1), 4)

    print("=" * 78)
    print(f"Средняя по классам ПДн (Macro):       Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1={macro_f1:.4f} (требование >= 0.80)")
    print(f"Средневзвешенная по ПДн (Weighted):   Precision={weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1={weighted_f1:.4f} (требование >= 0.85)")
    print(f"С учётом класса O (Macro / Weighted): Macro-F1={macro_f1_with_o:.4f}, Weighted-F1={weighted_f1_with_o:.4f}")
    print("=" * 78)

    report = {
        "dataset_file": f"data/{kp_file.name}",
        "num_documents": len(kp_cases),
        "total_tokens": total_tokens,
        "pii_tokens_total": pii_support_total,
        "per_attribute_token": per_attribute_token_metrics,
        "per_attribute_entity": per_attribute_entity_metrics,
        "per_file_entity": per_file_entity_stats,
        "class_O_token": o_metrics,
        "summary": {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "macro_f1_with_O": macro_f1_with_o,
            "weighted_f1_with_O": weighted_f1_with_o,
        },
    }

    out_json = DATA_DIR / "evaluation_recognition.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Результаты сохранены в {out_json}")
    return report


if __name__ == "__main__":
    evaluate()
