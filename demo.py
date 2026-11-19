#!/usr/bin/env python3
r"""Демонстрация работы системы распознавания и синонимического обезличивания ПДн (Вариант 24).

Показывает:
  1) синонимическое обезличивание текста со всеми 10 атрибутами Варианта 24;
  2) сохранение свойства идемпотентности при повторных упоминаниях в разных падежах;
  3) снятие морфологической омонимии двуполых имён, фамилий и двойных фамилий по контексту;
  4) полный сквозной конвейер (распознавание extract_pii -> обезличивание anon).

Запуск:
    .\.venv\Scripts\python.exe demo.py
    .\.venv\Scripts\python.exe demo.py --full
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import anon as A  # noqa: E402
from cli import load_extractor  # noqa: E402


TEXT = (
    "Индивидуальный предприниматель Иванов Иван Иванович, паспорт 45 10 123456, "
    "свидетельство о рождении III-МЮ 654321, ИНН 7707083893, ОГРНИП 321774600123462, "
    "код КЛАДР 7700000100000, IP 192.168.1.105, телефон +7 (999) 123-45-67, "
    "e-mail ivanov@example.com, карта 4084520359695002, СНИЛС 112-233-445 95."
)


def find_token(text: str, value: str) -> int:
    pattern = re.compile(
        r"(?<![A-Za-zА-Яа-яЁё0-9])"
        + re.escape(value)
        + r"(?![A-Za-zА-Яа-яЁё0-9])"
    )
    match = pattern.search(text)
    return match.start() if match else -1


def manual_spans(text: str) -> list[list[object]]:
    values = [
        ("SURNAME", "Иванов"),
        ("NAME", "Иван"),
        ("PATRONYMIC", "Иванович"),
        ("PASSPORT", "45 10 123456"),
        ("BIRTH_CERT", "III-МЮ 654321"),
        ("INN", "7707083893"),
        ("OGRNIP", "321774600123462"),
        ("KLADR", "7700000100000"),
        ("IP", "192.168.1.105"),
        ("PHONE", "+7 (999) 123-45-67"),
        ("EMAIL", "ivanov@example.com"),
        ("BANK_CARD", "4084520359695002"),
        ("SNILS", "112-233-445 95"),
    ]
    spans: list[list[object]] = []
    for attr, value in values:
        start = find_token(text, value)
        if start != -1:
            spans.append([attr, start, start + len(value)])
    return spans


def check_constraints(result: str) -> list[str]:
    checks: list[str] = []

    passport = re.search(r"(?<!\d)45 10 \d{6}(?!\d)", result)
    if passport:
        value = passport.group(0)
        checks.append(
            f"Паспорт: {value}, серия 45 10 сохранена, валиден: {A.validate_passport(value)}"
        )

    cert = re.search(r"III-МЮ \d{6}", result)
    if cert:
        value = cert.group(0)
        checks.append(
            f"Свидетельство о рождении: {value}, серия III-МЮ сохранена, валидно: {A.validate_birth_cert(value)}"
        )

    inn = re.search(r"(?<!\d)7707\d{6}(?!\d)", result)
    if inn:
        value = inn.group(0)
        checks.append(
            f"ИНН: {value}, префикс 7707, валиден: {A.validate_inn(value)}"
        )

    ogrnip = re.search(r"(?<!\d)32177\d{10}(?!\d)", result)
    if ogrnip:
        value = ogrnip.group(0)
        checks.append(
            f"ОГРНИП: {value}, префикс 32177, валиден (мод. 13): {A.validate_ogrnip(value)}"
        )

    kladr = re.search(r"(?<!\d)77\d{9}00(?!\d)", result)
    if kladr:
        value = kladr.group(0)
        checks.append(
            f"КЛАДР: {value}, регион 77, актуальность 00, валиден: {A.validate_kladr(value)}"
        )

    ip_match = re.search(r"(?<!\d)192\.168\.\d{1,3}\.\d{1,3}(?!\d)", result)
    if ip_match:
        value = ip_match.group(0)
        checks.append(
            f"IP-адрес: {value}, подсеть 192.168.x.x сохранена, валиден: {A.validate_ip(value)}"
        )

    card = re.search(r"(?<!\d)408452\d{6}5002(?!\d)", result)
    if card:
        value = card.group(0)
        checks.append(
            f"Карта: {value}, префикс 408452, суффикс 5002, Луна: {A.validate_luhn(value)}"
        )

    snils = re.search(r"\d{3}-\d{3}-\d{3} \d{2}", result)
    if snils:
        value = snils.group(0)
        checks.append(f"СНИЛС: {value}, валиден: {A.validate_snils(value)}")

    return checks


def demo_manual() -> None:
    spans = manual_spans(TEXT)
    print("Исходный текст:")
    print(" ", TEXT)
    print("\nРазметка сущностей ПДн:")
    for entity in spans:
        print("  ", entity)

    print("\nРезультаты трёх независимых запусков:")
    first_result = ""
    for index in range(1, 4):
        result = A.anon(TEXT, spans)
        if index == 1:
            first_result = result
        print(f"  {index}. {result}")

    print("\nПроверка ограничений (по первому запуску):")
    for line in check_constraints(first_result):
        print("  -", line)
    print(
        "  - Повторный запуск даёт другой результат:",
        first_result != A.anon(TEXT, spans),
    )


def demo_consistency_and_disambiguation() -> None:
    print("\n1) Согласованность повторных упоминаний (идемпотентность в разных падежах):")
    text = "Алексей Долматов передал договор Алексею Долматову"
    pairs = [
        ("NAME", "Алексей"),
        ("SURNAME", "Долматов"),
        ("NAME", "Алексею"),
        ("SURNAME", "Долматову"),
    ]
    spans: list[list[object]] = []
    position = 0
    for attr, value in pairs:
        start = text.index(value, position)
        spans.append([attr, start, start + len(value)])
        position = start + len(value)

    print(" ", text)
    for _ in range(2):
        print("  ->", A.anon(text, spans))

    print("\n2) Снятие омонимии двуполых имён и фамилий по синтаксическому контексту:")
    samples = [
        ("Саша подписал приказ", [["NAME", 0, 4]]),
        ("Саша подписала приказ", [["NAME", 0, 4]]),
        ("Документы получены от сотрудника Иванова", [["SURNAME", 33, 40]]),
        ("На совещании выступила гражданка Иванова", [["SURNAME", 33, 40]]),
        ("Иванов-Петров согласовал отчёт с Ивановым-Петровым", [["SURNAME", 0, 13], ["SURNAME", 33, 50]]),
    ]
    for sample_text, sample_spans in samples:
        print(f"  {sample_text!r} -> {A.anon(sample_text, sample_spans)!r}")


def demo_full_pipeline() -> None:
    print("\n" + "=" * 70)
    print("Полный конвейер: распознавание ПДн -> синонимическое обезличивание")
    print("=" * 70)

    try:
        extract_pii = load_extractor()
    except Exception as exc:  # noqa: BLE001
        print("Не удалось загрузить модуль распознавания:", exc)
        return

    entities = extract_pii(TEXT)
    print("Распознанные сущности:")
    for entity in entities:
        attr, start, end = entity
        print(f"  {entity} -> {TEXT[int(start):int(end)]!r}")

    result = A.anon(TEXT, entities)
    print("\nОбезличенный текст:")
    print(" ", result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Демонстрация системы распознавания и обезличивания ПДн")
    parser.add_argument(
        "--full",
        "--with-lab1",
        dest="full",
        action="store_true",
        help="дополнительно запустить полный конвейер с моделью распознавания RuBERT",
    )
    args = parser.parse_args()

    demo_manual()
    demo_consistency_and_disambiguation()
    if args.full:
        demo_full_pipeline()


if __name__ == "__main__":
    main()
