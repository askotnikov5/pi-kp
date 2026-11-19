#!/usr/bin/env python3
"""Интерфейс командной строки для распознавания и синонимического обезличивания ПДн.

Примеры:
    python cli.py "Андрей Яковлев отправился на прогулку"
    python cli.py Андрей Яковлев отправился на прогулку -v
    echo "Иван, ivanov@example.com" | python cli.py
    python cli.py "Андрей Яковлев" --spans "NAME:0:6,SURNAME:7:14"
    python cli.py "Андрей Яковлев" --repeat 3
    python cli.py "Андрей Яковлев" --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import anon as A  # noqa: E402


def parse_spans(value: str) -> list[list[object]]:
    """Разбирает --spans вида 'NAME:0:6,SURNAME:7:14'."""
    spans: list[list[object]] = []
    for chunk in re.split(r"[,;]", value):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.replace(":", " ").split()
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"не удалось разобрать спан {chunk!r}; ожидается ATTR:START:END"
            )
        attr, start, end = parts
        try:
            spans.append([attr, int(start), int(end)])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"START и END должны быть целыми числами: {chunk!r}"
            ) from exc
    return spans


def load_extractor():
    """Загружает функцию extract_pii из модуля recognizer."""
    from recognizer import extract_pii

    return extract_pii


def read_text(positional: list[str], option_text: str | None) -> str:
    if option_text is not None:
        return option_text
    if positional and positional != ["-"]:
        return " ".join(positional)
    if sys.stdin.isatty():
        raise SystemExit(
            "Нет текста. Передайте его аргументом или через stdin:\n"
            '  python cli.py "Андрей Яковлев отправился на прогулку"'
        )
    return sys.stdin.read().rstrip("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Распознавание и синонимическое обезличивание персональных данных (Вариант 24)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            '  %(prog)s "Андрей Яковлев отправился на прогулку"\n'
            "  %(prog)s Андрей Яковлев отправился на прогулку -v\n"
            '  %(prog)s "Андрей Яковлев" --spans "NAME:0:6,SURNAME:7:14"\n'
            '  %(prog)s "Андрей Яковлев" --repeat 3\n'
            '  %(prog)s "Андрей Яковлев" --json\n'
            '  echo "Иван, ivanov@example.com" | %(prog)s'
        ),
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="текст для обезличивания (можно без кавычек)",
    )
    parser.add_argument(
        "-t",
        "--text",
        dest="text_option",
        help="текст одной строкой (удобно, если он начинается с дефиса)",
    )
    parser.add_argument(
        "--spans",
        type=parse_spans,
        help='готовые спаны вместо автоматического распознавания: "NAME:0:6,SURNAME:7:14"',
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="сколько раз обезличить текст (по умолчанию 1)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="вывести результат и найденные сущности в JSON",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="показать найденные сущности в stderr",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="не загружать модель распознавания (требуется --spans)",
    )
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat должен быть >= 1")

    text = read_text(args.text, args.text_option)

    if args.spans is not None:
        entities = args.spans
    else:
        if args.no_model:
            parser.error("--no-model требует --spans")
        try:
            extract_pii = load_extractor()
            entities = extract_pii(text)
        except Exception as exc:  # noqa: BLE001
            print(
                f"Не удалось загрузить модуль распознавания: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            print(
                "Можно передать спаны вручную, например:\n"
                '  --spans "NAME:0:6,SURNAME:7:14"',
                file=sys.stderr,
            )
            return 2

    if args.verbose and not args.json:
        if entities:
            print("Найденные сущности:", file=sys.stderr)
            for attr, start, end in entities:
                s_i, e_i = int(start), int(end)
                snippet = text[s_i:e_i] if 0 <= s_i < e_i <= len(text) else ""
                print(f"  {str(attr):10} {s_i}:{e_i}  {snippet!r}", file=sys.stderr)
        else:
            print("ПДн не найдены — текст останется без изменений.", file=sys.stderr)

    results = [A.anon(text, entities) for _ in range(args.repeat)]

    if args.json:
        payload = {
            "text": text,
            "entities": [
                {"attr": attr, "start": start, "end": end}
                for attr, start, end in entities
            ],
            "results": results,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
