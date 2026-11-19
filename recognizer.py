"""Гибридный модуль распознавания персональных данных (Вариант 24).

Объединяет детерминированные правила и математическую валидацию контрольных
разрядов (ИНН, банковская карта, ОГРНИП, паспорт РФ, свидетельство о рождении,
код КЛАДР, IP-адрес, E-mail, СНИЛС, телефон) с нейросетевой моделью RuBERT
для извлечения неструктурированных антропонимов (Фамилия, Имя, Отчество).
"""

from __future__ import annotations

import ipaddress
from pathlib import Path
import re

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model" / "rubert_pii_ner"

_TOKENIZER = None
_MODEL = None
_ID2LABEL = None

_INN_COEFFS_10 = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_COEFFS_11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_COEFFS_12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)


def _ensure_model_weights() -> Path:
    """Собирает model.safetensors из частей model.safetensors.part*, если репозиторий только что склонирован."""
    weights_path = MODEL_DIR / "model.safetensors"
    if weights_path.is_file():
        return MODEL_DIR
    parts = sorted(MODEL_DIR.glob("model.safetensors.part*"))
    if parts:
        with open(weights_path, "wb") as out_f:
            for part in parts:
                with open(part, "rb") as in_f:
                    while True:
                        chunk = in_f.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)
    return MODEL_DIR


def _get_ner_model():
    """Ленивая загрузка токенизатора и дообученной модели RuBERT."""
    global _TOKENIZER, _MODEL, _ID2LABEL
    if _MODEL is not None:
        return _TOKENIZER, _MODEL, _ID2LABEL

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    model_path = str(_ensure_model_weights())
    _TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    _MODEL = AutoModelForTokenClassification.from_pretrained(model_path)
    _MODEL.eval()
    _ID2LABEL = _MODEL.config.id2label
    return _TOKENIZER, _MODEL, _ID2LABEL


def _inn_check_digit(digits, coeffs) -> int:
    return sum(d * c for d, c in zip(digits, coeffs)) % 11 % 10


def validate_luhn(card_number: str) -> bool:
    """Проверяет номер банковской карты (13-19 цифр) по алгоритму Луна."""
    digits = [int(c) for c in card_number if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def validate_inn(inn_str: str) -> bool:
    """Проверяет контрольные разряды ИНН (10 или 12 цифр) по правилам ФНС."""
    digits = [int(c) for c in inn_str if c.isdigit()]
    if len(digits) == 10:
        return digits[-1] == _inn_check_digit(digits[:9], _INN_COEFFS_10)
    if len(digits) == 12:
        return (
            digits[-2] == _inn_check_digit(digits[:10], _INN_COEFFS_11)
            and digits[-1] == _inn_check_digit(digits[:11], _INN_COEFFS_12)
        )
    return False


def validate_ogrnip(ogrnip_str: str) -> bool:
    """Проверяет 15-значный номер ОГРНИП и его контрольный разряд (остаток от деления на 13)."""
    digits_str = "".join(c for c in ogrnip_str if c.isdigit())
    if len(digits_str) != 15 or digits_str[0] not in ("3", "4"):
        return False
    first_14 = int(digits_str[:14])
    check = (first_14 % 13) % 10
    return int(digits_str[14]) == check


def validate_snils(snils_str: str) -> bool:
    """Проверяет СНИЛС (11 цифр и контрольное число)."""
    digits = [int(c) for c in snils_str if c.isdigit()]
    if len(digits) != 11:
        return False
    total = sum(d * (9 - i) for i, d in enumerate(digits[:9]))
    if total < 100:
        check = total
    elif total in (100, 101):
        check = 0
    else:
        check = total % 101
        if check == 100:
            check = 0
    return digits[9] * 10 + digits[10] == check


def validate_passport(passport_str: str) -> bool:
    """Проверяет серию (4 цифры, код региона 01..99) и номер (6 цифр) паспорта РФ."""
    digits = "".join(c for c in passport_str if c.isdigit())
    if len(digits) != 10:
        return False
    region = int(digits[:2])
    number = int(digits[4:])
    return 1 <= region <= 99 and 1 <= number <= 999999


def validate_birth_cert(cert_str: str) -> bool:
    """Проверяет серию (римские цифры + 2 кириллические буквы) и 6-значный номер свидетельства о рождении."""
    pattern = re.compile(
        r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|[IVXLCDM]{1,6})"
        r"[\-–—][А-ЯЁ]{2}(?:[\s№#\-]+)\d{6}$"
    )
    if not pattern.match(cert_str.strip()):
        return False
    digits = "".join(c for c in cert_str if c.isdigit())
    return len(digits) == 6 and int(digits) > 0


def validate_kladr(kladr_str: str) -> bool:
    """Проверяет 13- или 17-значный код КЛАДР (код региона 01..99 и признак актуальности 00..51)."""
    digits = "".join(c for c in kladr_str if c.isdigit())
    if len(digits) not in (13, 17):
        return False
    region = int(digits[:2])
    actual = int(digits[-2:])
    return 1 <= region <= 99 and 0 <= actual <= 51


def validate_ip(ip_str: str) -> bool:
    """Проверяет корректность IPv4 или IPv6 адреса."""
    candidate = ip_str.strip()
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return False


def extract_pii(text: str) -> list[list[object]]:
    """Извлекает персональные данные из текста в формате [[АТРИБУТ, start, end], ...]."""
    import torch

    results: list[list[object]] = []
    reserved_spans: list[tuple[int, int]] = []

    def is_reserved(start: int, end: int) -> bool:
        for r_start, r_end in reserved_spans:
            if not (end <= r_start or start >= r_end):
                return True
        return False

    def reserve(attr: str, start: int, end: int) -> bool:
        if not is_reserved(start, end):
            results.append([attr, start, end])
            reserved_spans.append((start, end))
            return True
        return False

    # 1. Извлекаем структурированные ПДн в порядке убывания специфичности

    # Свидетельство о рождении: например, III-МЮ 123456 или I-АБ № 654321
    birth_cert_pattern = re.compile(
        r"(?<![A-Za-zА-Яа-яЁё0-9])"
        r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|[IVXLCDM]{1,6})"
        r"[\-–—][А-ЯЁ]{2}(?:\s+№?\s*|\s*№\s*)\d{6}\b"
    )
    for match in birth_cert_pattern.finditer(text):
        if validate_birth_cert(match.group(0)):
            reserve("BIRTH_CERT", match.start(), match.end())

    # Банковская карта (4 группы по 4 цифры с разделителями или стандартный 16-значный номер по Луну)
    card_16_pattern = re.compile(
        r"\b(?:\d{4}[\s\-]){3}\d{4}\b|\b\d{16}\b"
    )
    for match in card_16_pattern.finditer(text):
        if validate_luhn(match.group(0)):
            reserve("BANK_CARD", match.start(), match.end())

    # ОГРНИП (15 цифр, начинается на 3 или 4, с валидным контрольным разрядом по модулю 13)
    ogrnip_pattern = re.compile(r"\b[34]\d{14}\b")
    for match in ogrnip_pattern.finditer(text):
        if validate_ogrnip(match.group(0)):
            reserve("OGRNIP", match.start(), match.end())

    # Код КЛАДР (13 или 17 цифр, код региона 01..99, признак актуальности 00..51)
    kladr_pattern = re.compile(r"\b\d{17}\b|\b\d{13}\b")
    for match in kladr_pattern.finditer(text):
        if validate_kladr(match.group(0)):
            reserve("KLADR", match.start(), match.end())

    # Банковская карта нестандартной длины (13-15 или 17-19 цифр по Луну, если не ОГРНИП/КЛАДР)
    card_other_pattern = re.compile(r"\b\d{13,19}\b")
    for match in card_other_pattern.finditer(text):
        if validate_luhn(match.group(0)):
            reserve("BANK_CARD", match.start(), match.end())

    # СНИЛС (XXX-XXX-XXX YY)
    snils_pattern = re.compile(r"\b\d{3}-\d{3}-\d{3}\s\d{2}\b")
    for match in snils_pattern.finditer(text):
        if validate_snils(match.group(0)):
            reserve("SNILS", match.start(), match.end())

    # ИНН (10 или 12 цифр с проверкой контрольных разрядов ФНС — до телефона, чтобы не было пересечений)
    inn_pattern = re.compile(r"\b\d{12}\b|\b\d{10}\b")
    for match in inn_pattern.finditer(text):
        if validate_inn(match.group(0)):
            reserve("INN", match.start(), match.end())

    # Номер телефона (с границей слева, чтобы не захватывать подстроку внутри длинных кодов)
    phone_pattern = re.compile(
        r"(?<!\d)(?:\+?7|8)\s*\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"
    )
    for match in phone_pattern.finditer(text):
        reserve("PHONE", match.start(), match.end())

    # Паспорт РФ: серия (XX XX или XXXX) и 6-значный номер (через пробел или №)
    passport_pattern = re.compile(
        r"\b\d{2}\s\d{2}(?:\s+№?\s*|\s*№\s*)\d{6}\b"
        r"|\b\d{4}(?:\s+№?\s*|\s*№\s*)\d{6}\b"
    )
    for match in passport_pattern.finditer(text):
        if validate_passport(match.group(0)):
            reserve("PASSPORT", match.start(), match.end())

    # E-mail
    email_pattern = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )
    for match in email_pattern.finditer(text):
        reserve("EMAIL", match.start(), match.end())

    # IP-адрес (IPv4 и IPv6)
    ipv4_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    for match in ipv4_pattern.finditer(text):
        if validate_ip(match.group(0)):
            reserve("IP", match.start(), match.end())

    ipv6_pattern = re.compile(
        r"(?<![A-Za-z0-9:])(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}(?![A-Za-z0-9:])"
    )
    for match in ipv6_pattern.finditer(text):
        if validate_ip(match.group(0)):
            reserve("IP", match.start(), match.end())

    # 2. Нейросеть RuBERT для неструктурированных ФИО (SURNAME, NAME, PATRONYMIC)
    word_matches = list(re.finditer(r"\S+", text))
    if word_matches:
        word_char_spans = []
        filtered_words = []

        for m in word_matches:
            raw_token = m.group(0)
            start_idx, end_idx = m.start(), m.end()
            stripped = raw_token.strip(".,;:!?()[]{}\"'«»")
            if stripped:
                offset = raw_token.find(stripped)
                start_idx += offset
                end_idx = start_idx + len(stripped)
                clean_token = stripped
            else:
                clean_token = raw_token

            if not is_reserved(start_idx, end_idx):
                filtered_words.append(clean_token)
                word_char_spans.append((start_idx, end_idx))

        if filtered_words:
            tokenizer, model, id2label = _get_ner_model()
            encoding = tokenizer(
                filtered_words,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            word_ids = encoding.word_ids(batch_index=0)

            with torch.no_grad():
                outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

            previous_word_idx = None
            for idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx != previous_word_idx:
                    pred_id = predictions[idx]
                    label = id2label[pred_id]
                    if label != "O":
                        start_pos, end_pos = word_char_spans[word_idx]
                        token_str = text[start_pos:end_pos]
                        # Имена собственные в русском тексте пишутся с заглавной буквы;
                        # это отсекает ложные срабатывания нейросети на нарицательные слова («паспорт» и т.п.)
                        if any(ch.isalpha() for ch in token_str) and not token_str.islower():
                            attr_name = label.replace("B-", "").replace("I-", "")
                            reserve(attr_name, start_pos, end_pos)
                    previous_word_idx = word_idx

    results.sort(key=lambda x: int(x[1]))
    return results


if __name__ == "__main__":
    test_text = (
        "Индивидуальный предприниматель Иванов Иван Иванович (ИНН: 7707083893, "
        "ОГРНИП: 321774600123462, паспорт: 45 10 123456, свидетельство о рождении: III-МЮ 654321, "
        "код КЛАДР: 7700000000000, IP: 192.168.1.105, e-mail: ivanov@example.com) "
        "оформил карту 4084520359695002."
    )
    print("Входной текст:\n", test_text)
    print("\nРаспознанные ПДн (формат: [<Атрибут>, <Начало>, <Конец>]):")
    pii_list = extract_pii(test_text)
    for item in pii_list:
        attr, start, end = item
        snippet = test_text[int(start):int(end)]
        print(f"  {item} -> '{snippet}'")
    print(f"\nВсего распознано атрибутов ПДн: {len(pii_list)}")
