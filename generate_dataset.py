"""Генератор тестовых выборок для оценки распознавания и обезличивания ПДн (Вариант 24).

Создаёт в каталоге data/:
  * test_variant24.jsonl — комплексный корпус (150 текстов, по 100-150 экземпляров
    каждого из 10 атрибутов Варианта 24);
  * test_<attr>.jsonl — индивидуальные тестовые выборки по 60 примеров на атрибут.
"""

from __future__ import annotations

import json
from pathlib import Path
import random

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

SURNAMES_M = [
    "Иванов", "Смирнов", "Кузнецов", "Попов", "Васильев", "Петров",
    "Соколов", "Михайлов", "Новиков", "Федоров", "Морозов", "Волков",
    "Алексеев", "Лебедев", "Семенов", "Егоров", "Павлов", "Козлов",
    "Степанов", "Николаев", "Орлов", "Андреев", "Макаров", "Никитин",
    "Захаров", "Зайцев", "Соловьев", "Борисов", "Яковлев", "Григорьев",
    "Романов", "Воробьев", "Сергеев", "Кузьмин", "Фролов", "Александров",
    "Дмитриев", "Королев", "Гусев", "Киселев", "Ильин", "Максимов",
    "Поляков", "Сорокин", "Виноградов", "Ковалев", "Белов", "Медведев",
    "Антонов", "Тарасов", "Жуков", "Баранов", "Филиппов", "Комаров",
    "Давыдов", "Беляев", "Герасимов", "Богданов", "Осипов", "Сидоров",
]

NAMES_M = [
    "Александр", "Алексей", "Андрей", "Антон", "Аркадий", "Арсений",
    "Богдан", "Борис", "Вадим", "Валерий", "Василий", "Виктор",
    "Виталий", "Владимир", "Владислав", "Вячеслав", "Геннадий", "Георгий",
    "Глеб", "Григорий", "Даниил", "Денис", "Дмитрий", "Евгений",
    "Егор", "Иван", "Игорь", "Илья", "Кирилл", "Константин",
    "Леонид", "Максим", "Матвей", "Михаил", "Никита", "Николай",
    "Олег", "Павел", "Петр", "Роман", "Сергей", "Станислав",
    "Степан", "Тимур", "Федор", "Филипп", "Эдуард", "Юрий", "Ярослав",
]

PATRONYMICS_M = [
    "Александрович", "Алексеевич", "Андреевич", "Антонович", "Борисович",
    "Вадимович", "Валерьевич", "Васильевич", "Викторович", "Витальевич",
    "Владимирович", "Вячеславович", "Геннадьевич", "Георгиевич", "Григорьевич",
    "Данилович", "Денисович", "Дмитриевич", "Евгеньевич", "Егорович",
    "Иванович", "Игоревич", "Ильич", "Кириллович", "Константинович",
    "Леонидович", "Максимович", "Михайлович", "Николаевич", "Олегович",
    "Павлович", "Петрович", "Романович", "Сергеевич", "Станиславович",
    "Степанович", "Федорович", "Юрьевич", "Ярославович",
]

DOMAINS = [
    "example.com", "mail.ru", "yandex.ru", "company.org",
    "corp-info.ru", "mpei.ru", "techservice.net", "finbank.ru",
]

ROMAN_SERIES = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
CYR_SERIES = ["МЮ", "АК", "БГ", "ДН", "ЕП", "ЖР", "ИК", "ЛМ", "НР", "СТ", "УФ", "ЧШ"]

_INN_10 = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)


def gen_inn(rng: random.Random, length: int = 10) -> str:
    region = f"{rng.randint(1, 99):02d}{rng.randint(1, 99):02d}"
    if length == 10:
        body = [int(c) for c in region] + [rng.randint(0, 9) for _ in range(5)]
        c10 = sum(d * c for d, c in zip(body, _INN_10)) % 11 % 10
        return "".join(str(d) for d in body + [c10])
    body = [int(c) for c in region] + [rng.randint(0, 9) for _ in range(6)]
    c11 = sum(d * c for d, c in zip(body, _INN_11)) % 11 % 10
    body11 = body + [c11]
    c12 = sum(d * c for d, c in zip(body11, _INN_12)) % 11 % 10
    return "".join(str(d) for d in body11 + [c12])


def gen_bank_card(rng: random.Random) -> str:
    prefix = rng.choice(["4276", "4084", "5469", "5536", "2200", "2202"])
    body = [int(c) for c in prefix] + [rng.randint(0, 9) for _ in range(11)]
    total = 0
    for idx, digit in enumerate(reversed(body)):
        if idx % 2 == 0:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    check = (10 - (total % 10)) % 10
    return "".join(str(d) for d in body + [check])


def gen_ogrnip(rng: random.Random) -> str:
    year = f"{rng.randint(15, 25):02d}"
    region = f"{rng.randint(1, 92):02d}"
    serial = f"{rng.randint(100000000, 999999999):09d}"
    first_14 = f"3{year}{region}{serial}"
    check = (int(first_14) % 13) % 10
    return f"{first_14}{check}"


def gen_passport(rng: random.Random) -> str:
    reg = f"{rng.randint(1, 92):02d}"
    yr = f"{rng.randint(10, 24):02d}"
    num = f"{rng.randint(100000, 999999):06d}"
    if rng.random() < 0.5:
        return f"{reg} {yr} {num}"
    return f"{reg}{yr} {num}"


def gen_birth_cert(rng: random.Random) -> str:
    roman = rng.choice(ROMAN_SERIES)
    letters = rng.choice(CYR_SERIES)
    num = f"{rng.randint(100000, 999999):06d}"
    return f"{roman}-{letters} {num}"


def gen_kladr(rng: random.Random) -> str:
    region = f"{rng.randint(1, 92):02d}"
    district = f"{rng.randint(0, 45):03d}"
    city = f"{rng.randint(1, 50):03d}"
    settlement = f"{rng.randint(0, 30):03d}"
    if rng.random() < 0.5:
        return f"{region}{district}{city}{settlement}00"
    street = f"{rng.randint(1, 999):04d}"
    return f"{region}{district}{city}{settlement}{street}00"


def gen_ip(rng: random.Random) -> str:
    choice = rng.random()
    if choice < 0.35:
        return f"192.168.{rng.randint(1, 254)}.{rng.randint(2, 254)}"
    if choice < 0.65:
        return f"10.{rng.randint(0, 254)}.{rng.randint(1, 254)}.{rng.randint(2, 254)}"
    return f"{rng.randint(31, 191)}.{rng.randint(1, 250)}.{rng.randint(1, 250)}.{rng.randint(2, 254)}"


def gen_email(rng: random.Random, idx: int) -> str:
    prefixes = ["user", "client", "office", "contact", "admin", "manager", "partner", "support"]
    local = f"{rng.choice(prefixes)}{idx}_{rng.randint(10, 99)}"
    return f"{local}@{rng.choice(DOMAINS)}"


def gen_phone(rng: random.Random) -> str:
    return f"+7 ({rng.randint(900, 999)}) {rng.randint(100, 989)}-{rng.randint(10, 99)}-{rng.randint(10, 99)}"


def gen_snils(rng: random.Random) -> str:
    digits = [rng.randint(0, 9) for _ in range(9)]
    if all(d == 0 for d in digits):
        digits[-1] = 1
    total = sum(d * (9 - i) for i, d in enumerate(digits))
    if total < 100:
        check = total
    elif total in (100, 101):
        check = 0
    else:
        check = total % 101
        if check == 100:
            check = 0
    s = "".join(str(d) for d in digits)
    return f"{s[:3]}-{s[3:6]}-{s[6:9]} {check:02d}"


def _build_spans(text: str, pairs: list[tuple[str, str]]) -> list[list[object]]:
    spans = []
    for attr, val in pairs:
        pos = text.find(val)
        if pos != -1:
            spans.append([attr, pos, pos + len(val)])
    spans.sort(key=lambda x: int(x[1]))
    return spans


def _extract_factrueval_snippets(max_docs: int = 90) -> list[tuple[str, list[list[object]]]]:
    """Извлекает реальные фрагменты текстов из factRuEval2016 (data/factrueval_test.jsonl)
    с эталонной разметкой SURNAME и NAME.
    """
    src = DATA_DIR / "factrueval_test.jsonl"
    if not src.is_file():
        return []

    snippets: list[tuple[str, list[list[object]]]] = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tokens: list[str] = item.get("tokens", [])
            tags: list[str] = item.get("ner_tags", [])
            if not tokens or len(tokens) != len(tags):
                continue

            # Разбиваем документ на предложения и берём первое предложение с реальным NAME/SURNAME
            sent_start = 0
            picked = None
            for idx, tok in enumerate(tokens):
                is_end = tok in (".", "!", "?") or idx == len(tokens) - 1
                if is_end:
                    s_toks = tokens[sent_start:idx + 1]
                    s_tags = tags[sent_start:idx + 1]
                    has_fio = any(t in ("B-SURNAME", "B-NAME") for t in s_tags)
                    # Ограничиваем длину предложения, чтобы текст был естественным и помещался в 512 токенов
                    if has_fio and 6 <= len(s_toks) <= 45:
                        picked = (s_toks, s_tags)
                        break
                    sent_start = idx + 1

            if picked is None:
                continue

            s_toks, s_tags = picked
            # Склеиваем токены в естественную строку с точным отслеживанием символьных координат
            no_space_before = {".", ",", ";", ":", "!", "?", ")", "]", "»", "%"}
            no_space_after = {"(", "[", "«"}
            pieces: list[str] = []
            spans: list[list[object]] = []
            cursor = 0
            prev_tok = ""
            for tok, tag in zip(s_toks, s_tags):
                if pieces and tok not in no_space_before and prev_tok not in no_space_after:
                    pieces.append(" ")
                    cursor += 1
                t_start = cursor
                pieces.append(tok)
                cursor += len(tok)
                t_end = cursor
                if tag in ("B-SURNAME", "B-NAME"):
                    spans.append([tag.replace("B-", ""), t_start, t_end])
                prev_tok = tok

            text = "".join(pieces)
            if not text.endswith((".", "!", "?")):
                text += "."
            snippets.append((text, spans))
            if len(snippets) >= max_docs:
                break
    return snippets


def generate_datasets() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)

    variant24_attrs = [
        "SURNAME", "NAME", "INN", "PASSPORT", "BANK_CARD",
        "OGRNIP", "IP", "EMAIL", "BIRTH_CERT", "KLADR",
    ]
    all_supported_attrs = variant24_attrs + ["PATRONYMIC", "PHONE", "SNILS"]

    for attr in all_supported_attrs:
        dataset = []
        for idx in range(60):
            surname = rng.choice(SURNAMES_M)
            name = rng.choice(NAMES_M)
            patronymic = rng.choice(PATRONYMICS_M)
            email = gen_email(rng, idx)
            inn = gen_inn(rng, 10 if idx % 2 == 0 else 12)
            card = gen_bank_card(rng)
            ogrnip = gen_ogrnip(rng)
            passport = gen_passport(rng)
            birth_cert = gen_birth_cert(rng)
            kladr = gen_kladr(rng)
            ip = gen_ip(rng)
            phone = gen_phone(rng)
            snils = gen_snils(rng)

            templates = {
                "SURNAME": (f"Гражданин {surname} подал официальное заявление.", surname),
                "NAME": (f"Сотрудник {name} успешно выполнил поставленную задачу.", name),
                "PATRONYMIC": (f"Менеджер {patronymic} подписал трудовой договор.", patronymic),
                "EMAIL": (f"Пишите нам на корпоративную почту {email} для связи.", email),
                "INN": (f"Организация зарегистрирована и имеет ИНН {inn} в реестре.", inn),
                "BANK_CARD": (f"Оплата заказа прошла с банковской карты {card} успешно.", card),
                "PASSPORT": (f"Документ удостоверяет паспорт {passport} гражданина РФ.", passport),
                "OGRNIP": (f"Предприниматель внесен в ЕГРИП под номером ОГРНИП {ogrnip} официально.", ogrnip),
                "IP": (f"Подключение к серверу выполнено с сетевого узла {ip} по протоколу.", ip),
                "BIRTH_CERT": (f"В архиве ЗАГС зарегистрировано свидетельство о рождении {birth_cert} ребенка.", birth_cert),
                "KLADR": (f"Адресному объекту присвоен идентификационный код КЛАДР {kladr} в базе.", kladr),
                "PHONE": (f"Служба технической поддержки доступна по телефону {phone} круглосуточно.", phone),
                "SNILS": (f"Страховой номер индивидуального лицевого счета СНИЛС {snils} проверен в системе.", snils),
            }
            text, val = templates[attr]
            expected = _build_spans(text, [(attr, val)])
            dataset.append({"text": text, "expected": expected})

        filename = DATA_DIR / f"test_{attr.lower()}.jsonl"
        with open(filename, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Создан файл data/{filename.name} ({len(dataset)} примеров)")

    # Комплексный корпус Варианта 24 (150 текстов):
    # Часть 1 — реальные предложения из factRuEval2016 (data/factrueval_test.jsonl),
    # дополненные недостающими структурированными реквизитами Варианта 24;
    # Часть 2 — дополнительные деловые документы до 150 текстов.
    fact_snippets = _extract_factrueval_snippets(max_docs=90)
    kp_dataset = []

    for i, (base_text, base_spans) in enumerate(fact_snippets):
        inn = gen_inn(rng, 12 if i % 2 == 0 else 10)
        passport = gen_passport(rng)
        card = gen_bank_card(rng)
        ogrnip = gen_ogrnip(rng)
        ip = gen_ip(rng)
        email = gen_email(rng, i)
        birth_cert = gen_birth_cert(rng)
        kladr = gen_kladr(rng)

        if i % 2 == 0:
            addon = (
                f" В карточке субъекта указаны паспорт {passport}, свидетельство о рождении "
                f"{birth_cert}, ИНН {inn}, ОГРНИП {ogrnip}, код КЛАДР {kladr}, "
                f"почта {email}, сетевой узел {ip} и карта {card}."
            )
            addon_pairs = [
                ("PASSPORT", passport),
                ("BIRTH_CERT", birth_cert),
                ("INN", inn),
                ("OGRNIP", ogrnip),
                ("KLADR", kladr),
                ("EMAIL", email),
                ("IP", ip),
                ("BANK_CARD", card),
            ]
        else:
            addon = (
                f" По данным реестра реквизиты включают паспорт {passport}, ИНН {inn}, "
                f"ОГРНИП {ogrnip}, свидетельство о рождении {birth_cert} и e-mail {email}."
            )
            addon_pairs = [
                ("PASSPORT", passport),
                ("INN", inn),
                ("OGRNIP", ogrnip),
                ("BIRTH_CERT", birth_cert),
                ("EMAIL", email),
            ]

        offset = len(base_text)
        full_text = base_text + addon
        addon_spans = [
            [attr, offset + s, offset + e]
            for attr, s, e in _build_spans(addon, addon_pairs)
        ]
        expected = sorted(base_spans + addon_spans, key=lambda x: int(x[1]))
        kp_dataset.append({"text": full_text, "expected": expected, "source": "factRuEval2016+variant24"})

    extra_needed = max(0, 150 - len(kp_dataset))
    for j in range(extra_needed):
        idx = len(fact_snippets) + j
        surname = rng.choice(SURNAMES_M)
        name = rng.choice(NAMES_M)
        inn = gen_inn(rng, 12 if j % 2 == 0 else 10)
        passport = gen_passport(rng)
        card = gen_bank_card(rng)
        ogrnip = gen_ogrnip(rng)
        ip = gen_ip(rng)
        email = gen_email(rng, idx)
        birth_cert = gen_birth_cert(rng)
        kladr = gen_kladr(rng)

        if j % 2 == 0:
            text = (
                f"По данным реестра предприниматель {surname} {name} "
                f"(паспорт {passport}, свидетельство о рождении {birth_cert}, ИНН {inn}, "
                f"ОГРНИП {ogrnip}) зарегистрирован по коду КЛАДР {kladr}, "
                f"использует почту {email}, IP-адрес {ip} и карту {card}."
            )
            pairs = [
                ("SURNAME", surname),
                ("NAME", name),
                ("PASSPORT", passport),
                ("BIRTH_CERT", birth_cert),
                ("INN", inn),
                ("OGRNIP", ogrnip),
                ("KLADR", kladr),
                ("EMAIL", email),
                ("IP", ip),
                ("BANK_CARD", card),
            ]
        else:
            text = (
                f"Оператор {surname} {name} подтвердил операцию по карте {card} "
                f"для клиента с ИНН {inn}, ОГРНИП {ogrnip} и кодом КЛАДР {kladr}; "
                f"уведомление направлено на {email} с адреса {ip} после сверки паспорта {passport} "
                f"и акта {birth_cert}."
            )
            pairs = [
                ("SURNAME", surname),
                ("NAME", name),
                ("BANK_CARD", card),
                ("INN", inn),
                ("OGRNIP", ogrnip),
                ("KLADR", kladr),
                ("EMAIL", email),
                ("IP", ip),
                ("PASSPORT", passport),
                ("BIRTH_CERT", birth_cert),
            ]

        expected = _build_spans(text, pairs)
        kp_dataset.append({"text": text, "expected": expected, "source": "synthetic_variant24"})

    kp_path = DATA_DIR / "test_variant24.jsonl"
    with open(kp_path, "w", encoding="utf-8") as f:
        for item in kp_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    counts = {attr: 0 for attr in variant24_attrs}
    for item in kp_dataset:
        for attr, _, _ in item["expected"]:
            if attr in counts:
                counts[attr] += 1
    print(
        f"\nСоздан комплексный тестовый корпус data/{kp_path.name}: {len(kp_dataset)} текстов "
        f"(из них {len(fact_snippets)} на базе реальных предложений factRuEval2016)."
    )
    for attr, cnt in counts.items():
        print(f"  - {attr:10s}: {cnt} шт. (требование >= 50)")


if __name__ == "__main__":
    generate_datasets()
