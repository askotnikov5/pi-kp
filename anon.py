"""Синонимическое обезличивание персональных данных (Вариант 24).

Публичный API:
    anon(text) -> (anonymized_text, attributes)
    anon(text, attributes) -> str
    mask_pii(text, attributes=..., full_text=False) -> str

`attributes` — список найденных сущностей:
    [["SURNAME", 0, 6], ["NAME", 7, 11], ...]

Поддерживаются все целевые атрибуты Варианта 24 (SURNAME, NAME, INN,
PASSPORT, BANK_CARD, OGRNIP, IP, EMAIL, BIRTH_CERT, KLADR), а также
дополнительные атрибуты PATRONYMIC, PHONE, SNILS.
"""

from __future__ import annotations

import ipaddress
from numbers import Integral
from pathlib import Path
import re
import secrets
import string
import sys
from typing import Iterable, NamedTuple, Sequence

try:  # pymorphy3 — основной морфологический анализатор
    import pymorphy3
except ImportError:  # pragma: no cover - на случай отсутствия зависимости
    pymorphy3 = None  # type: ignore[assignment]


__all__ = [
    "anon",
    "anonymize",
    "mask_pii",
    "validate_birth_cert",
    "validate_inn",
    "validate_ip",
    "validate_kladr",
    "validate_luhn",
    "validate_ogrnip",
    "validate_passport",
    "validate_snils",
]


# ---------------------------------------------------------------------------
# Инициализация зависимостей
# ---------------------------------------------------------------------------

_RNG = secrets.SystemRandom()

if pymorphy3 is not None:
    _MORPH = pymorphy3.MorphAnalyzer()
else:  # pragma: no cover
    _MORPH = None


# ---------------------------------------------------------------------------
# Атрибуты и их синонимы
# ---------------------------------------------------------------------------

FIO_ATTRS = frozenset({"SURNAME", "NAME", "PATRONYMIC"})

_ATTR_ALIASES = {
    # фамилия
    "SURNAME": "SURNAME",
    "LASTNAME": "SURNAME",
    "LAST_NAME": "SURNAME",
    "ФАМИЛИЯ": "SURNAME",
    # имя
    "NAME": "NAME",
    "FIRSTNAME": "NAME",
    "FIRST_NAME": "NAME",
    "ИМЯ": "NAME",
    # отчество
    "PATRONYMIC": "PATRONYMIC",
    "MIDDLENAME": "PATRONYMIC",
    "MIDDLE_NAME": "PATRONYMIC",
    "ОТЧЕСТВО": "PATRONYMIC",
    # e-mail
    "EMAIL": "EMAIL",
    "E_MAIL": "EMAIL",
    "MAIL": "EMAIL",
    "ПОЧТА": "EMAIL",
    "ЭЛЕКТРОННАЯ_ПОЧТА": "EMAIL",
    # ИНН
    "INN": "INN",
    "ИНН": "INN",
    # телефон
    "PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "ТЕЛЕФОН": "PHONE",
    "НОМЕР_ТЕЛЕФОНА": "PHONE",
    # банковская карта
    "BANK_CARD": "BANK_CARD",
    "BANKCARD": "BANK_CARD",
    "CARD": "BANK_CARD",
    "CREDIT_CARD": "BANK_CARD",
    "БАНКОВСКАЯ_КАРТА": "BANK_CARD",
    "КАРТА": "BANK_CARD",
    # СНИЛС (бонус: есть в ЛР №1)
    "SNILS": "SNILS",
    "СНИЛС": "SNILS",
    # паспорт РФ (Вариант 24)
    "PASSPORT": "PASSPORT",
    "PASSPORT_NUMBER": "PASSPORT",
    "ПАСПОРТ": "PASSPORT",
    "НОМЕР_ПАСПОРТА": "PASSPORT",
    # ОГРНИП (Вариант 24)
    "OGRNIP": "OGRNIP",
    "ОГРНИП": "OGRNIP",
    "НОМЕР_ОГРНИП": "OGRNIP",
    # IP-адрес (Вариант 24)
    "IP": "IP",
    "IP_ADDRESS": "IP",
    "IP_АДРЕС": "IP",
    "IPV4": "IP",
    "IPV6": "IP",
    # свидетельство о рождении (Вариант 24)
    "BIRTH_CERT": "BIRTH_CERT",
    "BIRTH_CERTIFICATE": "BIRTH_CERT",
    "СВИДЕТЕЛЬСТВО_О_РОЖДЕНИИ": "BIRTH_CERT",
    "НОМЕР_СВИДЕТЕЛЬСТВА_О_РОЖДЕНИИ": "BIRTH_CERT",
    # код КЛАДР (Вариант 24)
    "KLADR": "KLADR",
    "КЛАДР": "KLADR",
    "КОД_КЛАДР": "KLADR",
}

_KIND = {
    "SURNAME": "Surn",
    "NAME": "Name",
    "PATRONYMIC": "Patr",
}


# ---------------------------------------------------------------------------
# Вспомогательные структуры
# ---------------------------------------------------------------------------

class _Entity(NamedTuple):
    attr: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


class _Gram(NamedTuple):
    case: str
    gender: str
    number: str


class _Info(NamedTuple):
    """Разобранная информация об одном ФИО-токене."""

    attr: str
    word: str
    score: float
    cases: frozenset[str]
    genders: frozenset[str]
    numbers: frozenset[str]
    case_gender_pairs: frozenset[tuple[str, str]]
    case_guess: str
    gender_guess: str
    number_guess: str
    lemma: str


class _Person:
    """Обезличенный «человек»: одна замена на все его упоминания."""

    __slots__ = ("gender", "attrs", "bases")

    def __init__(self, gender: str, attrs: dict[str, str]) -> None:
        self.gender = gender
        self.attrs = dict(attrs)
        self.bases: dict[tuple[str, int], str] = {}

    def base(self, attr: str, shape: int = 1) -> str:
        """Возвращает замену для атрибута с учётом числа дефисных частей.

        shape=1 — обычная фамилия/имя, shape=2 — «Иванов-Петров».
        Одна и та же основа переиспользуется во всех падежах.
        """
        shape = max(1, int(shape))
        key = (attr, shape)
        if key in self.bases:
            return self.bases[key]

        if shape == 1:
            # Если человек уже встречался с дефисной формой, берём её
            # первую часть, чтобы «Иванов-Петров» и «Иванов» согласовались.
            for (other_attr, other_shape), value in self.bases.items():
                if other_attr == attr and other_shape > 1:
                    return value.split("-")[0]
            value = _replacement_base(attr, self.gender, self.attrs.get(attr, ""))
        else:
            parts = [self.base(attr, 1)]
            attempts = 0
            while len(parts) < shape and attempts < 100:
                attempts += 1
                candidate = _pick_base(attr, self.gender, "sing")
                if candidate not in parts:
                    parts.append(candidate)
            while len(parts) < shape:
                parts.append(_pick_base(attr, self.gender, "sing"))
            value = "-".join(parts)

        self.bases[key] = value
        return value


def _lemma_compatible(old: str, new: str) -> bool:
    """Совместимы ли леммы: полное имя и инициал «И.» — да."""
    if old == new:
        return True
    if len(old) == 1 and new.startswith(old):
        return True
    if len(new) == 1 and old.startswith(new):
        return True
    # «Иванов-Петров» и краткое «Иванов» — тоже один человек.
    old_parts = old.split("-")
    new_parts = new.split("-")
    if len(old_parts) > 1 and new in old_parts:
        return True
    if len(new_parts) > 1 and old in new_parts:
        return True
    return False


class _PersonRegistry:
    """Сопоставляет повторные упоминания одного человека одной замене."""

    def __init__(self) -> None:
        self.people: list[_Person] = []

    def resolve(self, attrs: dict[str, str], gender: str) -> _Person:
        for person in self.people:
            if person.gender != gender:
                continue
            shared = person.attrs.keys() & attrs.keys()
            if not shared:
                continue
            # Все общие атрибуты должны совпадать. Разные фамилии при
            # одинаковом имени — это разные люди.
            if all(
                _lemma_compatible(person.attrs[key], attrs[key])
                for key in shared
            ):
                for key, value in attrs.items():
                    old = person.attrs.get(key)
                    # Полное имя информативнее инициала.
                    if old is None or (len(old) == 1 and len(value) > 1):
                        person.attrs[key] = value
                return person
        person = _Person(gender, attrs)
        self.people.append(person)
        return person


class _FioPlan(NamedTuple):
    gram: _Gram
    base: str


_DEFAULT_CASE = "nomn"
_DEFAULT_GENDER = "masc"
_DEFAULT_NUMBER = "sing"


# ---------------------------------------------------------------------------
# Нормализация входа
# ---------------------------------------------------------------------------

def _normalize_attr(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    key = re.sub(r"[\s\-]+", "_", raw.strip().upper())
    key = key.replace("Ё", "Е")
    return _ATTR_ALIASES.get(key)


def _as_index(value: object) -> int | None:
    """Приводит индекс к int, отбрасывая bool и откровенный мусор."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
        return int(value)
    return None


def _coerce_entities(text: str, attributes: object) -> list[_Entity]:
    """Аккуратно приводит пользовательский список к списку _Entity.

    Всё, что не похоже на корректный кортеж (атрибут, start, end), молча
    пропускается: обезличивание не должно падать из-за «мусора» на входе.
    """
    if attributes is None or isinstance(attributes, (str, bytes)):
        return []

    # Разрешаем передать одну сущность без внешнего списка: ["NAME", 0, 4]
    if (
        isinstance(attributes, (list, tuple))
        and len(attributes) == 3
        and isinstance(attributes[0], str)
        and _as_index(attributes[1]) is not None
        and _as_index(attributes[2]) is not None
    ):
        attributes = [attributes]

    try:
        iterator = iter(attributes)  # type: ignore[arg-type]
    except TypeError:
        return []

    result: list[_Entity] = []
    for item in iterator:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        attr = _normalize_attr(item[0])
        if attr is None:
            continue
        start = _as_index(item[1])
        end = _as_index(item[2])
        if start is None or end is None:
            continue
        if start < 0 or end > len(text) or end <= start:
            continue
        result.append(_Entity(attr, start, end))
    return result


def _select_non_overlapping(entities: Iterable[_Entity]) -> list[_Entity]:
    """Сортирует сущности и убирает пересечения.

    При конфликте выигрывает более длинная сущность (она «специфичнее»),
    при равной длине — та, что идёт раньше.
    """
    ordered = sorted(entities, key=lambda e: (e.start, -e.length))
    selected: list[_Entity] = []
    cursor = -1
    for entity in ordered:
        if entity.start < cursor:
            continue
        selected.append(entity)
        cursor = entity.end
    return selected


# ---------------------------------------------------------------------------
# Морфология ФИО
# ---------------------------------------------------------------------------

def _morph_parse(word: str):
    if _MORPH is None:
        return []
    try:
        return _MORPH.parse(word)
    except Exception:  # pragma: no cover - защита от неожиданных входов
        return []


def _candidates(attr: str, word: str):
    """Возвращает разборы, относящиеся к нужному типу слова."""
    parses = _morph_parse(word)
    if not parses:
        return []
    kind = _KIND.get(attr)
    if kind:
        tagged = [p for p in parses if kind in p.tag.grammemes]
        if tagged:
            return tagged
    nouns = [p for p in parses if p.tag.POS == "NOUN"]
    return nouns or parses


def _guess_gender(word: str, attr: str, parses) -> str:
    """Эвристика рода, когда pymorphy не даёт однозначного ответа."""
    lower = word.lower().replace("ё", "е")

    if attr == "SURNAME":
        fem_nom = (
            "ова", "ева", "ина", "ына", "ская", "цкая", "ая", "яя",
        )
        mas = (
            "ов", "ев", "ин", "ын", "ский", "цкий", "ий", "ый",
            "ко", "ук", "юк", "ич", "ец",
        )
        if lower.endswith(fem_nom) and any(p.tag.gender == "femn" for p in parses):
            return "femn"
        if lower.endswith("ой"):
            # Толстой (м.) vs Ивановой (ж. р., косвенный падеж)
            if not any(
                p.tag.gender == "masc" and p.tag.case == "nomn" for p in parses
            ):
                return "femn"
            return "masc"
        if lower.endswith(mas):
            return "masc"
        if lower.endswith(("а", "я")):
            return "femn"
        genders = {p.tag.gender for p in parses}
        if "femn" in genders and "masc" not in genders:
            return "femn"
        return "masc"

    if attr == "PATRONYMIC":
        return "femn" if lower.endswith("на") else "masc"

    # NAME
    male_exceptions = {
        "никита", "илья", "фома", "лука", "кузьма", "савва", "данила",
        "добрыня", "гаврила", "михайла", "иона", "савватий",
    }
    if lower in male_exceptions:
        return "masc"
    if lower.endswith(("а", "я")):
        return "femn"
    genders = {p.tag.gender for p in parses}
    if "femn" in genders and "masc" not in genders:
        return "femn"
    return "masc"


def _parse_genders(parse) -> tuple[str, ...]:
    """Извлекает допустимые роды из разбора pymorphy3 (включая общий род ms-f)."""
    if parse.tag.gender in ("masc", "femn"):
        return (parse.tag.gender,)
    if "ms-f" in parse.tag.grammemes or "Ms-f" in parse.tag.grammemes:
        return ("masc", "femn")
    return ()


def _info_for_entity(
    text: str,
    entity: _Entity,
    gender_hint: str | None = None,
    case_hint: str | None = None,
) -> _Info:
    raw_word = text[entity.start:entity.end]
    # Распознаватель из ЛР №1 иногда включает в спан запятую или точку.
    # Для анализа берём только буквенное ядро, а при замене пунктуация
    # всё равно сохраняется.
    _, core, _ = _strip_nonletters(raw_word)
    word = core or raw_word
    parses = _candidates(entity.attr, word)
    score = max((p.score for p in parses), default=0.0)

    cases = frozenset(p.tag.case for p in parses if p.tag.case)
    genders = frozenset(g for p in parses for g in _parse_genders(p))
    numbers = frozenset(p.tag.number for p in parses if p.tag.number)
    cg_pairs = frozenset(
        (p.tag.case, g)
        for p in parses
        if p.tag.case
        for g in _parse_genders(p)
    )

    # Контекстная подсказка позволяет разрешить омонимию:
    # 1) для имён общего рода (Саша, Женя, Валя) — выбрать род по глаголу/существительному;
    # 2) для фамилий вида «Иванова» без имени — отличить ж.р. им.п. («гражданка Иванова»)
    #    от м.р. род./вин.п. («у Иванова», «от сотрудника Иванова»).
    if gender_hint in ("masc", "femn") and gender_hint in genders:
        gender_guess = gender_hint
    elif (
        case_hint
        and entity.attr == "SURNAME"
        and case_hint != "nomn"
        and (case_hint, "masc") in cg_pairs
        and (case_hint, "femn") not in cg_pairs
    ):
        gender_guess = "masc"
    else:
        gender_guess = _guess_gender(word, entity.attr, parses)

    # Уточняем падеж/число по разборам, совпадающим с выбранным родом:
    # это важно для «Иванова», где первый разбор — мужской родительный,
    # а нам нужен женский именительный (если контекст не указал мужской род).
    same_gender = [p for p in parses if gender_guess in _parse_genders(p)]
    pool = same_gender or parses

    if case_hint:
        case_matched = [p for p in pool if p.tag.case == case_hint]
        if case_matched:
            pool = case_matched

    def top(field: str, default: str) -> str:
        values = [getattr(p.tag, field) for p in pool if getattr(p.tag, field)]
        if values:
            return values[0]
        all_values = [getattr(p.tag, field) for p in parses if getattr(p.tag, field)]
        return all_values[0] if all_values else default

    # Лемма нужна, чтобы узнать одно и то же ФИО в разных падежах:
    # «Алексей» и «Алексею» должны дать «алексей».
    lemma_source = pool or parses
    lemma = ""
    for parse in lemma_source:
        if parse.tag.case == "nomn" and parse.tag.number == "sing":
            lemma = parse.normal_form
            break
    if not lemma and lemma_source:
        lemma = lemma_source[0].normal_form
    if not lemma:
        lemma = word
    lemma = lemma.lower().replace("ё", "е")

    return _Info(
        attr=entity.attr,
        word=raw_word,
        score=score,
        cases=cases,
        genders=genders,
        numbers=numbers,
        case_gender_pairs=cg_pairs,
        case_guess=top("case", _DEFAULT_CASE),
        gender_guess=gender_guess,
        number_guess=top("number", _DEFAULT_NUMBER),
        lemma=lemma,
    )


def _intersect(sets: Sequence[frozenset[str]]) -> frozenset[str]:
    if not sets:
        return frozenset()
    result = set(sets[0])
    for value in sets[1:]:
        result &= value
    return frozenset(result)


def _choose(infos: Sequence[_Info], field: str) -> str:
    """Выбирает общий род/падеж/число для группы ФИО.

    Сначала пробуем пересечение множеств допустимых значений. Если оно
    неоднозначно, доверяем более информативному атрибуту: NAME → PATRONYMIC
    → SURNAME. Это позволяет корректно обрабатывать «Иванова Мария».
    """
    sets = [getattr(info, "cases" if field == "case" else field + "s") for info in infos]
    common = _intersect(sets)
    if len(common) == 1:
        return next(iter(common))

    for preferred in ("NAME", "PATRONYMIC", "SURNAME"):
        for info in infos:
            if info.attr == preferred:
                value = (
                    info.gender_guess
                    if field == "gender"
                    else getattr(info, field + "_guess")
                )
                if value:
                    return value

    best = max(infos, key=lambda i: i.score)
    value = (
        best.gender_guess if field == "gender" else getattr(best, field + "_guess")
    )
    defaults = {
        "case": _DEFAULT_CASE,
        "gender": _DEFAULT_GENDER,
        "number": _DEFAULT_NUMBER,
    }
    return value or defaults[field]


def _can_join(previous: _Entity, current: _Entity, text: str) -> bool:
    gap = text[previous.end:current.start]
    return len(gap) <= 6 and re.search(r"[A-Za-zА-Яа-яЁё0-9]", gap) is None


def _duplicate_role(group: Sequence[_Entity], current: _Entity) -> bool:
    # В одном ФИО не может быть двух имён/фамилий/отчеств. Это защищает
    # от склейки двух соседних людей в одну группу.
    return any(entity.attr == current.attr for entity in group)


def _group_fio(text: str, entities: Sequence[_Entity]) -> list[list[_Entity]]:
    groups: list[list[_Entity]] = []
    current: list[_Entity] = []
    for entity in entities:
        if (
            current
            and _can_join(current[-1], entity, text)
            and len(current) < 3
            and not _duplicate_role(current, entity)
        ):
            current.append(entity)
        else:
            if current:
                groups.append(current)
            current = [entity]
    if current:
        groups.append(current)
    return groups


def _word_shape(raw: str) -> int:
    """Сколько дефисных частей в слове: Иванов = 1, Иванов-Петров = 2."""
    _, core, _ = _strip_nonletters(raw)
    if not core:
        return 1
    return max(1, len([part for part in core.split("-") if part]))


_PREP_CASES: dict[str, str] = {
    "у": "gent",
    "от": "gent",
    "для": "gent",
    "без": "gent",
    "из": "gent",
    "до": "gent",
    "около": "gent",
    "кроме": "gent",
    "после": "gent",
    "вместо": "gent",
    "против": "gent",
    "к": "datv",
    "ко": "datv",
    "по": "datv",
    "благодаря": "datv",
    "вопреки": "datv",
    "с": "ablt",
    "со": "ablt",
    "между": "ablt",
    "над": "ablt",
    "перед": "ablt",
    "о": "loct",
    "об": "loct",
    "обо": "loct",
    "при": "loct",
}


def _infer_context_hints(
    text: str,
    group: Sequence[_Entity],
    fio_spans: Sequence[tuple[int, int]],
) -> tuple[str | None, str | None]:
    """Определяет род и падеж группы ФИО по ближайшему синтаксическому контексту.

    Используется для разрешения морфологической омонимии:
      * двуполых имён («Саша подписал» vs «Саша подписала», «гражданка Женя»);
      * омонимичных фамилий без имени («у Иванова», «сотрудника Петрова» vs
        «гражданка Иванова выступила»).
    """
    if not group:
        return None, None

    start = group[0].start
    end = group[-1].end

    # Берём локальный фрагмент внутри текущей клаузы (останавливаемся на знаках препинания)
    left_slice = re.split(r"[.,!?;:\n]", text[max(0, start - 90):start])[-1]
    right_slice = re.split(r"[.,!?;:\n]", text[end:min(len(text), end + 90)])[0]

    def _non_fio_words(chunk: str, base_offset: int) -> list[str]:
        words: list[str] = []
        for match in re.finditer(r"[А-Яа-яЁё]+", chunk):
            w_start = base_offset + match.start()
            w_end = base_offset + match.end()
            if any(not (w_end <= fs or w_start >= fe) for fs, fe in fio_spans):
                continue
            words.append(match.group(0))
        return words

    left_base = start - len(left_slice)
    left_words = _non_fio_words(left_slice, left_base)[-3:]
    right_words = _non_fio_words(right_slice, end)[:3]

    gender_hint: str | None = None
    case_hint: str | None = None

    # 1. Проверяем слово непосредственно перед ФИО (предлог или определяемое существительное)
    if left_words:
        prev_word = left_words[-1].lower().replace("ё", "е")
        if prev_word in _PREP_CASES:
            case_hint = _PREP_CASES[prev_word]
        else:
            for parse in _morph_parse(prev_word)[:3]:
                if (
                    parse.tag.POS == "NOUN"
                    and "anim" in parse.tag.grammemes
                    and parse.tag.number == "sing"
                ):
                    if parse.tag.gender in ("masc", "femn") and gender_hint is None:
                        gender_hint = parse.tag.gender
                    if parse.tag.case and case_hint is None:
                        case_hint = parse.tag.case
                    break

    # 2. Ищем согласованный глагол прошедшего времени или причастие/прилагательное/титул в окне
    for word in [*reversed(left_words), *right_words]:
        if gender_hint is not None:
            break
        lower = word.lower().replace("ё", "е")
        for parse in _morph_parse(lower)[:2]:
            pos = parse.tag.POS
            if (
                pos in ("VERB", "PRTS")
                and parse.tag.tense == "past"
                and parse.tag.number == "sing"
                and parse.tag.gender in ("masc", "femn")
            ):
                gender_hint = parse.tag.gender
                break
            if (
                pos in ("ADJF", "PRTF")
                and parse.tag.number == "sing"
                and parse.tag.gender in ("masc", "femn")
            ):
                gender_hint = parse.tag.gender
                break
            if (
                pos == "NOUN"
                and "anim" in parse.tag.grammemes
                and parse.tag.number == "sing"
                and parse.tag.gender in ("masc", "femn")
            ):
                gender_hint = parse.tag.gender
                break

    return gender_hint, case_hint


def _resolve_fio(text: str, entities: Sequence[_Entity]) -> dict[_Entity, _FioPlan]:
    result: dict[_Entity, _FioPlan] = {}
    registry = _PersonRegistry()
    fio = [e for e in entities if e.attr in FIO_ATTRS]
    fio_spans = [(e.start, e.end) for e in fio]
    for group in _group_fio(text, fio):
        # Контекстное снятие омонимии нужно только для одиночного ФИО-токена
        # (например, «Саша подписал» или «от сотрудника Иванова»), когда рядом
        # нет парного имени/фамилии/отчества.
        if len(group) == 1:
            gender_hint, case_hint = _infer_context_hints(text, group, fio_spans)
        else:
            gender_hint, case_hint = None, None
        infos = [
            _info_for_entity(text, entity, gender_hint=gender_hint, case_hint=case_hint)
            for entity in group
        ]

        # Проверяем совместное пересечение пар (падеж, род):
        # для «Саша Иванова» пересечение пар даёт единственную пару ('nomn', 'femn'),
        # а для «Саши Иванова» — ('gent', 'masc').
        joint_cg = set(infos[0].case_gender_pairs) if infos else set()
        for info in infos[1:]:
            joint_cg &= set(info.case_gender_pairs)
        if gender_hint in ("masc", "femn"):
            filtered_cg = {pair for pair in joint_cg if pair[1] == gender_hint}
            if filtered_cg:
                joint_cg = filtered_cg
        if case_hint:
            filtered_cg = {pair for pair in joint_cg if pair[0] == case_hint}
            if filtered_cg:
                joint_cg = filtered_cg

        if len(joint_cg) == 1:
            chosen_case, chosen_gender = next(iter(joint_cg))
        else:
            chosen_case = _choose(infos, "case")
            chosen_gender = _choose(infos, "gender")

        gram = _Gram(
            case=chosen_case,
            gender=chosen_gender,
            number=_choose(infos, "number"),
        )
        attrs = {info.attr: info.lemma for info in infos}
        person = registry.resolve(attrs, gram.gender)
        for entity in group:
            shape = _word_shape(text[entity.start:entity.end])
            result[entity] = _FioPlan(
                gram=gram,
                base=person.base(entity.attr, shape),
            )
    return result


# ---------------------------------------------------------------------------
# Генерация ФИО
# ---------------------------------------------------------------------------

# Списки подобраны так, чтобы pymorphy3 уверенно разбирал и склонял все
# слова. Faker тоже подошёл бы, но он иногда выдаёт редкие имена и слова,
# которые pymorphy трактует как обычные существительные (например,
# «Мясник»), из-за чего падежные формы получаются неверными.

_SURNAMES_M = (
    "Иванов", "Смирнов", "Кузнецов", "Попов", "Васильев", "Петров",
    "Соколов", "Михайлов", "Новиков", "Фёдоров", "Морозов", "Волков",
    "Алексеев", "Лебедев", "Семёнов", "Егоров", "Павлов", "Козлов",
    "Степанов", "Николаев", "Орлов", "Андреев", "Макаров", "Никитин",
    "Захаров", "Зайцев", "Соловьёв", "Борисов", "Яковлев", "Григорьев",
    "Романов", "Воробьёв", "Сергеев", "Кузьмин", "Фролов", "Александров",
    "Дмитриев", "Королёв", "Гусев", "Киселёв", "Ильин", "Максимов",
    "Поляков", "Сорокин", "Виноградов", "Ковалёв", "Белов", "Медведев",
    "Антонов", "Тарасов", "Жуков", "Баранов", "Филиппов", "Комаров",
    "Давыдов", "Беляев", "Герасимов", "Богданов", "Осипов", "Сидоров",
    "Матвеев", "Титов", "Марков", "Миронов", "Крылов", "Куликов",
    "Карпов", "Власов", "Мельников", "Денисов", "Гаврилов", "Тихонов",
    "Казаков", "Афанасьев", "Данилов", "Савельев", "Тимофеев", "Фомин",
    "Чернов", "Абрамов", "Мартынов", "Ефимов", "Федотов", "Щербаков",
    "Назаров", "Калинин", "Исаев", "Чернышёв", "Быков", "Маслов",
    "Родионов", "Коновалов", "Лазарев", "Воронин", "Климов", "Филатов",
    "Пономарёв", "Голубев", "Кудрявцев", "Прохоров", "Наумов", "Потапов",
    "Журавлёв", "Овчинников", "Трофимов", "Леонов", "Соболев", "Ермаков",
    "Колесников", "Гончаров", "Емельянов", "Никифоров", "Грачёв", "Котов",
    "Гришин", "Ефремов", "Архипов", "Громов", "Малышев",
    "Панов", "Моисеев", "Румянцев", "Акимов", "Кондратьев", "Бирюков",
    "Горбунов", "Анисимов", "Ерёмин", "Тихомиров", "Галкин", "Лукьянов",
    "Михеев", "Скворцов", "Юдин", "Белоусов", "Нестеров", "Симонов",
    "Прокофьев", "Харитонов", "Князев", "Цветков", "Левин", "Митрофанов",
    "Воронов", "Аксёнов", "Мальцев", "Логинов", "Горшков",
    "Савин", "Краснов", "Майоров", "Демидов", "Елисеев", "Рыбаков",
    "Сафонов", "Платонов", "Филимонов", "Гущин", "Ширяев", "Кулагин",
    "Толстой", "Достоевский",
)

_NAMES_M = (
    "Александр", "Алексей", "Андрей", "Антон", "Аркадий", "Арсений",
    "Артём", "Богдан", "Борис", "Вадим", "Валерий",
    "Василий", "Виктор", "Виталий", "Владимир", "Владислав",
    "Всеволод", "Вячеслав", "Геннадий", "Георгий", "Герман", "Глеб",
    "Григорий", "Даниил", "Денис", "Дмитрий", "Евгений", "Егор",
    "Захар", "Иван", "Игорь", "Илья", "Кирилл", "Константин", "Лев",
    "Леонид", "Максим", "Марат", "Матвей", "Михаил", "Никита",
    "Николай", "Олег", "Павел", "Пётр", "Платон", "Роман", "Ростислав",
    "Савелий", "Семён", "Сергей", "Станислав", "Степан",
    "Тарас", "Тимур", "Трофим", "Фёдор", "Филипп", "Эдуард", "Юрий",
    "Ярослав",
)

_NAMES_F = (
    "Александра", "Алина", "Алла", "Анастасия", "Ангелина", "Анна",
    "Антонина", "Валентина", "Валерия", "Варвара", "Вера", "Вероника",
    "Виктория", "Галина", "Дарья", "Диана", "Евгения", "Екатерина",
    "Елена", "Елизавета", "Жанна", "Зинаида", "Зоя", "Инна", "Ирина",
    "Карина", "Кира", "Клавдия", "Кристина", "Ксения", "Лариса",
    "Лидия", "Любовь", "Людмила", "Маргарита", "Марина", "Мария",
    "Милана", "Надежда", "Наталья", "Нина", "Оксана", "Ольга",
    "Полина", "Раиса", "Регина", "Светлана", "Серафима", "Снежана",
    "Софья", "Тамара", "Татьяна", "Ульяна", "Юлия", "Яна",
)

_PATRONYMICS_M = (
    "Александрович", "Алексеевич", "Андреевич", "Антонович",
    "Аркадьевич", "Арсеньевич", "Артёмович", "Богданович",
    "Борисович", "Вадимович", "Валентинович", "Валерьевич",
    "Васильевич", "Викторович", "Витальевич", "Владимирович",
    "Владиславович", "Всеволодович", "Вячеславович", "Геннадьевич",
    "Георгиевич", "Германович", "Глебович", "Григорьевич",
    "Данилович", "Денисович", "Дмитриевич", "Евгеньевич",
    "Егорович", "Захарович", "Иванович", "Игоревич", "Ильич",
    "Кириллович", "Константинович", "Львович", "Леонидович",
    "Максимович", "Маркович", "Матвеевич", "Михайлович", "Никитич",
    "Николаевич", "Олегович", "Павлович", "Петрович", "Платонович",
    "Романович", "Ростиславович", "Русланович", "Савельевич",
    "Семёнович", "Сергеевич", "Станиславович", "Степанович",
    "Тарасович", "Тимурович", "Трофимович", "Фёдорович",
    "Филиппович", "Эдуардович", "Юрьевич", "Ярославович",
)

_LATIN_NAMES_M = (
    "Alexander", "Alexey", "Andrew", "Anton", "Boris", "Daniel",
    "Dmitry", "Eugene", "Ivan", "Maxim", "Michael", "Nikolay",
    "Pavel", "Roman", "Sergey", "Victor", "Vladimir", "Yuri",
)
_LATIN_NAMES_F = (
    "Alexandra", "Alina", "Anastasia", "Anna", "Daria", "Elena",
    "Elizabeth", "Irina", "Ksenia", "Maria", "Marina", "Natalia",
    "Olga", "Polina", "Svetlana", "Tatiana", "Valentina", "Vera",
)
_LATIN_SURNAMES = (
    "Ivanov", "Smirnov", "Kuznetsov", "Popov", "Vasiliev", "Petrov",
    "Sokolov", "Mikhailov", "Novikov", "Fedorov", "Morozov", "Volkov",
    "Alekseev", "Lebedev", "Semenov", "Egorov", "Pavlov", "Kozlov",
)


def _female_surname(male: str) -> str:
    if male.endswith(("ов", "ев", "ёв", "ин", "ын")):
        return male + "а"
    if male.endswith(("ский", "цкий")):
        return male[:-2] + "ая"
    if male.endswith(("ой", "ый")):
        return male[:-2] + "ая"
    if male.endswith("ко"):
        return male  # несклоняемая фамилия
    return male + "а"


def _female_patronymic(male: str) -> str:
    if male == "Ильич":
        return "Ильинична"
    if male.endswith(("ович", "евич")):
        return male[:-2] + "на"
    if male.endswith("ич"):
        return male[:-2] + "ична"
    return male + "на"


def _simple_lemma(word: str) -> str:
    parses = _morph_parse(word)
    if parses:
        return parses[0].normal_form.lower().replace("ё", "е")
    return word.lower().replace("ё", "е")


def _base_conflicts(candidate: str, forbidden: str) -> bool:
    """Проверяет, не совпадает ли замена с исходным словом по лемме."""
    if not forbidden:
        return False
    forbidden_parts = forbidden.split("-")
    candidate_parts = candidate.split("-")
    if len(forbidden_parts) == len(candidate_parts) and len(candidate_parts) > 1:
        return all(
            _simple_lemma(candidate_part) == forbidden_part
            for candidate_part, forbidden_part in zip(candidate_parts, forbidden_parts)
        )
    if len(forbidden) == 1:
        # Инициал: сравниваем только первую букву.
        return _simple_lemma(candidate)[:1] == forbidden
    return _simple_lemma(candidate) == forbidden


def _replacement_base(attr: str, gender: str, forbidden: str = "") -> str:
    """Случайная основа, по возможности не совпадающая с исходной леммой."""
    candidate = _pick_base(attr, gender, "sing")
    for _ in range(80):
        if not _base_conflicts(candidate, forbidden):
            return candidate
        candidate = _pick_base(attr, gender, "sing")
    return candidate


def _pick_base(attr: str, gender: str, number: str) -> str:
    female = gender == "femn"
    if attr == "SURNAME":
        male = _RNG.choice(_SURNAMES_M)
        return _female_surname(male) if female else male
    if attr == "PATRONYMIC":
        male = _RNG.choice(_PATRONYMICS_M)
        return _female_patronymic(male) if female else male
    return _RNG.choice(_NAMES_F if female else _NAMES_M)


def _inflect(base: str, gram: _Gram, attr: str) -> str:
    if _MORPH is None:
        return base

    kind = _KIND.get(attr)
    parses = _morph_parse(base)
    if not parses:
        return base

    candidates = list(parses)
    if kind:
        tagged = [p for p in candidates if kind in p.tag.grammemes]
        if tagged:
            candidates = tagged
    nouns = [p for p in candidates if p.tag.POS == "NOUN"]
    if nouns:
        candidates = nouns

    if gram.number == "sing":
        same_gender = [p for p in candidates if p.tag.gender == gram.gender]
        if same_gender:
            candidates = same_gender

    grams = {gram.case, gram.number}
    if gram.number == "sing" and gram.gender in ("masc", "femn"):
        grams.add(gram.gender)

    for parse in candidates:
        try:
            form = parse.inflect(grams)
        except Exception:  # pragma: no cover
            form = None
        if form is not None:
            return form.word

    # Мягкий откат: без рода (иногда мешает для несклоняемых слов)
    for parse in candidates:
        try:
            form = parse.inflect({gram.case, gram.number})
        except Exception:  # pragma: no cover
            form = None
        if form is not None:
            return form.word

    return base


def _match_case(original: str, generated: str) -> str:
    letters = [char for char in original if char.isalpha()]
    if not letters:
        return generated
    if original.isupper():
        return generated.upper()
    if letters[0].isupper():
        return generated[:1].upper() + generated[1:]
    return generated.lower()


def _strip_nonletters(token: str) -> tuple[str, str, str]:
    match = re.search(r"[A-Za-zА-Яа-яЁё]", token)
    if not match:
        return token, "", ""
    start = match.start()
    end = max(i for i, char in enumerate(token) if char.isalpha()) + 1
    return token[:start], token[start:end], token[end:]


def _random_letters(length: int, *, cyrillic: bool) -> str:
    alphabet = "абвгдежзиклмнопрстуфхцчшщэюя" if cyrillic else "abcdefghijklmnopqrstuvwxyz"
    return "".join(_RNG.choice(alphabet) for _ in range(max(1, length)))


def _gen_person_core(
    core: str,
    attr: str,
    gram: _Gram,
    base: str | None = None,
) -> str:
    # Инициалы: «И.» -> «А.», сохраняем длину и регистр. Если замена уже
    # известна, берём первую букву из неё — тогда «И.» и «Ивану» одного
    # человека дадут согласованные «А.» и «Александру».
    if len(core) == 1:
        if base:
            letter = base[:1].lower()
            if not re.fullmatch(r"[а-яё]", letter):
                letter = _RNG.choice("абвгдежзиклмнопрстуфхцчшщэюя")
        else:
            letter = _RNG.choice("абвгдежзиклмнопрстуфхцчшщэюя")
        return _match_case(core, letter)

    provided_base = base is not None
    if base is None:
        base = _pick_base(attr, gram.gender, gram.number)
    form = _inflect(base, gram, attr)

    # Совсем неизвестное слово (аббревиатура, опечатка) без готовой
    # замены — сохраняем длину и алфавит, чтобы не сломать структуру.
    if not provided_base and not _morph_parse(core):
        cyrillic = bool(re.search(r"[А-Яа-яЁё]", core))
        return _match_case(core, _random_letters(len(core), cyrillic=cyrillic))

    return _match_case(core, form)


def _gen_person_token(
    token: str,
    attr: str,
    gram: _Gram,
    base: str | None = None,
) -> str:
    prefix, core, suffix = _strip_nonletters(token)
    if not core:
        return token
    for _ in range(6):
        generated = _gen_person_core(core, attr, gram, base)
        if generated.lower() != core.lower():
            return prefix + generated + suffix
    return prefix + generated + suffix


def _gen_latin_token(token: str) -> str:
    """Запасной вариант для ФИО, записанных латиницей."""
    prefix, core, suffix = _strip_nonletters(token)
    if not core:
        return token
    if len(core) == 1:
        return prefix + _match_case(core, _RNG.choice(string.ascii_uppercase)) + suffix
    pool = _LATIN_SURNAMES + _LATIN_NAMES_M + _LATIN_NAMES_F
    for _ in range(8):
        candidate = _match_case(core, _RNG.choice(pool))
        if candidate.lower() != core.lower():
            return prefix + candidate + suffix
    return prefix + _match_case(core, _RNG.choice(pool)) + suffix


def _anonymize_fio(
    raw: str,
    attr: str,
    gram: _Gram,
    base: str | None = None,
) -> str:
    # Для дефисных форм («Иванов-Петров») основа тоже дефисная:
    # «Смирнов-Кузнецов». Каждая часть склоняется отдельно.
    base_parts = None
    if base and "-" in base and "-" in raw:
        base_parts = base.split("-")

    parts = re.split(r"([\s\-]+)", raw)
    result: list[str] = []
    part_index = 0
    for part in parts:
        if not part or re.fullmatch(r"[\s\-]+", part):
            result.append(part)
            continue
        if re.search(r"[А-Яа-яЁё]", part):
            if base_parts and part_index < len(base_parts):
                part_base = base_parts[part_index]
            else:
                part_base = base
            result.append(_gen_person_token(part, attr, gram, part_base))
            part_index += 1
        elif re.search(r"[A-Za-z]", part):
            result.append(_gen_latin_token(part))
        else:
            result.append(part)
    return "".join(result)


# ---------------------------------------------------------------------------
# Контрольные разряды
# ---------------------------------------------------------------------------

_INN_COEFFS_10 = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_COEFFS_11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_COEFFS_12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)


def _inn_check_digit(digits: Sequence[int], coeffs: Sequence[int]) -> int:
    return sum(digit * coeff for digit, coeff in zip(digits, coeffs)) % 11 % 10


def validate_inn(value: str) -> bool:
    """Проверяет контрольные разряды ИНН (10 или 12 цифр)."""
    digits = [int(char) for char in value if char.isdigit()]
    if len(digits) == 10:
        return digits[-1] == _inn_check_digit(digits[:9], _INN_COEFFS_10)
    if len(digits) == 12:
        if digits[-2] != _inn_check_digit(digits[:10], _INN_COEFFS_11):
            return False
        return digits[-1] == _inn_check_digit(digits[:11], _INN_COEFFS_12)
    return False


def _luhn_sum(digits: Sequence[int]) -> int:
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total


def validate_luhn(value: str) -> bool:
    """Проверяет номер банковской карты по алгоритму Луна."""
    digits = [int(char) for char in value if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    return _luhn_sum(digits) % 10 == 0


def _snils_check(digits: Sequence[int]) -> int:
    total = sum(digit * (9 - index) for index, digit in enumerate(digits[:9]))
    if total < 100:
        return total
    if total in (100, 101):
        return 0
    check = total % 101
    return 0 if check == 100 else check


def validate_snils(value: str) -> bool:
    """Проверяет контрольное число СНИЛС."""
    digits = [int(char) for char in value if char.isdigit()]
    if len(digits) != 11:
        return False
    return digits[9] * 10 + digits[10] == _snils_check(digits)


def _ogrnip_check_digit(digits_14: Sequence[str]) -> str:
    number = int("".join(digits_14))
    return str((number % 13) % 10)


def validate_ogrnip(value: str) -> bool:
    """Проверяет 15-значный номер ОГРНИП и его контрольный разряд (модуль 13)."""
    digits = [char for char in value if char.isdigit()]
    if len(digits) != 15 or digits[0] not in ("3", "4"):
        return False
    return digits[14] == _ogrnip_check_digit(digits[:14])


def validate_passport(value: str) -> bool:
    """Проверяет 10-значный паспорт РФ (4 цифры серии с регионом 01..99 и 6 цифр номера)."""
    digits = "".join(char for char in value if char.isdigit())
    if len(digits) != 10:
        return False
    region = int(digits[:2])
    number = int(digits[4:])
    return 1 <= region <= 99 and 1 <= number <= 999999


def validate_birth_cert(value: str) -> bool:
    """Проверяет серию (римские цифры + 2 кириллические буквы) и 6-значный номер свидетельства о рождении."""
    pattern = re.compile(
        r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|[IVXLCDM]{1,6})"
        r"[\-–—][А-ЯЁа-яё]{2}(?:[\s№#\-]+)\d{6}$"
    )
    if not pattern.match(value.strip()):
        return False
    digits = "".join(char for char in value if char.isdigit())
    return len(digits) == 6 and int(digits) > 0


def validate_kladr(value: str) -> bool:
    """Проверяет 13- или 17-значный код КЛАДР (код региона 01..99 и признак актуальности 00..51)."""
    digits = "".join(char for char in value if char.isdigit())
    if len(digits) not in (13, 17):
        return False
    region = int(digits[:2])
    actual = int(digits[-2:])
    return 1 <= region <= 99 and 0 <= actual <= 51


def validate_ip(value: str) -> bool:
    """Проверяет корректность IPv4 или IPv6 адреса."""
    try:
        ipaddress.ip_address(value.strip())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Генерация числовых атрибутов
# ---------------------------------------------------------------------------

def _rand_digit() -> str:
    return _RNG.choice(string.digits)


def _substitute_digits(raw: str, new_digits: Sequence[str]) -> str:
    """Подставляет новые цифры в исходную строку, сохраняя все разделители."""
    result: list[str] = []
    index = 0
    for char in raw:
        if char.isdigit():
            if index < len(new_digits):
                result.append(new_digits[index])
                index += 1
            else:
                result.append(char)
        else:
            result.append(char)
    return "".join(result)


def _digits_of(raw: str) -> list[str]:
    return [char for char in raw if char.isdigit()]


def _anonymize_inn(raw: str) -> str:
    digits = _digits_of(raw)
    length = len(digits)

    if length < 4:
        new_digits = [_rand_digit() for _ in range(length)]
    elif length == 10:
        body = digits[:4] + [_rand_digit() for _ in range(5)]
        new_digits = body + [str(_inn_check_digit([int(d) for d in body], _INN_COEFFS_10))]
    elif length == 12:
        body = digits[:4] + [_rand_digit() for _ in range(6)]  # 10 цифр
        check11 = str(_inn_check_digit([int(d) for d in body], _INN_COEFFS_11))
        body11 = body + [check11]
        check12 = str(_inn_check_digit([int(d) for d in body11], _INN_COEFFS_12))
        new_digits = body11 + [check12]
    else:
        # Нестандартная длина: сохраняем первые 4 цифры и обезличиваем остальные.
        new_digits = digits[:4] + [_rand_digit() for _ in range(length - 4)]

    if new_digits == digits and length > 4:
        return _anonymize_inn(raw) if length in (10, 12) else raw
    return _substitute_digits(raw, new_digits)


def _luhn_fix(digits: list[str], middle: Sequence[int]) -> bool:
    """Подбирает одну цифру в середине так, чтобы номер стал валиден по Луну."""
    numeric = [int(d) for d in digits]
    for index in middle:
        original = numeric[index]
        for candidate in range(10):
            numeric[index] = candidate
            if _luhn_sum(numeric) % 10 == 0:
                digits[index] = str(candidate)
                return True
        numeric[index] = original
    return False


def _anonymize_card(raw: str) -> str:
    digits = _digits_of(raw)
    length = len(digits)

    if length < 10:
        # Невозможно одновременно сохранить 6+4 цифры — просто обезличиваем.
        new_digits = [_rand_digit() for _ in range(length)]
        return _substitute_digits(raw, new_digits)

    middle = list(range(6, length - 4))
    if not middle:
        # 10 цифр: первые 6 и последние 4 занимают всё число, менять нечего.
        return raw

    for _ in range(8):
        new_digits = list(digits)
        for index in middle:
            new_digits[index] = _rand_digit()
        if not _luhn_fix(new_digits, middle):
            break
        if new_digits != digits:
            return _substitute_digits(raw, new_digits)

    # Крайне маловероятный откат: меняем один средний разряд вручную.
    new_digits = list(digits)
    for index in middle:
        for candidate in range(10):
            new_digits[index] = str(candidate)
            if _luhn_sum([int(d) for d in new_digits]) % 10 == 0:
                return _substitute_digits(raw, new_digits)
    return raw


def _anonymize_phone(raw: str) -> str:
    digits = _digits_of(raw)
    if len(digits) <= 4:
        # Меньше 5 цифр — сохранять и обезличивать нечего.
        return raw

    for _ in range(8):
        new_digits = digits[:4] + [_rand_digit() for _ in range(len(digits) - 4)]
        if new_digits != digits:
            return _substitute_digits(raw, new_digits)
    return _substitute_digits(raw, digits[:4] + [_rand_digit() for _ in range(len(digits) - 4)])


def _anonymize_snils(raw: str) -> str:
    digits = _digits_of(raw)
    if len(digits) != 11:
        return _substitute_digits(raw, [_rand_digit() for _ in range(len(digits))])

    for _ in range(8):
        base = [_rand_digit() for _ in range(9)]
        check = _snils_check([int(d) for d in base])
        new_digits = base + [str(check // 10), str(check % 10)]
        if new_digits != digits:
            return _substitute_digits(raw, new_digits)
    return _substitute_digits(raw, new_digits)


def _anonymize_passport(raw: str) -> str:
    """Сохраняет 4-значную серию паспорта и разделители, заменяя 6-значный номер."""
    digits = _digits_of(raw)
    if len(digits) <= 4:
        return raw

    series = digits[:4]
    num_len = len(digits) - 4
    for _ in range(10):
        first = _RNG.choice("123456789")
        rest = [_rand_digit() for _ in range(num_len - 1)]
        new_digits = series + [first] + rest
        if new_digits != digits:
            return _substitute_digits(raw, new_digits)
    return _substitute_digits(raw, new_digits)


def _anonymize_ogrnip(raw: str) -> str:
    """Сохраняет первые 5 цифр ОГРНИП (признак 3, год, регион) и пересчитывает 15-й контрольный разряд."""
    digits = _digits_of(raw)
    if len(digits) != 15:
        if len(digits) <= 5:
            return raw
        return _substitute_digits(
            raw, digits[:5] + [_rand_digit() for _ in range(len(digits) - 5)]
        )

    prefix = digits[:5]
    for _ in range(10):
        body = prefix + [_rand_digit() for _ in range(9)]
        if all(d == "0" for d in body[5:]):
            body[-1] = "1"
        check = _ogrnip_check_digit(body)
        new_digits = body + [check]
        if new_digits != digits:
            return _substitute_digits(raw, new_digits)
    return _substitute_digits(raw, new_digits)


def _anonymize_birth_cert(raw: str) -> str:
    """Сохраняет серию свидетельства о рождении (римские цифры и буквы), заменяя 6-значный номер."""
    digits = _digits_of(raw)
    if not digits:
        return raw
    for _ in range(10):
        first = _RNG.choice("123456789")
        rest = [_rand_digit() for _ in range(len(digits) - 1)]
        new_digits = [first] + rest
        if new_digits != digits:
            return _substitute_digits(raw, new_digits)
    return _substitute_digits(raw, new_digits)


def _anonymize_kladr(raw: str) -> str:
    """Сохраняет код субъекта РФ (первые 2 цифры) и признак актуальности (последние 2 цифры)."""
    digits = _digits_of(raw)
    length = len(digits)
    if length not in (13, 17):
        if length <= 4:
            return raw
        mid = [_rand_digit() for _ in range(length - 4)]
        return _substitute_digits(raw, digits[:2] + mid + digits[-2:])

    region = digits[:2]
    actual = digits[-2:]
    middle_len = length - 4
    orig_middle = digits[2:-2]

    for _ in range(10):
        # Сохраняем нулевые/ненулевые блоки структуры КЛАДР (РРР ГГГ ППП [УУУУ]),
        # но гарантированно изменяем хотя бы один блок.
        seg_lengths = (3, 3, 3) if length == 13 else (3, 3, 3, 4)
        new_mid: list[str] = []
        pos = 0
        for seg_len in seg_lengths:
            seg = orig_middle[pos:pos + seg_len]
            pos += seg_len
            if all(ch == "0" for ch in seg):
                new_mid.extend(["0"] * seg_len)
            else:
                val = _RNG.randint(1, 99)
                new_mid.extend(list(f"{val:0{seg_len}d}"))
        if all(ch == "0" for ch in new_mid):
            val = _RNG.randint(1, 99)
            new_mid[3:6] = list(f"{val:03d}")
        if new_mid != orig_middle:
            return _substitute_digits(raw, region + new_mid + actual)

    fallback_mid = list(orig_middle)
    fallback_mid[-1] = "1" if fallback_mid[-1] != "1" else "2"
    return _substitute_digits(raw, region + fallback_mid + actual)


def _anonymize_ip(raw: str) -> str:
    """Обезличивает IPv4/IPv6 с сохранением семейства протокола и класса подсети."""
    stripped = raw.strip()
    try:
        addr = ipaddress.ip_address(stripped)
    except ValueError:
        return _substitute_digits(raw, [_rand_digit() for _ in _digits_of(raw)])

    if isinstance(addr, ipaddress.IPv4Address):
        octets = [int(part) for part in stripped.split(".")]
        for _ in range(10):
            if octets[0] == 10:
                cand = [10, _RNG.randint(0, 255), _RNG.randint(0, 255), _RNG.randint(1, 254)]
            elif octets[0] == 172 and 16 <= octets[1] <= 31:
                cand = [172, octets[1], _RNG.randint(0, 255), _RNG.randint(1, 254)]
            elif octets[0] == 192 and octets[1] == 168:
                cand = [192, 168, _RNG.randint(0, 255), _RNG.randint(1, 254)]
            elif octets[0] == 127:
                cand = [127, 0, 0, _RNG.randint(2, 254)]
            else:
                cand = [octets[0], _RNG.randint(1, 254), _RNG.randint(1, 254), _RNG.randint(1, 254)]
            if cand != octets:
                return ".".join(str(x) for x in cand)
        octets[-1] = 2 if octets[-1] != 2 else 3
        return ".".join(str(x) for x in octets)

    # IPv6: сохраняем первые 4 хекстета (64-битный сетевой префикс)
    exploded = addr.exploded.split(":")
    upper = any(ch.isupper() for ch in stripped if ch.isalpha())
    for _ in range(10):
        tail = [f"{_RNG.randint(1, 0xFFFE):04x}" for _ in range(4)]
        cand_addr = ipaddress.IPv6Address(":".join(exploded[:4] + tail))
        formatted = cand_addr.compressed
        if upper:
            formatted = formatted.upper()
        if formatted.lower() != stripped.lower():
            return formatted
    return stripped


# ---------------------------------------------------------------------------
# E-mail
# ---------------------------------------------------------------------------

def _random_local_part(local: str) -> str:
    result: list[str] = []
    for char in local:
        if char.isalpha():
            alphabet = string.ascii_uppercase if char.isupper() else string.ascii_lowercase
            result.append(_RNG.choice(alphabet))
        elif char.isdigit():
            result.append(_rand_digit())
        else:
            result.append(char)
    return "".join(result)


def _anonymize_email(raw: str) -> str:
    if "@" not in raw:
        # Не e-mail, но обезличить всё равно нужно: сохраняем пунктуацию.
        for _ in range(8):
            candidate = _random_local_part(raw)
            if candidate != raw:
                return candidate
        return _random_local_part(raw)

    local, _, domain = raw.rpartition("@")
    if not local:
        return raw

    for _ in range(8):
        candidate = _random_local_part(local)
        if candidate != local:
            return candidate + "@" + domain
    return _random_local_part(local) + "@" + domain


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def _anonymize_value(
    raw: str,
    entity: _Entity,
    plan: _FioPlan | None,
) -> str:
    if entity.attr in FIO_ATTRS:
        if plan is None:
            gram = _Gram("nomn", "masc", "sing")
            plan = _FioPlan(
                gram=gram,
                base=_pick_base(entity.attr, gram.gender, gram.number),
            )
        return _anonymize_fio(raw, entity.attr, plan.gram, plan.base)
    if entity.attr == "EMAIL":
        return _anonymize_email(raw)
    if entity.attr == "INN":
        return _anonymize_inn(raw)
    if entity.attr == "PHONE":
        return _anonymize_phone(raw)
    if entity.attr == "BANK_CARD":
        return _anonymize_card(raw)
    if entity.attr == "SNILS":
        return _anonymize_snils(raw)
    if entity.attr == "PASSPORT":
        return _anonymize_passport(raw)
    if entity.attr == "OGRNIP":
        return _anonymize_ogrnip(raw)
    if entity.attr == "BIRTH_CERT":
        return _anonymize_birth_cert(raw)
    if entity.attr == "KLADR":
        return _anonymize_kladr(raw)
    if entity.attr == "IP":
        return _anonymize_ip(raw)
    return raw


_AUTO = object()


def _auto_extract_pii(text: str) -> list[list[object]]:
    """Вызывает распознаватель extract_pii из модуля recognizer."""
    root_dir = Path(__file__).resolve().parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from recognizer import extract_pii

    return extract_pii(text)


def _anon_text_with_attributes(text: str, attributes: object) -> str:
    if not isinstance(text, str):
        raise TypeError("text должен быть строкой")

    entities = _select_non_overlapping(_coerce_entities(text, attributes))
    if not entities:
        return text

    fio_plans = _resolve_fio(text, entities)
    # Повторяющиеся не-ФИО значения (один и тот же телефон, e-mail и т.п.)
    # тоже должны заменяться одинаково в пределах одного текста.
    exact_cache: dict[tuple[str, str], str] = {}

    pieces: list[str] = []
    cursor = 0
    for entity in entities:
        pieces.append(text[cursor:entity.start])
        raw = text[entity.start:entity.end]
        if entity.attr in FIO_ATTRS:
            replacement = _anonymize_value(raw, entity, fio_plans.get(entity))
        else:
            key = (entity.attr, raw)
            replacement = exact_cache.get(key)
            if replacement is None:
                replacement = _anonymize_value(raw, entity, None)
                exact_cache[key] = replacement
        pieces.append(replacement)
        cursor = entity.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def anon(text: str, attributes: object = _AUTO):
    """Обезличивает персональные данные в тексте.

    Поддерживает два режима вызова:
      1) ``anon(text, attributes) -> str`` (режим ЛР №2: обезличивает переданные спаны);
      2) ``anon(text) -> (anonymized_text, attributes)`` (режим КР: сам распознаёт ПДн
         моделью и правилами ЛР №1 и возвращает кортеж из обезличенного текста и списка
         найденных атрибутов ``[["NAME", 0, 4], ...]``).
    """
    if not isinstance(text, str):
        raise TypeError("text должен быть строкой")
    if attributes is _AUTO:
        detected = _auto_extract_pii(text)
        return _anon_text_with_attributes(text, detected), detected
    return _anon_text_with_attributes(text, attributes)


def mask_pii(
    text: str,
    attributes: object = _AUTO,
    *,
    full_text: bool = False,
    mask_char: str = "*",
) -> str:
    """Маскирует персональные данные (или весь текст) символом ``*``.

    Используется как базовый метод (baseline) при сравнении сохранения смысла
    через BERTScore в КМ-3.
    """
    if not isinstance(text, str):
        raise TypeError("text должен быть строкой")
    if full_text:
        return mask_char * len(text)
    if attributes is _AUTO:
        attributes = _auto_extract_pii(text)
    entities = _select_non_overlapping(_coerce_entities(text, attributes))
    if not entities:
        return text
    pieces: list[str] = []
    cursor = 0
    for entity in entities:
        pieces.append(text[cursor:entity.start])
        pieces.append(mask_char * (entity.end - entity.start))
        cursor = entity.end
    pieces.append(text[cursor:])
    return "".join(pieces)


# Синоним на случай, если удобнее называть функцию полным словом.
anonymize = anon


if __name__ == "__main__":
    sample = (
        "Иванов Иван Иванович, ИНН 7707083893, телефон +7 (999) 123-45-67, "
        "e-mail ivanov@example.com, карта 4084520359695002."
    )

    def span(attr: str, value: str) -> list[object]:
        start = sample.find(value)
        return [attr, start, start + len(value)]

    found = [
        span("SURNAME", "Иванов"),
        span("NAME", "Иван"),
        span("PATRONYMIC", "Иванович"),
        span("INN", "7707083893"),
        span("PHONE", "+7 (999) 123-45-67"),
        span("EMAIL", "ivanov@example.com"),
        span("BANK_CARD", "4084520359695002"),
    ]
    print("До:   ", sample)
    print("После:", anon(sample, found))
