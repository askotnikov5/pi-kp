"""Модульные тесты системы распознавания и синонимического обезличивания ПДн (Вариант 24).

Запуск:
    python -m unittest -v test_project
"""

import argparse
import re
import subprocess
import sys
import unittest
from pathlib import Path

import pymorphy3

import anon as A
import cli


MORPH = pymorphy3.MorphAnalyzer()
CYRILLIC_WORD = re.compile(r"[А-ЯЁ][а-яё]+")


def has_gram(word: str, kind: str, *, gender: str, case: str, number: str = "sing") -> bool:
    """Есть ли у слова разбор с нужными граммемами."""
    for parse in MORPH.parse(word):
        tag = parse.tag
        if kind not in tag.grammemes:
            continue
        if tag.case != case or tag.number != number:
            continue
        if number == "plur" or tag.gender == gender:
            return True
    return False


def digits(value: str) -> str:
    return "".join(char for char in value if char.isdigit())


def non_digits(value: str) -> str:
    return "".join(char for char in value if not char.isdigit())


def lemma_of(word: str, kind: str) -> str:
    for parse in MORPH.parse(word):
        if kind in parse.tag.grammemes:
            return parse.normal_form
    return MORPH.parse(word)[0].normal_form


def token_spans(text: str, pairs: list[tuple[str, str]]) -> list[list[object]]:
    """Строит спаны по значениям, не путая «Иван» с частью «Иванов»."""
    spans: list[list[object]] = []
    for attr, value in pairs:
        pattern = re.compile(
            r"(?<![A-Za-zА-Яа-яЁё0-9])"
            + re.escape(value)
            + r"(?![A-Za-zА-Яа-яЁё0-9])"
        )
        match = pattern.search(text)
        if match is None:
            raise AssertionError(f"не найдено {value!r} в {text!r}")
        spans.append([attr, match.start(), match.end()])
    return spans


def sequential_token_spans(
    text: str, pairs: list[tuple[str, str]]
) -> list[list[object]]:
    """Строит спаны подряд, чтобы различать повторяющиеся «И.»."""
    spans: list[list[object]] = []
    position = 0
    for attr, value in pairs:
        start = text.index(value, position)
        spans.append([attr, start, start + len(value)])
        position = start + len(value)
    return spans


class TestContract(unittest.TestCase):
    def test_example_from_task(self):
        text = "Иван получил кредит"
        result = A.anon(text, [["NAME", 0, 4]])
        self.assertNotEqual(result, text)
        self.assertTrue(result.endswith(" получил кредит"))
        first = result.split()[0]
        self.assertRegex(first, CYRILLIC_WORD)

    def test_several_entities_and_surrounding_text(self):
        text = "Клиент Иванов, телефон +7 (999) 123-45-67, почта ivanov@example.com."
        result = A.anon(
            text,
            [
                ["SURNAME", 7, 13],
                ["PHONE", 23, 41],
                ["EMAIL", 49, 67],
            ],
        )
        self.assertTrue(result.startswith("Клиент "))
        self.assertTrue(result.endswith("."))
        self.assertIn(", телефон ", result)
        self.assertIn(", почта ", result)

    def test_empty_attributes_returns_text(self):
        text = "Иван получил кредит"
        self.assertEqual(A.anon(text, []), text)
        self.assertEqual(A.anon(text, None), text)

    def test_text_must_be_string(self):
        with self.assertRaises(TypeError):
            A.anon(123, [["NAME", 0, 1]])

    def test_single_entity_without_outer_list(self):
        text = "Иван получил кредит"
        self.assertNotEqual(A.anon(text, ["NAME", 0, 4]), text)

    def test_anonymize_alias(self):
        self.assertIs(A.anon, A.anonymize)


class TestFioMorphology(unittest.TestCase):
    def _assert_fio(self, result, gender, case, number="sing"):
        surname, name, patronymic = result.split()
        self.assertTrue(
            has_gram(surname, "Surn", gender=gender, case=case, number=number),
            msg=f"фамилия {surname!r}: нет Surn/{gender}/{case}/{number}",
        )
        self.assertTrue(
            has_gram(name, "Name", gender=gender, case=case, number=number),
            msg=f"имя {name!r}: нет Name/{gender}/{case}/{number}",
        )
        self.assertTrue(
            has_gram(patronymic, "Patr", gender=gender, case=case, number=number),
            msg=f"отчество {patronymic!r}: нет Patr/{gender}/{case}/{number}",
        )

    def test_male_nominative(self):
        text = "Иванов Иван Иванович"
        spans = [["SURNAME", 0, 6], ["NAME", 7, 11], ["PATRONYMIC", 12, 20]]
        self._assert_fio(A.anon(text, spans), "masc", "nomn")

    def test_female_nominative(self):
        text = "Иванова Мария Ивановна"
        spans = [["SURNAME", 0, 7], ["NAME", 8, 13], ["PATRONYMIC", 14, 22]]
        self._assert_fio(A.anon(text, spans), "femn", "nomn")

    def test_male_dative(self):
        text = "Иванову Ивану Ивановичу"
        spans = [["SURNAME", 0, 7], ["NAME", 8, 13], ["PATRONYMIC", 14, 23]]
        self._assert_fio(A.anon(text, spans), "masc", "datv")

    def test_female_genitive(self):
        text = "Ивановой Марии Ивановны"
        spans = [["SURNAME", 0, 8], ["NAME", 9, 14], ["PATRONYMIC", 15, 23]]
        self._assert_fio(A.anon(text, spans), "femn", "gent")

    def test_male_genitive_with_name_disambiguates(self):
        # «Иванова» сама по себе неоднозначна: мужской родительный или
        # женский именительный. Имя «Ивана» снимает неоднозначность.
        text = "у Иванова Ивана"
        result = A.anon(text, [["SURNAME", 2, 9], ["NAME", 10, 15]])
        surname, name = result.split()[1:]
        self.assertTrue(has_gram(surname, "Surn", gender="masc", case="gent"))
        self.assertTrue(has_gram(name, "Name", gender="masc", case="gent"))

    def test_same_person_same_replacement_in_different_cases(self):
        # Пример с защиты: Алексей Долматов -> Вася Токарев,
        # Алексею Долматову -> Васе Токареву (та же замена в другом падеже).
        text = "Алексей Долматов дал в рот Алексею Долматову"
        spans = token_spans(
            text,
            [
                ("NAME", "Алексей"),
                ("SURNAME", "Долматов"),
                ("NAME", "Алексею"),
                ("SURNAME", "Долматову"),
            ],
        )
        words = A.anon(text, spans).split()
        # name surname дал в рот name surname
        self.assertEqual(lemma_of(words[0], "Name"), lemma_of(words[5], "Name"))
        self.assertEqual(lemma_of(words[1], "Surn"), lemma_of(words[6], "Surn"))
        self.assertTrue(has_gram(words[0], "Name", gender="masc", case="nomn"))
        self.assertTrue(has_gram(words[5], "Name", gender="masc", case="datv"))
        self.assertTrue(has_gram(words[1], "Surn", gender="masc", case="nomn"))
        self.assertTrue(has_gram(words[6], "Surn", gender="masc", case="datv"))

    def test_same_full_name_same_replacement(self):
        text = "Иванов Иван Иванович дал Иванову Ивану Ивановичу"
        spans = token_spans(
            text,
            [
                ("SURNAME", "Иванов"),
                ("NAME", "Иван"),
                ("PATRONYMIC", "Иванович"),
                ("SURNAME", "Иванову"),
                ("NAME", "Ивану"),
                ("PATRONYMIC", "Ивановичу"),
            ],
        )
        words = A.anon(text, spans).split()
        # surname name patronymic дал surname name patronymic
        self.assertEqual(lemma_of(words[0], "Surn"), lemma_of(words[4], "Surn"))
        self.assertEqual(lemma_of(words[1], "Name"), lemma_of(words[5], "Name"))
        self.assertEqual(lemma_of(words[2], "Patr"), lemma_of(words[6], "Patr"))

    def test_same_surname_same_replacement(self):
        text = "Долматов дал Долматову"
        spans = token_spans(
            text,
            [("SURNAME", "Долматов"), ("SURNAME", "Долматову")],
        )
        words = A.anon(text, spans).split()
        self.assertEqual(lemma_of(words[0], "Surn"), lemma_of(words[2], "Surn"))
        self.assertTrue(has_gram(words[0], "Surn", gender="masc", case="nomn"))
        self.assertTrue(has_gram(words[2], "Surn", gender="masc", case="datv"))

    def test_full_name_and_initial_are_consistent(self):
        text = "Иван Петров и И. Петров"
        spans = sequential_token_spans(
            text,
            [
                ("NAME", "Иван"),
                ("SURNAME", "Петров"),
                ("NAME", "И."),
                ("SURNAME", "Петров"),
            ],
        )
        words = A.anon(text, spans).split()
        # name surname и initial surname
        self.assertEqual(words[0][:1].lower(), words[3][:1].lower())
        self.assertEqual(lemma_of(words[1], "Surn"), lemma_of(words[4], "Surn"))
        self.assertRegex(words[3], r"^[А-Яа-яЁё]\.$")

    def test_plural_surname(self):
        result = A.anon("Ивановы", [["SURNAME", 0, 7]])
        self.assertTrue(has_gram(result, "Surn", gender="masc", case="nomn", number="plur"))

    def test_case_is_preserved(self):
        upper = A.anon("ИВАНОВ", [["SURNAME", 0, 6]])
        lower = A.anon("иванов", [["SURNAME", 0, 6]])
        self.assertEqual(upper, upper.upper())
        self.assertEqual(lower, lower.lower())

    def test_trailing_punctuation_in_span(self):
        # Распознаватель ЛР №1 иногда включает запятую в спан.
        result = A.anon("Ивановна,", [["PATRONYMIC", 0, 9]])
        self.assertTrue(result.endswith(","))
        self.assertTrue(
            has_gram(result[:-1], "Patr", gender="femn", case="nomn")
        )
        self.assertNotEqual(result, "Ивановна,")

    def test_hyphenated_surname_structure(self):
        result = A.anon("Иванов-Петров", [["SURNAME", 0, 13]])
        self.assertEqual(result.count("-"), 1)
        left, right = result.split("-")
        self.assertRegex(left, CYRILLIC_WORD)
        self.assertRegex(right, CYRILLIC_WORD)
        self.assertNotEqual(result, "Иванов-Петров")

    def test_initial_structure(self):
        result = A.anon("И. Иванов", [["NAME", 0, 2], ["SURNAME", 3, 9]])
        self.assertEqual(result[1], ".")
        self.assertNotEqual(result[0], "И")

    def test_latin_fallback(self):
        result = A.anon("Ivanov", [["SURNAME", 0, 6]])
        self.assertRegex(result, r"^[A-Za-z]+$")
        self.assertNotEqual(result, "Ivanov")

    def test_fio_is_random(self):
        values = {
            A.anon("Иван получил кредит", [["NAME", 0, 4]])
            for _ in range(25)
        }
        self.assertGreater(len(values), 1)


class TestEmail(unittest.TestCase):
    def test_domain_is_preserved(self):
        text = "Почта first.last+tag@mail.example.ru"
        result = A.anon(text, [["EMAIL", 6, 29]])
        self.assertTrue(result.endswith("@mail.example.ru"))
        local = result[6:].split("@")[0]
        self.assertNotEqual(local, "first.last+tag")
        self.assertEqual(len(local), len("first.last+tag"))
        self.assertEqual(local[5], ".")
        self.assertEqual(local[10], "+")

    def test_domain_case_is_preserved(self):
        result = A.anon("Ivanov@Example.COM", [["EMAIL", 0, 17]])
        self.assertTrue(result.endswith("@Example.COM"))

    def test_email_is_random(self):
        values = {A.anon("ivanov@example.com", [["EMAIL", 0, 18]]) for _ in range(25)}
        self.assertGreater(len(values), 1)

    def test_not_an_email_does_not_crash(self):
        result = A.anon("простотекст", [["EMAIL", 0, 11]])
        self.assertEqual(len(result), len("простотекст"))
        self.assertNotEqual(result, "простотекст")


class TestPhone(unittest.TestCase):
    TEXT = "8 (999) 123-45-67"

    def test_format_and_prefix(self):
        result = A.anon(self.TEXT, [["PHONE", 0, 17]])
        self.assertEqual(digits(result)[:4], digits(self.TEXT)[:4])
        self.assertEqual(non_digits(result), non_digits(self.TEXT))
        self.assertEqual(len(digits(result)), len(digits(self.TEXT)))
        self.assertNotEqual(result, self.TEXT)

    def test_digits_after_prefix_are_randomized(self):
        values = {A.anon(self.TEXT, [["PHONE", 0, 17]]) for _ in range(25)}
        self.assertGreater(len(values), 1)

    def test_short_number_is_unchanged(self):
        self.assertEqual(A.anon("1234", [["PHONE", 0, 4]]), "1234")

    def test_plus_and_brackets_are_preserved(self):
        text = "+7 (999) 123-45-67"
        result = A.anon(text, [["PHONE", 0, 18]])
        self.assertTrue(result.startswith("+7 (999) "))
        self.assertEqual(result[7], ")")


class TestInn(unittest.TestCase):
    def test_ten_digits(self):
        seen = set()
        for _ in range(60):
            result = A.anon("7707083893", [["INN", 0, 10]])
            value = digits(result)
            self.assertEqual(value[:4], "7707")
            self.assertEqual(len(value), 10)
            self.assertTrue(A.validate_inn(value))
            self.assertNotEqual(value, "7707083893")
            seen.add(value)
        self.assertGreaterEqual(len(seen), 58)

    def test_twelve_digits(self):
        for _ in range(60):
            result = A.anon("770708389312", [["INN", 0, 12]])
            value = digits(result)
            self.assertEqual(value[:4], "7707")
            self.assertEqual(len(value), 12)
            self.assertTrue(A.validate_inn(value))

    def test_invalid_original_checksum_still_produces_valid(self):
        result = A.anon("7707083894", [["INN", 0, 10]])
        value = digits(result)
        self.assertTrue(A.validate_inn(value))
        self.assertEqual(value[:4], "7707")

    def test_separators_are_preserved(self):
        result = A.anon("7707 083893", [["INN", 0, 11]])
        self.assertEqual(result[4], " ")
        self.assertTrue(A.validate_inn(digits(result)))

    def test_unexpected_length(self):
        result = A.anon("12345", [["INN", 0, 5]])
        value = digits(result)
        self.assertEqual(len(value), 5)
        self.assertEqual(value[:4], "1234")

    def test_validator(self):
        self.assertTrue(A.validate_inn("7707083893"))
        self.assertFalse(A.validate_inn("7707083894"))
        self.assertFalse(A.validate_inn("123"))


class TestBankCard(unittest.TestCase):
    VALID = "4084520359695002"

    def test_prefix_suffix_and_luhn(self):
        seen = set()
        for _ in range(60):
            result = A.anon(self.VALID, [["BANK_CARD", 0, 16]])
            value = digits(result)
            self.assertEqual(value[:6], "408452")
            self.assertEqual(value[-4:], "5002")
            self.assertEqual(len(value), 16)
            self.assertTrue(A.validate_luhn(value))
            self.assertNotEqual(value, self.VALID)
            seen.add(value)
        self.assertGreaterEqual(len(seen), 58)

    def test_thirteen_digits(self):
        result = A.anon("4084520359695", [["BANK_CARD", 0, 13]])
        value = digits(result)
        self.assertEqual(value[:6], "408452")
        self.assertEqual(value[-4:], "9695")
        self.assertTrue(A.validate_luhn(value))

    def test_spaces_are_preserved(self):
        text = "4084 5203 5969 5002"
        result = A.anon(text, [["BANK_CARD", 0, 19]])
        self.assertEqual(
            [i for i, char in enumerate(text) if char == " "],
            [i for i, char in enumerate(result) if char == " "],
        )
        value = digits(result)
        self.assertEqual(value[:6], "408452")
        self.assertEqual(value[-4:], "5002")
        self.assertTrue(A.validate_luhn(value))

    def test_short_value_does_not_crash(self):
        result = A.anon("12345678", [["BANK_CARD", 0, 8]])
        self.assertEqual(len(digits(result)), 8)
        self.assertRegex(result, r"^\d{8}$")

    def test_ten_digits_cannot_satisfy_all_constraints(self):
        # Первые 6 + последние 4 занимают все 10 цифр, менять нечего.
        self.assertEqual(
            A.anon("4084525002", [["BANK_CARD", 0, 10]]),
            "4084525002",
        )

    def test_validator(self):
        self.assertTrue(A.validate_luhn(self.VALID))
        self.assertFalse(A.validate_luhn(self.VALID[:-1] + "3"))
        self.assertFalse(A.validate_luhn("1234"))


class TestSnils(unittest.TestCase):
    def test_format_and_checksum(self):
        text = "112-233-445 95"
        for _ in range(30):
            result = A.anon(text, [["SNILS", 0, 14]])
            self.assertEqual(non_digits(result), non_digits(text))
            self.assertTrue(A.validate_snils(digits(result)))
            self.assertNotEqual(result, text)

    def test_validator(self):
        self.assertTrue(A.validate_snils("112-233-445 95"))
        self.assertFalse(A.validate_snils("112-233-445 94"))
        self.assertFalse(A.validate_snils("123"))


class TestRobustness(unittest.TestCase):
    def test_unknown_attribute_is_ignored(self):
        text = "Иван"
        self.assertEqual(A.anon(text, [["PASSPORT", 0, 4]]), text)

    def test_out_of_range_is_ignored(self):
        text = "Иван"
        self.assertEqual(A.anon(text, [["NAME", -1, 4]]), text)
        self.assertEqual(A.anon(text, [["NAME", 0, 99]]), text)
        self.assertEqual(A.anon(text, [["NAME", 4, 4]]), text)
        self.assertEqual(A.anon(text, [["NAME", 3, 1]]), text)

    def test_malformed_entries_are_ignored(self):
        text = "Иван"
        attributes = [
            None,
            "NAME",
            {"attr": "NAME", "start": 0, "end": 4},
            ["NAME"],
            ["NAME", "a", "b"],
            ["NAME", True, 4],
            ["NAME", None, 4],
            ["NAME", 0, 4],  # единственная корректная
        ]
        result = A.anon(text, attributes)
        self.assertNotEqual(result, text)

    def test_float_and_string_indices(self):
        self.assertNotEqual(A.anon("Иван", [["NAME", 0.0, 4.0]]), "Иван")
        self.assertNotEqual(A.anon("Иван", [["NAME", "0", "4"]]), "Иван")

    def test_overlapping_entities_do_not_crash(self):
        text = "Иванов"
        result = A.anon(text, [["SURNAME", 0, 6], ["NAME", 0, 3]])
        self.assertNotEqual(result, text)

    def test_aliases(self):
        self.assertNotEqual(
            A.anon("Иван", [["ИМЯ", 0, 4]]),
            "Иван",
        )
        self.assertTrue(
            A.anon("ivanov@example.com", [["E-MAIL", 0, 18]]).endswith(
                "@example.com"
            )
        )
        self.assertTrue(
            A.validate_luhn(
                digits(A.anon("4084520359695002", [["BANK-CARD", 0, 16]]))
            )
        )

    def test_nested_and_duplicate_entities(self):
        text = "Иван"
        self.assertNotEqual(A.anon(text, [["NAME", 0, 4], ["NAME", 0, 4]]), text)

    def test_repeated_non_fio_values_are_consistent(self):
        text = (
            "ivanov@example.com и снова ivanov@example.com, "
            "+7 (999) 123-45-67 и +7 (999) 123-45-67"
        )
        spans: list[list[object]] = []
        for attr, value in [
            ("EMAIL", "ivanov@example.com"),
            ("PHONE", "+7 (999) 123-45-67"),
        ]:
            for match in re.finditer(re.escape(value), text):
                spans.append([attr, match.start(), match.end()])
        result = A.anon(text, spans)
        emails = re.findall(r"[A-Za-z]+@example\.com", result)
        phones = re.findall(r"\+7 \(999\) [\d-]+", result)
        self.assertEqual(len(emails), 2)
        self.assertEqual(len(phones), 2)
        self.assertEqual(emails[0], emails[1])
        self.assertEqual(phones[0], phones[1])


class TestCli(unittest.TestCase):
    def test_parse_spans(self):
        self.assertEqual(
            cli.parse_spans("NAME:0:6, SURNAME:7:14"),
            [["NAME", 0, 6], ["SURNAME", 7, 14]],
        )
        self.assertEqual(
            cli.parse_spans("NAME:0:6;SURNAME:7:14"),
            [["NAME", 0, 6], ["SURNAME", 7, 14]],
        )

    def test_parse_spans_rejects_bad_input(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cli.parse_spans("NAME:0")
        with self.assertRaises(argparse.ArgumentTypeError):
            cli.parse_spans("NAME:a:6")

    def test_cli_with_manual_spans(self):
        result = subprocess.run(
            [
                sys.executable,
                "cli.py",
                "Андрей",
                "Яковлев",
                "--no-model",
                "--spans",
                "NAME:0:6,SURNAME:7:14",
            ],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotEqual(result.stdout.strip(), "Андрей Яковлев")
        self.assertEqual(len(result.stdout.strip().split()), 2)


class TestVariant24Attributes(unittest.TestCase):
    def test_passport_preserves_series_and_format(self):
        raw = "45 10 № 123456"
        res = A.anon(raw, [["PASSPORT", 0, len(raw)]])
        self.assertNotEqual(res, raw)
        self.assertTrue(res.startswith("45 10 № "))
        self.assertTrue(A.validate_passport(res))

    def test_ogrnip_preserves_prefix_and_checksum(self):
        raw = "321774600123451"
        self.assertTrue(A.validate_ogrnip(raw))
        seen = set()
        for _ in range(30):
            res = A.anon(raw, [["OGRNIP", 0, len(raw)]])
            self.assertNotEqual(res, raw)
            self.assertTrue(res.startswith("32177"))
            self.assertTrue(A.validate_ogrnip(res), f"Invalid OGRNIP: {res}")
            seen.add(res)
        self.assertGreater(len(seen), 10)

    def test_birth_cert_preserves_series(self):
        raw = "III-МЮ 123456"
        self.assertTrue(A.validate_birth_cert(raw))
        res = A.anon(raw, [["BIRTH_CERT", 0, len(raw)]])
        self.assertNotEqual(res, raw)
        self.assertTrue(res.startswith("III-МЮ "))
        self.assertTrue(A.validate_birth_cert(res))

    def test_kladr_preserves_region_and_relevance(self):
        for raw in ("7700000100000", "50012000005001200"):
            self.assertTrue(A.validate_kladr(raw))
            res = A.anon(raw, [["KLADR", 0, len(raw)]])
            self.assertNotEqual(res, raw)
            self.assertEqual(res[:2], raw[:2])
            self.assertEqual(res[-2:], raw[-2:])
            self.assertEqual(len(res), len(raw))
            self.assertTrue(A.validate_kladr(res))

    def test_ip_preserves_family_and_subnet(self):
        for raw, prefix in (
            ("192.168.1.105", "192.168."),
            ("10.0.15.42", "10."),
            ("85.142.20.11", "85."),
        ):
            self.assertTrue(A.validate_ip(raw))
            res = A.anon(raw, [["IP", 0, len(raw)]])
            self.assertNotEqual(res, raw)
            self.assertTrue(res.startswith(prefix))
            self.assertTrue(A.validate_ip(res))

        ipv6 = "2001:db8:85a3::8a2e:370:7334"
        self.assertTrue(A.validate_ip(ipv6))
        res_v6 = A.anon(ipv6, [["IP", 0, len(ipv6)]])
        self.assertNotEqual(res_v6, ipv6)
        self.assertTrue(A.validate_ip(res_v6))


class TestContextDisambiguation(unittest.TestCase):
    def test_unisex_name_with_verb_context(self):
        text_m = "Саша подписал приказ"
        res_m = A.anon(text_m, [["NAME", 0, 4]])
        first_m = res_m.split()[0]
        parses_m = [p for p in MORPH.parse(first_m) if "Name" in p.tag.grammemes]
        self.assertTrue(any(p.tag.gender == "masc" for p in parses_m), res_m)

        text_f = "Саша подписала приказ"
        res_f = A.anon(text_f, [["NAME", 0, 4]])
        first_f = res_f.split()[0]
        parses_f = [p for p in MORPH.parse(first_f) if "Name" in p.tag.grammemes]
        self.assertTrue(any(p.tag.gender == "femn" for p in parses_f), res_f)

        # Два разных человека с именем Саша в одном сложносочинённом предложении
        compound = "Саша подписал договор, а Саша подписала акт."
        res_c = A.anon(compound, [["NAME", 0, 4], ["NAME", 25, 29]])
        words_c = res_c.split()
        self.assertTrue(
            any(p.tag.gender == "masc" for p in MORPH.parse(words_c[0]) if "Name" in p.tag.grammemes),
            res_c,
        )
        self.assertTrue(
            any(p.tag.gender == "femn" for p in MORPH.parse(words_c[4]) if "Name" in p.tag.grammemes),
            res_c,
        )

    def test_unisex_name_with_surname(self):
        text_m = "Саша Иванов"
        res_m = A.anon(text_m, [["NAME", 0, 4], ["SURNAME", 5, 11]])
        name_m, surn_m = res_m.split()
        self.assertTrue(
            any(p.tag.gender == "masc" for p in MORPH.parse(name_m) if "Name" in p.tag.grammemes),
            res_m,
        )

        text_f = "Саша Иванова"
        res_f = A.anon(text_f, [["NAME", 0, 4], ["SURNAME", 5, 12]])
        name_f, surn_f = res_f.split()
        self.assertTrue(
            any(p.tag.gender == "femn" for p in MORPH.parse(name_f) if "Name" in p.tag.grammemes),
            res_f,
        )

    def test_standalone_surname_disambiguated_by_preposition_and_noun(self):
        text_gen_m = "Документы получены от сотрудника Иванова"
        start = text_gen_m.index("Иванова")
        res = A.anon(text_gen_m, [["SURNAME", start, start + 7]])
        new_surname = res.split()[-1]
        self.assertTrue(
            any(
                p.tag.gender == "masc" and p.tag.case == "gent"
                for p in MORPH.parse(new_surname)
                if "Surn" in p.tag.grammemes
            ),
            f"Expected masculine genitive surname, got: {res}",
        )

    def test_mask_pii(self):
        text = "Иван получил кредит"
        self.assertEqual(A.mask_pii(text, [["NAME", 0, 4]]), "**** получил кредит")
        self.assertEqual(A.mask_pii(text, full_text=True), "*" * len(text))


if __name__ == "__main__":
    unittest.main(verbosity=2)
