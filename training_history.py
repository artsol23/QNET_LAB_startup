"""Чтение и проверка журналов обучения без Qt и PyTorch."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


HISTORY_KEYS = (
    "epoch",
    "loss",
    "accuracy",
    "val_loss",
    "val_accuracy",
    "val_assignment_fidelity",
    "val_macro_f1",
    "val_xeb",
    "val_xeb_linear",
    "lr",
)


def clean_history(rows) -> list[dict]:
    """Проверить историю и привести значения к числовым типам.

    epoch — положительное целое число.
    Остальные поля — числа или None.
    Неизвестные поля пропускаются.
    """
    if not isinstance(rows, list):
        raise ValueError("Журнал должен быть списком записей эпох.")

    cleaned = []

    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or "epoch" not in row:
            raise ValueError(
                f"Запись {row_number}: нужен столбец epoch."
            )

        item = {}

        for key in HISTORY_KEYS:
            value = row.get(key)

            if value is None:
                item[key] = None
                continue

            if isinstance(value, str):
                value = value.strip()

                if value.lower() in ("", "none", "null", "nan"):
                    item[key] = None
                    continue

            # bool является подклассом int, но не является
            # корректной записью числовой метрики.
            if isinstance(value, bool):
                raise ValueError(
                    f"Запись {row_number}, поле {key}: "
                    "ожидалось число, получено логическое значение."
                )

            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Запись {row_number}, поле {key}: "
                    f"некорректное число {value!r}."
                ) from exc

            item[key] = number if math.isfinite(number) else None

        epoch = item["epoch"]

        if epoch is None or epoch < 1 or not epoch.is_integer():
            raise ValueError(
                f"Запись {row_number}: некорректный номер эпохи "
                f"{row.get('epoch')!r}."
            )

        item["epoch"] = int(epoch)
        cleaned.append(item)

    return cleaned


def read_history(path: str | Path) -> list[dict]:
    """Прочитать CSV, JSON-список или history из project.json."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open(encoding="utf-8-sig") as stream:
            payload = json.load(stream)

        if isinstance(payload, dict):
            if "history" not in payload:
                raise ValueError(
                    "В JSON отсутствует поле history."
                )
            rows = payload["history"]
        else:
            rows = payload

    elif suffix == ".csv":
        with path.open(
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            sample = stream.read(4096)
            stream.seek(0)

            if not sample.strip():
                raise ValueError("Файл журнала пуст.")

            # Разделитель определяем по заголовку:
            # в строках данных могут быть пустые метрики.
            header = sample.splitlines()[0]

            try:
                dialect = csv.Sniffer().sniff(
                    header,
                    delimiters=",;\t",
                )
            except csv.Error:
                dialect = csv.excel

            reader = csv.DictReader(stream, dialect=dialect)

            if reader.fieldnames is None:
                raise ValueError("В CSV отсутствует заголовок.")

            reader.fieldnames = [
                name.strip()
                for name in reader.fieldnames
            ]

            if "epoch" not in reader.fieldnames:
                raise ValueError(
                    "В CSV отсутствует столбец epoch."
                )

            rows = list(reader)

    else:
        raise ValueError(
            "Поддерживаются журналы в формате CSV или JSON."
        )

    return clean_history(rows)
