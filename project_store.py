"""Работа с файлами проекта без зависимости от Qt и PyTorch."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


class ProjectStore:
    """Пути, JSON, контроль целостности и восстановление данных проекта."""

    def __init__(self, folder: str | Path):
        self.project_dir = Path(folder).resolve()

    @staticmethod
    def stamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def inside(self, relative: str | Path) -> Path:
        """Получить путь, не выходящий за пределы проекта."""
        path = (self.project_dir / relative).resolve()

        if (
            path != self.project_dir
            and self.project_dir not in path.parents
        ):
            raise ValueError(
                f"Путь {relative} выходит за пределы папки проекта."
            )

        return path

    def relative_path(self, path: str | Path) -> str:
        """Получить относительный путь для записи в project.json."""
        return (
            Path(path)
            .resolve()
            .relative_to(self.project_dir)
            .as_posix()
        )

    def unique_path(
        self,
        folder: str,
        prefix: str,
        suffix: str,
    ) -> Path:
        """Подобрать свободное имя файла в подпапке проекта."""
        base = self.inside(folder)
        base.mkdir(exist_ok=True)

        stamp = self.stamp()
        path = base / f"{prefix}_{stamp}{suffix}"

        counter = 1
        while path.exists():
            path = base / f"{prefix}_{stamp}_{counter}{suffix}"
            counter += 1

        return path

    @staticmethod
    def sha256(path: str | Path) -> str:
        """Вычислить SHA-256, читая файл блоками."""
        digest = hashlib.sha256()

        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)

        return digest.hexdigest()

    @staticmethod
    def jsonable(value):
        """Привести значения к типам, поддерживаемым JSON."""
        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, dict):
            return {
                str(key): ProjectStore.jsonable(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [ProjectStore.jsonable(item) for item in value]

        if isinstance(value, float) and not math.isfinite(value):
            return None

        return value

    def write_json(self, path: Path, payload) -> None:
        """Записать JSON через временный файл и атомарное переименование."""
        temporary = path.with_name(path.name + ".tmp")

        temporary.write_text(
            json.dumps(
                self.jsonable(payload),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    def read_dataset(
        self,
        entry: dict,
        warnings: list[str],
        load_data: Callable[[Path], dict],
    ) -> dict:
        """Восстановить набор данных через переданный загрузчик.

        entry обновляется результатами проверки SHA-256.
        warnings пополняется предупреждениями для пользователя.
        """
        if entry.get("copied"):
            file = self.inside(entry["file"])

            if self.project_dir not in file.parents:
                raise ValueError(
                    "Путь к данным выходит за пределы проекта."
                )
        else:
            # Исходные данные могут храниться вне проекта,
            # если пользователь импортировал их без копирования.
            file = Path(entry["file"])

        if not file.is_file():
            raise ValueError(
                f"Файл данных проекта не найден: {file}"
            )

        entry["actual_sha256"] = self.sha256(file)

        if entry["actual_sha256"] != entry.get("sha256"):
            warnings.append(
                f"Файл {file.name} изменился после импорта "
                "(SHA-256 не совпадает). "
                "Разбиение сброшено; проверка на test для прежних "
                "весов будет помечена как неподтверждённая."
            )
            entry["sha_mismatch"] = True

        data = load_data(file)

        if entry.get("labels_override"):
            labels_file = self.inside(entry["labels_override"])

            if not labels_file.is_file():
                warnings.append(
                    "Файл исправленных меток "
                    f"{entry['labels_override']} не найден — "
                    "используются исходные метки подготовки."
                )
                entry["labels_override"] = None
                return data

            labels = np.load(labels_file, allow_pickle=False)

            if (
                data["labels"] is None
                or labels.shape != data["labels"].shape
            ):
                raise ValueError(
                    "Файл исправленных меток не соответствует данным."
                )

            data["prepared_labels"] = data["labels"]
            data["labels"] = labels.astype(np.int64)

        return data
