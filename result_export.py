"""Экспорт готовых результатов без зависимости от Qt и PyTorch."""

from __future__ import annotations

import csv
from pathlib import Path

from project_store import ProjectStore


def write_predictions_csv(
    path: str | Path,
    *,
    result: dict,
    register_states,
) -> None:
    """Записать предсказания и вероятности через временный файл."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")

    atoms = result["atoms"]
    predictions = result["predictions"]
    probabilities = result["probabilities"]
    shot_ids = result["shot_ids"]
    labels = result.get("labels")

    count = len(predictions)

    if len(register_states) != count or len(shot_ids) != count:
        raise ValueError(
            "Число предсказаний, состояний регистра "
            "и идентификаторов измерений не совпадает."
        )

    if predictions.ndim != 2 or predictions.shape[1] != len(atoms):
        raise ValueError(
            "Форма массива предсказаний не соответствует списку атомов."
        )

    if labels is not None and labels.shape != predictions.shape:
        raise ValueError(
            "Форма массива меток не соответствует предсказаниям."
        )

    if len(probabilities) != len(atoms):
        raise ValueError(
            "Число массивов вероятностей не соответствует числу атомов."
        )

    for channel, values in zip(atoms, probabilities):
        if values.ndim != 2 or len(values) != count:
            raise ValueError(
                f"Некорректная форма вероятностей атома {channel}."
            )

    header = ["shot_id", "register_state"]

    for index, channel in enumerate(atoms):
        if labels is not None:
            header.append(f"label_{channel}")

        header.append(f"prediction_{channel}")
        header.extend(
            f"p_{channel}_{level}"
            for level in range(probabilities[index].shape[1])
        )

    try:
        with temporary.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(header)

            for row_index in range(count):
                row = [
                    int(shot_ids[row_index]),
                    int(register_states[row_index]),
                ]

                for atom_index in range(len(atoms)):
                    if labels is not None:
                        row.append(
                            int(labels[row_index, atom_index])
                        )

                    row.append(
                        int(predictions[row_index, atom_index])
                    )

                    row.extend(
                        f"{value:.6g}"
                        for value in probabilities[atom_index][row_index]
                    )

                writer.writerow(row)

        temporary.replace(path)

    finally:
        temporary.unlink(missing_ok=True)


def write_report_json(
    path: str | Path,
    *,
    result: dict,
    metadata: dict,
) -> None:
    """Записать отчёт с метриками и происхождением результатов."""
    path = Path(path)

    metric_keys = (
        "accuracy",
        "assignment_fidelity",
        "macro_f1",
        "loss",
        "xeb_labels",
        "xeb_linear",
        "benchmarks",
        "excluded",
    )

    report = {
        **metadata,
        "metrics": {
            key: result.get(key)
            for key in metric_keys
        },
        "per_atom": result.get("per_atom"),
        "warnings": result.get("warnings"),
        "notes": (
            "score = доля измерений со всеми атомами верно; "
            "F_nQ — геометрическое среднее assignment fidelity "
            "атомов; физические оценки включают SPAM."
        ),
    }

    ProjectStore(path.parent).write_json(path, report)
