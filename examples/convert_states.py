"""Пересчёт калибровочных кривых кукварта в формат приложения.

Исходный ``states.npz`` — калибровка четырёхуровневой системы (кукварта),
считанной одним импульсом на трёх частотах. Файл содержит четыре массива:

* ``arr_0`` — квадратура I всех выстрелов, int16
  ``[1, выстрел, состояние, отсчёт]``;
* ``arr_1`` — средняя по выстрелам I для каждого состояния, float64
  ``[состояние, отсчёт]``;
* ``arr_2`` — квадратура Q всех выстрелов, та же форма, что у ``arr_0``;
* ``arr_3`` — средняя по выстрелам Q ``[состояние, отсчёт]``.

Средние — это в точности ``arr_0``/``arr_2``, усреднённые по выстрелам;
скрипт проверяет это при пересчёте и падает, если раскладка файла другая.

Приложению нужен формат ``iq [N, A, T] complex64`` и ``labels [N, A] int64``
(README.txt, раздел о наборах данных). Кукварт — это один канал
считывания, поэтому ``A = 1``, а метка — подготовленное состояние 0…3.
Демодуляции нет: три частоты уже смешаны в одном отклике, кривые
сохраняются как есть. Комплексный сигнал собирается из квадратур так же,
как в ``convert_q13.py``: ``I - 1j*Q``.

Строки результата идут в порядке «выстрел, состояние», поэтому подряд
идущий кусок файла содержит все четыре состояния — штатное разбиение на
train/val/test в приложении остаётся корректным.

Запуск (из любого каталога)::

    python examples/convert_states.py
    python examples/convert_states.py --chunk 4096 --force
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from uuid import uuid4

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

CHUNK = 2048  # выстрелов за один проход (ограничивает память)


def convert(src, dst, *, chunk=CHUNK):
    """Прочитать сырой ``src`` и записать ``dst`` в формате приложения."""
    raw = np.load(src, mmap_mode="r")
    expected = {"arr_0", "arr_1", "arr_2", "arr_3"}
    if expected - set(raw.files):
        raise ValueError(f"Ожидались массивы {sorted(expected)}, а в файле: {raw.files}.")
    port_i, port_q = raw["arr_0"], raw["arr_2"]
    mean_i, mean_q = raw["arr_1"], raw["arr_3"]
    if port_i.shape != port_q.shape or port_i.ndim != 4 or port_i.shape[0] != 1:
        raise ValueError(f"Ожидались квадратуры формы [1, выстрел, состояние, отсчёт], "
                         f"а в файле {port_i.shape} и {port_q.shape}.")
    n_shots, n_states, n_samples = port_i.shape[1:]
    if mean_i.shape != (n_states, n_samples) or mean_q.shape != mean_i.shape:
        raise ValueError(f"Средние должны иметь форму [{n_states}, {n_samples}], "
                         f"а в файле {mean_i.shape} и {mean_q.shape}.")

    iq = np.empty((n_shots * n_states, 1, n_samples), dtype=np.complex64)
    labels = np.tile(np.arange(n_states, dtype=np.int64), n_shots)[:, None]
    sum_i = np.zeros((n_states, n_samples), dtype=np.float64)
    sum_q = np.zeros((n_states, n_samples), dtype=np.float64)

    started = time.time()
    for first in range(0, n_shots, chunk):
        last = min(first + chunk, n_shots)
        block_i = port_i[0, first:last].astype(np.float32)
        block_q = port_q[0, first:last].astype(np.float32)
        sum_i += block_i.sum(axis=0, dtype=np.float64)
        sum_q += block_q.sum(axis=0, dtype=np.float64)
        # Квадратуры -> комплексная трасса, как в convert_q13.py.
        iq[first * n_states:last * n_states, 0] = (block_i - 1j * block_q).reshape(-1, n_samples)
        print(f"{last}/{n_shots} выстрелов ({time.time() - started:.1f} c)")

    for name, got, want in (("arr_1", sum_i / n_shots, mean_i), ("arr_3", sum_q / n_shots, mean_q)):
        if not np.allclose(got, want, rtol=1e-9, atol=1e-3):
            raise ValueError(f"{name} не совпадает со средним по выстрелам: раскладка файла другая.")

    # Запись во временный файл рядом с целевым: недописанный набор не появится.
    temporary = dst.with_name(f"{dst.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, iq=iq, labels=labels)
        temporary.replace(dst)
    finally:
        temporary.unlink(missing_ok=True)
    return iq, labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", nargs="?", default=HERE / "states.npz")
    parser.add_argument("dst", nargs="?", default=HERE / "ququart_4levels.npz")
    parser.add_argument("--chunk", type=int, default=CHUNK, help="выстрелов за проход")
    parser.add_argument("--force", action="store_true", help="перезаписать существующий dst")
    args = parser.parse_args(argv)

    src, dst = pathlib.Path(args.src), pathlib.Path(args.dst)
    if dst.exists() and not args.force:
        parser.error(f"{dst} уже существует; добавьте --force, чтобы перезаписать.")
    try:
        iq, labels = convert(src, dst, chunk=args.chunk)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Ошибка пересчёта: {error}\n")

    print(f"готово: {dst}")
    print(f"  iq     {iq.shape} {iq.dtype}, {iq.nbytes / 2 ** 20:.0f} МБ в памяти")
    print(f"  labels {labels.shape} {labels.dtype}, состояния {np.unique(labels)}")
    print(f"  размер файла {dst.stat().st_size / 2 ** 20:.0f} МБ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
