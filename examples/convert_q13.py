"""Пересчёт сырых экспериментальных данных Q13 в формат приложения.

Исходный ``Q13.npz`` содержит результат многократных экспериментов над двумя
кубитами: четыре подготовленных состояния (00, 01, 10, 11) и две квадратуры
одной линии считывания. Данные записаны ДО демодуляции, поэтому в каждой
квадратуре смешаны отклики двух кубитов на разных несущих частотах, а сам
сигнал вещественный.

Приложению нужен другой формат (см. README.txt, раздел о наборах данных):
``iq [N, A, T] complex64`` и ``labels [N, A] int64``, где ``A`` — каналы
мультиплексирования, и в каждом канале лежит уже демодулированная комплексная
трасса ``I + jQ`` одного кубита.

Пересчёт: смеситель на несущую кубита (``I - jQ`` после умножения на
``exp(-2*pi*i*f*t)`` даёт квадратуру внутрь канала 0), затем децимация
``resample_poly`` с FIR-фильтром (он же подавляет вторую несущую и образы
смесителя). Строки результата идут в порядке «выстрел, состояние», так что
подряд идущий кусок файла содержит все четыре состояния — штатное разбиение
на train/val/test в приложении остаётся корректным.

Несущие ниже измерены по спектру усреднённой трассы (уточнение по нулевому
дополнению БПФ); ошибка уточнения ~4e-5 отсчёта^-1, набег фазы за трассу
меньше 0.6 рад. Децимация x10 (T = 230) выбрана по замеру точности:
2300, 1150, 460, 230 и 115 отсчётов дают одинаковый результат, поэтому берётся
наименьший объём без потерь.

Файл ``Q13.npz`` — zip-архив, и он читается в память целиком (около 600 МБ
на два порта): пошаговая обработка ограничивает только результат, а не чтение
исходника.

Запуск (из любого каталога)::

    python examples/convert_q13.py
    python examples/convert_q13.py --decim 20 --dst /tmp/q13_light.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
from scipy.signal import resample_poly

HERE = pathlib.Path(__file__).resolve().parent

# Несущие каналов, в единицах отсчёта^-1: кубит 1 и кубит 2.
F_ATOMS = (0.177960, 0.034976)

# Возможные состояния в порядке строк исходного файла: 00, 01, 10, 11.
STATES = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)

DECIM = 10        # прореживание по времени
CHUNK = 1000      # выстрелов за один проход (ограничивает память)


def convert(src, dst, *, decim=DECIM, chunk=CHUNK, f_atoms=F_ATOMS):
    """Прочитать сырой ``src`` и записать ``dst`` в формате приложения."""
    raw = np.load(src, mmap_mode="r")
    if {"arr_0", "arr_1"} - set(raw.files):
        raise ValueError(f"Ожидались массивы arr_0 и arr_1 (две квадратуры), а в файле: {raw.files}.")
    port_i, port_q = raw["arr_0"], raw["arr_1"]
    if port_i.shape != port_q.shape or port_i.ndim != 3:
        raise ValueError(f"Ожидалась одинаковая форма [выстрел, состояние, отсчёт]: "
                         f"{port_i.shape} и {port_q.shape}.")
    n_shots, n_states, n_samples = port_i.shape
    if n_states != len(STATES) or len(f_atoms) != STATES.shape[1]:
        raise ValueError("Число состояний или каналов не соответствует STATES.")

    mixer = [np.exp(-2j * np.pi * f * np.arange(n_samples)).astype(np.complex64) for f in f_atoms]
    # resample_poly округляет число отсчётов вверх: n_samples // decim не годится.
    n_out = -(-n_samples // decim)
    iq = np.empty((n_shots * n_states, len(f_atoms), n_out), dtype=np.complex64)
    labels = np.tile(STATES, (n_shots, 1))

    started = time.time()
    for first in range(0, n_shots, chunk):
        last = min(first + chunk, n_shots)
        block = np.empty((last - first, n_states, len(f_atoms), n_out), dtype=np.complex64)
        for state in range(n_states):
            # Вещественный отклик двух квадратур -> аналитический сигнал.
            signal = (port_i[first:last, state].astype(np.float32)
                      - 1j * port_q[first:last, state].astype(np.float32))
            for atom, mix in enumerate(mixer):
                block[:, state, atom] = resample_poly(signal * mix, 1, decim, axis=-1)
        iq[first * n_states:last * n_states] = block.reshape(-1, len(f_atoms), n_out)
        print(f"{last}/{n_shots} выстрелов ({time.time() - started:.1f} c)")

    np.savez(dst, iq=iq, labels=labels)
    return iq, labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", nargs="?", default=HERE / "Q13.npz")
    parser.add_argument("dst", nargs="?", default=HERE / "q13_2qubits.npz")
    parser.add_argument("--decim", type=int, default=DECIM, help="прореживание по времени (по умолчанию 10)")
    parser.add_argument("--chunk", type=int, default=CHUNK, help="выстрелов за проход")
    parser.add_argument("--force", action="store_true", help="перезаписать существующий dst")
    args = parser.parse_args(argv)

    src, dst = pathlib.Path(args.src), pathlib.Path(args.dst)
    if dst.exists() and not args.force:
        parser.error(f"{dst} уже существует; добавьте --force, чтобы перезаписать.")
    try:
        iq, labels = convert(src, dst, decim=args.decim, chunk=args.chunk)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Ошибка пересчёта: {error}\n")

    print(f"готово: {dst}")
    print(f"  iq     {iq.shape} {iq.dtype}, {iq.nbytes / 2 ** 20:.0f} МБ в памяти")
    print(f"  labels {labels.shape} {labels.dtype}")
    print(f"  размер файла {dst.stat().st_size / 2 ** 20:.0f} МБ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
