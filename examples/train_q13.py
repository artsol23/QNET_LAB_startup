"""Обучение сети режима matched на данных Q13 и отчёт по точности.

Данные готовит ``convert_q13.py`` (файл ``q13_2qubits.npz``). Скрипт повторяет
то, что делает приложение: разбиение по выстрелам, обучение с валидацией,
затем метрики на отложенной части и матрица ошибок по четырём состояниям.

Запуск (из любого каталога)::

    python examples/train_q13.py
    python examples/train_q13.py --epochs 60 --hidden 64 32 --seed 7
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from blas_threads import limit_blas_threads  # noqa: E402

# До первого import numpy: OpenBLAS читает число потоков при загрузке.
limit_blas_threads()

import numpy as np  # noqa: E402
from neural_service import NeuralService  # noqa: E402

DATA = HERE / "q13_2qubits.npz"
LEVELS = [2, 2]
ATOMS = [0, 1]
STATE_NAMES = ("00", "01", "10", "11")


def load(path=DATA):
    """Прочитать набор данных через тот же загрузчик, что и приложение."""
    return NeuralService().load_data(str(path))


def split(n, *, seed=0, test_frac=1 / 6, val_frac=1 / 10):
    """Разбить измерения на train/val/test, сохранив все состояния в каждой части.

    Строки файла идут в порядке «выстрел, состояние», поэтому перемешиваются
    именно выстрелы: четыре состояния одного выстрела всегда попадают в одну
    часть и не утекают между train и test.
    """
    perm = np.random.default_rng(seed).permutation(n)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    return (np.sort(perm[n_test + n_val:]),      # train
            np.sort(perm[n_test:n_test + n_val]),  # val
            np.sort(perm[:n_test]))              # test


def fit(data, train_idx, val_idx, *, hidden=(22, 11), seed=42, epochs=40,
        batch_size=256, learning_rate=1e-3, on_epoch=None):
    """Обучить новую сеть режима matched; вернуть (сервис, результат, время)."""
    service = NeuralService()
    config = service.configure(LEVELS, list(hidden), atoms=ATOMS, mode="matched", seed=seed)
    started = time.time()
    result = service.train(data, train_indices=train_idx, val_indices=val_idx, val_data=data,
                           config=config, epochs=epochs, batch_size=batch_size,
                           learning_rate=learning_rate, callback=on_epoch)
    return service, result, time.time() - started


def confusion(service, data, indices):
    """Матрица 4x4: строка — подготовленное состояние, столбец — предсказанное."""
    report = service.validate(data, indices)
    predicted = np.asarray(report["predictions"])
    truth = data["labels"][indices]
    table = np.zeros((4, 4), dtype=int)
    for want, got in zip(truth[:, 0] * 2 + truth[:, 1], predicted[:, 0] * 2 + predicted[:, 1]):
        table[want, got] += 1
    return report, table


def print_report(title, report, table):
    print(f"--- {title} ({len(report['predictions'])}) ---")
    for key in ("accuracy", "assignment_fidelity", "macro_f1", "loss"):
        print(f"  {key:20s} {report[key]:.4f}")
    print(f"  бенчмарки: {report['benchmarks']}")
    for atom, stats in enumerate(report["per_atom"]):
        print(f"  атом {atom}: accuracy {stats['accuracy']:.4f}")
        print(f"    матрица ошибок: {stats['confusion'].tolist()}")
    print("  состояния:      " + "  ".join(f"→{name}" for name in STATE_NAMES))
    for name, row in zip(STATE_NAMES, table):
        print(f"    {name}  " + "  ".join(f"{value:5d}" for value in row)
              + f"   верно {row.max() / row.sum():.1%}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=DATA, help="набор данных в формате приложения")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[22, 11])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    data = load(args.data)
    n = len(data["labels"])
    print(f"N = {n}  A = {data['iq'].shape[1]}  T = {data['iq'].shape[2]}")
    train_idx, val_idx, test_idx = split(n)
    print(f"train {len(train_idx)}  val {len(val_idx)}  test {len(test_idx)}")

    def on_epoch(epoch):
        if epoch["epoch"] % 5 == 0 or epoch["epoch"] == 1:
            print(f"  эпоха {epoch['epoch']:3d}  loss {epoch['loss']:.4f}"
                  f"  train {epoch['accuracy']:.4f}  val {epoch.get('val_accuracy', float('nan')):.4f}")

    service, result, seconds = fit(data, train_idx, val_idx, hidden=args.hidden, seed=args.seed,
                                   epochs=args.epochs, batch_size=args.batch_size,
                                   learning_rate=args.lr, on_epoch=on_epoch)
    print(f"\nобучение: {seconds:.1f} c, эпох {result['epochs_completed']}, "
          f"отменено={result['cancelled']}")
    print(f"параметров: {service.describe()['parameters']}  признаков: {service.feature_count}")

    for title, indices in (("валидация", val_idx), ("тест", test_idx)):
        report, table = confusion(service, data, indices)
        print_report(title, report, table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
