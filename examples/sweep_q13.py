"""Проверка, упирается ли точность на данных Q13 в размер сети.

Сеть перебирается от совсем маленькой до крупной; если точность от размера
не зависит, то предел задан данными, а не моделью. Данные и разбиение те же,
что в ``train_q13.py``.

Запуск (из любого каталога)::

    python examples/sweep_q13.py
    python examples/sweep_q13.py --epochs 20 --hidden 8 4 --hidden 512 256 --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_q13 import DATA, fit, load, split  # noqa: E402

SIZES = [[4, 2], [22, 11], [64, 32], [128, 64]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, nargs="+", action="append", default=None,
                        help="размер скрытых слоёв; можно повторить (по умолчанию %s)" % SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    args = parser.parse_args(argv)

    torch.set_num_threads(1)

    data = load(args.data)
    train_idx, val_idx, test_idx = split(len(data["labels"]))
    print(f"train {len(train_idx)}  val {len(val_idx)}  test {len(test_idx)}")
    print(f"{'скрытые слои':>14s}  {'seed':>4s}  {'параметров':>10s}  {'val':>6s}  {'test':>6s}")

    rows = []
    for hidden in args.hidden or SIZES:
        for seed in args.seeds:
            service, result, _ = fit(data, train_idx, val_idx, hidden=hidden, seed=seed,
                                     epochs=args.epochs)
            val = service.validate(data, val_idx)["accuracy"]
            test = service.validate(data, test_idx)["accuracy"]
            print(f"{str(hidden):>14s}  {seed:4d}  {service.describe()['parameters']:10d}"
                  f"  {val:.4f}  {test:.4f}")
            rows.append(test)

    if rows:
        print(f"\nтест: {min(rows):.4f} … {max(rows):.4f}, среднее {sum(rows) / len(rows):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
