"""Проверки скриптов examples/ и работы validation в train().

* разбиение Q13 не должно отдавать состояния одного выстрела в разные части:
  иначе сеть видит test при обучении;
* пересчёт Q13 должен работать при decim, не делящем длину трассы;
* индексы validation, заданные вместе с val_data, должны относиться к
  val_data, а не к всему набору, и пересечение с train запрещено.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import convert_q13  # noqa: E402
from train_q13 import STATES_PER_SHOT, split  # noqa: E402

from neural_service import NeuralService  # noqa: E402


def make_noisy_data(samples=300, trace=8):
    """Метки случайны: сеть выучит train и не выучит validation."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=(samples, 1)).astype(np.int64)
    iq = rng.normal(0, 1, (samples, 1, trace)).astype(np.complex64)
    return {"iq": iq, "labels": labels}


class SplitTests(unittest.TestCase):
    def test_states_of_one_shot_stay_together(self):
        shots = 20
        train, validation, test = split(shots * STATES_PER_SHOT)

        def shot_of(indices):
            return set((indices // STATES_PER_SHOT).tolist())

        self.assertEqual(shot_of(train) & shot_of(validation), set())
        self.assertEqual(shot_of(train) & shot_of(test), set())
        self.assertEqual(shot_of(validation) & shot_of(test), set())

    def test_every_shot_is_complete_in_one_part(self):
        shots = 20
        train, validation, test = split(shots * STATES_PER_SHOT)

        for indices in (train, validation, test):
            counts = np.bincount(indices // STATES_PER_SHOT, minlength=shots)
            # Выстрел либо целиком в этой части, либо целиком вне её.
            self.assertTrue(np.all((counts == 0) | (counts == STATES_PER_SHOT)))

    def test_all_rows_used_once(self):
        shots = 20
        total = shots * STATES_PER_SHOT
        train, validation, test = split(total)

        joined = np.concatenate([train, validation, test])
        self.assertEqual(len(joined), total)
        self.assertEqual(len(np.unique(joined)), total)


class ConvertTests(unittest.TestCase):
    def test_decim_that_does_not_divide_the_trace(self):
        shots, states, samples = 2, 4, 20
        rng = np.random.default_rng(0)
        source = Path(tempfile.mkdtemp())
        raw = source / "raw.npz"
        np.savez(
            raw,
            arr_0=rng.normal(0, 1, (shots, states, samples)).astype(np.float32),
            arr_1=rng.normal(0, 1, (shots, states, samples)).astype(np.float32),
        )

        iq, labels = convert_q13.convert(raw, source / "out.npz", decim=3)

        self.assertEqual(iq.shape, (shots * states, 2, -(-samples // 3)))
        self.assertEqual(labels.shape, (shots * states, 2))


class ValidationIndicesTests(unittest.TestCase):
    def test_overlapping_validation_is_rejected(self):
        service = NeuralService()
        service.configure([2], [8], atoms=[0], mode="matched", seed=1)
        data = make_noisy_data(60)
        indices = np.arange(len(data["labels"]))

        with self.assertRaises(ValueError):
            service.train(data, train_indices=indices[:40], val_indices=indices[:40],
                          val_data=data, epochs=1, batch_size=16)

    def test_validation_covers_only_given_indices(self):
        service = NeuralService()
        service.configure([2], [16], atoms=[0], mode="matched", seed=1)
        data = make_noisy_data(300)
        indices = np.arange(len(data["labels"]))
        train_idx, val_idx = indices[:200], indices[200:]

        report = service.train(data, train_indices=train_idx, val_indices=val_idx,
                               val_data=data, epochs=20, batch_size=32, learning_rate=0.02)

        reported = report["last"]["val_accuracy"]
        whole = service.validate(data, indices)["accuracy"]
        validation = service.validate(data, val_idx)["accuracy"]

        # Validation — это только val_idx, а не весь набор с train.
        self.assertLess(reported, whole)
        self.assertAlmostEqual(reported, validation, delta=0.05)


if __name__ == "__main__":
    unittest.main()
