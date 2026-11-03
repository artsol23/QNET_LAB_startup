"""Проверки исправлений мелких ошибок.

* метрика F_nQ (assignment_fidelity) не должна пропадать, если в
  оцениваемой подвыборке нет одного из уровней;
* отчёт об обучении должен содержать ключ "excluded" и при отмене
  до начала первой эпохи;
* точность на обучающей выборке не должна зависеть от dropout;
* запись project.json не должна ломаться при одновременной записи
  из двух потоков и не должна оставлять временных файлов;
* ползунок и поле learning rate должны показывать одно значение.
"""

import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from neural_service import NeuralService
from project_store import ProjectStore

from main import preload_system_icu

preload_system_icu()

from mainwindow import MainWindow


def make_data(samples=64, samples_count=8):
    """Разделимая задача: уровень 0 — I ≈ +2, уровень 1 — I ≈ -2."""
    rng = np.random.default_rng(0)
    labels = np.array([[index % 2] for index in range(samples)], dtype=np.int64)
    iq = rng.normal(0, 0.05, (samples, 1, samples_count)).astype(np.complex64)
    iq[labels[:, 0] == 0, 0, :] += 2.0
    iq[labels[:, 0] == 1, 0, :] -= 2.0
    return {"iq": iq, "labels": labels}


class AssignmentFidelityTests(unittest.TestCase):
    def test_fidelity_present_when_one_level_is_missing(self):
        state = SimpleNamespace(config={"levels": [2], "atoms": [0]})
        evaluated = {
            "loss_sum": 0.0,
            "right": 10,
            "n": 10,
            "confusion": [np.array([[10, 0], [0, 0]])],
            "has_labels": True,
            "xeb_sum": 0.0,
            "xeb_n": 0,
        }

        metrics = NeuralService._metrics(state, evaluated)

        self.assertIsNotNone(metrics["assignment_fidelity"])
        self.assertEqual(metrics["assignment_fidelity"], 1.0)
        self.assertEqual(metrics["per_atom"][0]["assignment_fidelity"], 1.0)
        self.assertIsNotNone(metrics["macro_f1"])


class TrainReportTests(unittest.TestCase):
    def test_report_contains_excluded_on_early_cancel(self):
        service = NeuralService()
        service.configure([2], hidden_layers=[8], atoms=[0], mode="raw", seed=3)
        cancelled = threading.Event()
        cancelled.set()

        report = service.train(make_data(), epochs=1, cancel_event=cancelled)

        self.assertTrue(report["cancelled"])
        self.assertIn("excluded", report)


class TrainAccuracyTests(unittest.TestCase):
    def test_accuracy_on_train_is_not_distorted_by_dropout(self):
        service = NeuralService()
        service.configure([2], hidden_layers=[16], atoms=[0], mode="raw",
                          dropout=0.8, seed=7)
        data = make_data()

        report = service.train(data, epochs=25, batch_size=16, learning_rate=0.05)

        service.model.eval()
        with torch.no_grad():
            features = service._tensor(data["iq"])
            logits = service._logits(features)
            predicted = torch.stack([h.argmax(1) for h in service._split_logits(logits)], 1)
            target = torch.as_tensor(data["labels"], dtype=torch.long, device=service.device)
            clean = float((predicted == target).all(1).float().mean())

        self.assertGreaterEqual(clean, 0.98)
        # Без dropout точность обучения должна совпадать с оценкой без
        # dropout; раньше она считалась по зашумлённым логитам и была
        # заметно ниже.
        self.assertGreaterEqual(report["last"]["accuracy"], clean - 0.02)


class ProjectStoreWriteTests(unittest.TestCase):
    def test_concurrent_writes_do_not_break_the_file(self):
        store = ProjectStore(Path("/tmp"))
        target = Path("/tmp/dsh_write_test") / "project.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"value": list(range(2000))}
        errors = []

        def writer(index):
            try:
                store.write_json(target, dict(payload, writer=index))
            except Exception as exc:  # noqa: BLE001 — ошибка должна попасть в тест
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        saved = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(saved["value"], payload["value"])
        self.assertEqual(sorted(p.name for p in target.parent.iterdir()), ["project.json"])

    def test_temporary_file_is_removed(self):
        store = ProjectStore(Path("/tmp"))
        target = Path("/tmp/dsh_write_test") / "project.json"
        target.parent.mkdir(parents=True, exist_ok=True)

        store.write_json(target, {"a": 1})

        self.assertEqual(sorted(p.name for p in target.parent.iterdir()), ["project.json"])


class FakeSpin:
    def __init__(self, value=3e-3):
        self._value = float(value)

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = float(value)


class FakeSlider:
    def __init__(self, minimum=-60, maximum=-10, value=-25):
        self._minimum = minimum
        self._maximum = maximum
        self._value = value

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = int(value)


class FakeWindow:
    """Состояние окна с методами связи ползунка и поля learning rate."""

    _on_lr_slider = MainWindow._on_lr_slider
    _on_lr_spin = MainWindow._on_lr_spin
    _set_learning_rate = MainWindow._set_learning_rate

    def __init__(self):
        self._lr_updating = False
        self.lr_spin = FakeSpin()
        self.lr_slider = FakeSlider()


class LearningRateTests(unittest.TestCase):
    def make_state(self):
        return FakeWindow()

    def test_set_learning_rate_moves_slider_and_keeps_value(self):
        state = self.make_state()

        MainWindow._set_learning_rate(state, 1e-4)

        self.assertEqual(state.lr_spin.value(), 1e-4)
        self.assertEqual(state.lr_slider.value(), -40)

    def test_spin_value_is_not_rounded_to_slider_position(self):
        state = self.make_state()

        MainWindow._on_lr_spin(state, 3e-3)

        self.assertEqual(state.lr_spin.value(), 3e-3)
        self.assertEqual(state.lr_slider.value(), -25)

    def test_slider_sets_learning_rate(self):
        state = self.make_state()
        # Пользователь передвинул ползунок: значение виджета уже изменилось.
        state.lr_slider.setValue(-30)

        MainWindow._on_lr_slider(state, state.lr_slider.value())

        self.assertAlmostEqual(state.lr_spin.value(), 10 ** (-3.0), places=9)
        self.assertEqual(state.lr_slider.value(), -30)


if __name__ == "__main__":
    unittest.main()
