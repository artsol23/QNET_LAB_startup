"""Проверки: оценок, считаемых по трассам, в результатах больше нет.

Набор данных может быть и до демодуляции, поэтому переходы между
уровнями в early/late трети, наблюдаемая доля ранних возбуждений, SNR
пятен и доля трасс-событий убраны. Ошибки распада и возбуждения по
матрице ошибок и оценка эффективной температуры остаются.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from types import SimpleNamespace

import numpy as np
from PyQt6.QtWidgets import QApplication

from neural_service import NeuralService

app = QApplication.instance() or QApplication([])

from mainwindow import MainWindow  # noqa: E402


def make_data(samples=120, trace=8):
    rng = np.random.default_rng(0)
    labels = np.array([[index % 2] for index in range(samples)], dtype=np.int64)
    iq = rng.normal(0, 0.05, (samples, 1, trace)).astype(np.complex64)
    iq[labels[:, 0] == 0, 0, :] += 2.0
    iq[labels[:, 0] == 1, 0, :] -= 2.0
    return {"iq": iq, "labels": labels}


def make_service():
    service = NeuralService()
    service.configure([2], [8], atoms=[0], mode="matched", seed=1)
    service.train(make_data(), epochs=2, batch_size=16, learning_rate=0.02)
    return service


class ValidateResultTests(unittest.TestCase):
    def test_no_trace_based_estimates(self):
        service = make_service()

        result = service.validate(make_data())

        for key in ("separation", "thermal_population", "thermal_corrected"):
            self.assertNotIn(key, result)
        for entry in result["per_atom"]:
            for key in ("separation", "transition_down", "transition_up",
                        "thermal_population", "thermal_corrected"):
                self.assertNotIn(key, entry)
            self.assertIn("decay_error", entry)
            self.assertIn("excitation_error", entry)

    def test_temperature_not_computed_without_frequency(self):
        service = make_service()

        result = service.validate(make_data())

        for entry in result["per_atom"]:
            self.assertNotIn("effective_temperature_mK", entry)


class TemperatureTests(unittest.TestCase):
    def test_temperature_from_confusion_matrix(self):
        cm = np.array([[90, 10], [0, 100]])
        support = cm.sum(1)

        value = NeuralService._effective_temperature(cm, support, 5.0)

        p_exc = 10 / 100
        p0 = 1 - p_exc
        h, k_b = 6.62607015e-34, 1.380649e-23
        expected = h * 5.0 * 1e9 / (k_b * np.log(p0 / p_exc)) * 1e3
        self.assertAlmostEqual(value, expected, places=6)

    def test_without_frequency(self):
        cm = np.array([[90, 10], [0, 100]])

        self.assertIsNone(NeuralService._effective_temperature(cm, cm.sum(1), None))

    def test_without_excited_population(self):
        cm = np.array([[100, 0], [0, 100]])

        self.assertIsNone(NeuralService._effective_temperature(cm, cm.sum(1), 5.0))

    def test_temperature_present_when_frequency_given(self):
        service = make_service()

        result = service.validate(make_data(), f01_ghz=5.0)

        entry = result["per_atom"][0]
        confusion, support = entry["confusion"], entry["support"]
        p_exc = float(confusion[0, 1:].sum()) / float(support[0])
        if p_exc > 0:
            self.assertIn("effective_temperature_mK", entry)
            self.assertAlmostEqual(
                entry["effective_temperature_mK"],
                NeuralService._effective_temperature(confusion, support, 5.0),
                places=6,
            )
        else:
            # Нет измерений, отнесённых выше: температура не оценивается.
            self.assertNotIn("effective_temperature_mK", entry)


class FilterDiagnosticsTests(unittest.TestCase):
    def test_no_event_fraction(self):
        service = make_service()

        diagnostics = service.filter_diagnostics(make_data())

        for entry in diagnostics.values():
            self.assertNotIn("event_fraction", entry)


class InterfaceTests(unittest.TestCase):
    def test_no_frequency_field(self):
        window = MainWindow()
        try:
            self.assertFalse(hasattr(window, "f01_edit"))
        finally:
            window._tasks.shutdown()

    def test_view_is_renamed(self):
        self.assertIn("Распад и возбуждение", MainWindow.VALIDATION_VIEWS)
        self.assertNotIn("Распад и температура", MainWindow.VALIDATION_VIEWS)


class ValidateExternalTests(unittest.TestCase):
    """Кнопка «Валидировать» должна запускать проверку."""

    def make_state(self, *, need_eval=None):
        calls = []
        errors = []

        class FakeService:
            def validate(self, data, *args, **kwargs):
                return {"accuracy": 1.0}

        def guard(function):
            try:
                return function()
            except Exception as exc:  # noqa: BLE001 — как в MainWindow._guard
                errors.append(str(exc))
                return None

        state = SimpleNamespace(
            _busy=False,
            service=FakeService(),
            eval_data={"iq": np.zeros((2, 1, 2), dtype=np.complex64)},
            _need_eval=need_eval or (lambda labels=False: None),
            _guard=guard,
            _run_evaluation=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        return state, calls, errors

    def test_validation_starts(self):
        state, calls, errors = self.make_state()

        MainWindow.validate_external(state)

        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], "Валидация")

    def test_validation_not_started_without_data(self):
        def need_eval(labels=False):
            raise ValueError("Загрузите экспериментальные данные на вкладке «Фильтры».")

        state, calls, errors = self.make_state(need_eval=need_eval)

        MainWindow.validate_external(state)

        self.assertEqual(calls, [])
        self.assertEqual(len(errors), 1)


if __name__ == "__main__":
    unittest.main()
