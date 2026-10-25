"""Проверка фиксации происхождения результатов."""

import copy
import unittest
from types import SimpleNamespace

import numpy as np

# Сохраняем предусмотренный приложением порядок загрузки Qt и PyTorch.
from main import preload_system_icu

preload_system_icu()

from PyQt6 import QtSvg  # noqa: F401
from mainwindow import MainWindow


class FakeService:
    def __init__(self):
        self.description = {
            "fitted": True,
            "epochs": 10,
            "config": {
                "atoms": [0],
                "levels": [2],
            },
        }

    def describe(self):
        return self.description


class ResultProvenanceTests(unittest.TestCase):
    def make_window_state(self, *, dirty=False):
        """Минимальное состояние для _run_evaluation без QWidget."""
        state = SimpleNamespace(
            _busy=False,
            service=FakeService(),
            current_weights="weights/model_001.pt",
            dirty=dirty,
            project_dir=None,
            model_fingerprint="train-fingerprint",
            manifest={
                "datasets": {
                    "eval": {
                        "file": "data/experiment.npz",
                        "sha256": "original-data-hash",
                    }
                }
            },
        )

        # В тестах исключения должны доходить до unittest.
        state._guard = lambda function: function()

        pending = {}
        completed = {}

        def run(title, work, done):
            # Имитируем постановку задачи в очередь.
            pending.update(
                title=title,
                work=work,
                done=done,
            )

        def validation_done(result, *, source, metadata):
            completed.update(
                result=result,
                source=source,
                metadata=copy.deepcopy(metadata),
            )

        state._run = run
        state._validation_done = validation_done

        return state, pending, completed

    def start_evaluation(
        self,
        state,
        *,
        indices=None,
        f01=None,
    ):
        data = {
            "iq": np.zeros((4, 1, 3), dtype=np.complex64),
        }

        MainWindow._run_evaluation(
            state,
            "Проверка",
            lambda: {"accuracy": 0.95},
            source="экспериментальные данные",
            dataset_role="eval",
            data=data,
            indices=indices,
            f01=f01,
        )

    @staticmethod
    def finish_evaluation(pending):
        payload = pending["work"]()
        pending["done"](payload)

    def test_metadata_is_independent_of_later_state_changes(self):
        state, pending, completed = self.make_window_state()
        frequencies = [5.1]

        self.start_evaluation(
            state,
            f01=frequencies,
        )

        # Меняем исходные объекты после постановки задачи.
        state.current_weights = "weights/another_model.pt"
        state.service.description["epochs"] = 999
        state.service.description["config"]["levels"][0] = 3
        state.manifest["datasets"]["eval"]["sha256"] = "changed"
        frequencies[0] = 9.9

        self.finish_evaluation(pending)

        metadata = completed["metadata"]

        self.assertEqual(
            metadata["weights"],
            "weights/model_001.pt",
        )
        self.assertEqual(metadata["model"]["epochs"], 10)
        self.assertEqual(
            metadata["model"]["config"]["levels"],
            [2],
        )
        self.assertEqual(
            metadata["data"]["eval"]["sha256"],
            "original-data-hash",
        )
        self.assertEqual(
            metadata["evaluation_parameters"]["f01_ghz"],
            [5.1],
        )

    def test_unsaved_model_has_no_weights_reference(self):
        state, pending, completed = self.make_window_state(
            dirty=True,
        )

        self.start_evaluation(state)
        self.finish_evaluation(pending)

        metadata = completed["metadata"]

        self.assertFalse(metadata["weights_saved"])
        self.assertIsNone(metadata["weights"])

    def test_all_data_selection_records_sample_count(self):
        state, pending, completed = self.make_window_state()

        self.start_evaluation(state)
        self.finish_evaluation(pending)

        self.assertEqual(
            completed["metadata"]["selection"],
            {"kind": "all", "count": 4},
        )
        self.assertIn("started", completed["metadata"])
        self.assertIn("completed", completed["metadata"])

    def test_selection_hash_is_independent_of_input_integer_type(self):
        hashes = []

        for dtype in (np.int32, np.int64):
            state, pending, completed = self.make_window_state()

            self.start_evaluation(
                state,
                indices=np.array([0, 2, 3], dtype=dtype),
            )
            self.finish_evaluation(pending)

            selection = completed["metadata"]["selection"]

            self.assertEqual(selection["count"], 3)
            self.assertEqual(
                selection["encoding"],
                "int64-little-endian",
            )
            hashes.append(selection["sha256"])

        self.assertEqual(hashes[0], hashes[1])

    def test_different_selection_has_different_hash(self):
        hashes = []

        for indices in ([0, 1], [0, 2]):
            state, pending, completed = self.make_window_state()

            self.start_evaluation(
                state,
                indices=np.array(indices),
            )
            self.finish_evaluation(pending)

            hashes.append(
                completed["metadata"]["selection"]["sha256"]
            )

        self.assertNotEqual(hashes[0], hashes[1])

    def test_failed_calculation_does_not_publish_result(self):
        state, pending, completed = self.make_window_state()

        def fail():
            raise ValueError("Ошибка вычисления")

        MainWindow._run_evaluation(
            state,
            "Проверка",
            fail,
            source="экспериментальные данные",
            dataset_role="eval",
            data={
                "iq": np.zeros((4, 1, 3), dtype=np.complex64),
            },
        )

        with self.assertRaises(ValueError):
            pending["work"]()

        # TaskRunner при исключении не вызывает done.
        self.assertEqual(completed, {})


if __name__ == "__main__":
    unittest.main()
