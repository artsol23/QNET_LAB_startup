"""Проверки экспорта без Qt и PyTorch."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from result_export import write_predictions_csv, write_report_json


class ResultExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def make_result():
        """Два измерения, два канала с номерами 2 и 5."""
        return {
            "atoms": [2, 5],
            "shot_ids": np.array([101, 205]),
            "predictions": np.array([
                [0, 1],
                [1, 0],
            ]),
            "labels": np.array([
                [0, 1],
                [0, 0],
            ]),
            "probabilities": [
                np.array([
                    [0.9, 0.1],
                    [0.2, 0.8],
                ]),
                np.array([
                    [0.1, 0.8, 0.1],
                    [0.7, 0.2, 0.1],
                ]),
            ],
        }

    @staticmethod
    def read_csv(path):
        with path.open(
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            return reader.fieldnames, rows

    def test_predictions_keep_channel_numbers_and_shot_ids(self):
        path = self.root / "predictions.csv"

        write_predictions_csv(
            path,
            result=self.make_result(),
            register_states=np.array([1, 3]),
        )

        fields, rows = self.read_csv(path)

        self.assertEqual(fields, [
            "shot_id",
            "register_state",
            "label_2",
            "prediction_2",
            "p_2_0",
            "p_2_1",
            "label_5",
            "prediction_5",
            "p_5_0",
            "p_5_1",
            "p_5_2",
        ])

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["shot_id"], "101")
        self.assertEqual(rows[1]["shot_id"], "205")
        self.assertEqual(rows[1]["register_state"], "3")

        # Во втором измерении предсказание канала 2
        # отличается от исходной метки.
        self.assertEqual(rows[1]["label_2"], "0")
        self.assertEqual(rows[1]["prediction_2"], "1")

        self.assertAlmostEqual(
            float(rows[0]["p_5_1"]),
            0.8,
        )

    def test_unlabeled_predictions_have_no_label_columns(self):
        result = self.make_result()
        del result["labels"]

        path = self.root / "unlabeled.csv"

        write_predictions_csv(
            path,
            result=result,
            register_states=np.array([1, 3]),
        )

        fields, rows = self.read_csv(path)

        self.assertFalse(
            any(name.startswith("label_") for name in fields)
        )
        self.assertIn("prediction_2", fields)
        self.assertIn("prediction_5", fields)
        self.assertEqual(len(rows), 2)

    def test_invalid_shapes_do_not_overwrite_existing_file(self):
        path = self.root / "predictions.csv"
        path.write_text("previous content", encoding="utf-8")

        result = self.make_result()
        result["probabilities"][0] = np.array([[0.9, 0.1]])

        with self.assertRaises(ValueError):
            write_predictions_csv(
                path,
                result=result,
                register_states=np.array([1, 3]),
            )

        self.assertEqual(
            path.read_text(encoding="utf-8"),
            "previous content",
        )

    def test_failed_replace_preserves_old_file_and_removes_temp(self):
        path = self.root / "predictions.csv"
        path.write_text("previous content", encoding="utf-8")

        with patch.object(
            Path,
            "replace",
            side_effect=OSError("Запись недоступна"),
        ):
            with self.assertRaises(OSError):
                write_predictions_csv(
                    path,
                    result=self.make_result(),
                    register_states=np.array([1, 3]),
                )

        self.assertEqual(
            path.read_text(encoding="utf-8"),
            "previous content",
        )
        self.assertFalse(
            path.with_name(path.name + ".tmp").exists()
        )

    def test_report_contains_metadata_and_numpy_metrics(self):
        path = self.root / "report.json"

        result = {
            "accuracy": np.float64(0.95),
            "assignment_fidelity": 0.96,
            "macro_f1": 0.94,
            "per_atom": [
                {
                    "atom": np.int64(2),
                    "confusion": np.array([
                        [9, 1],
                        [0, 10],
                    ]),
                }
            ],
            "warnings": ["Проверка"],
        }

        metadata = {
            "source": "отложенный test",
            "weights": "weights/model.pt",
            "model": {"fitted": True},
            "data": {"train": {"sha256": "example"}},
        }

        write_report_json(
            path,
            result=result,
            metadata=metadata,
        )

        report = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(report["source"], "отложенный test")
        self.assertEqual(report["weights"], "weights/model.pt")
        self.assertAlmostEqual(
            report["metrics"]["accuracy"],
            0.95,
        )
        self.assertEqual(
            report["per_atom"][0]["confusion"],
            [[9, 1], [0, 10]],
        )
        self.assertEqual(report["warnings"], ["Проверка"])
        self.assertIsNone(report["metrics"]["xeb_linear"])
        self.assertIn("notes", report)


if __name__ == "__main__":
    unittest.main()
