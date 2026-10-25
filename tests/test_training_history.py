import tempfile
import unittest
from pathlib import Path

from training_history import clean_history, read_history


class TrainingHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_normalizes_numbers_and_missing_metrics(self):
        result = clean_history([
            {
                "epoch": "2",
                "loss": "0.25",
                "accuracy": "0.9",
                "val_loss": "",
                "val_accuracy": "NaN",
            }
        ])

        self.assertEqual(result[0]["epoch"], 2)
        self.assertEqual(result[0]["loss"], 0.25)
        self.assertEqual(result[0]["accuracy"], 0.9)
        self.assertIsNone(result[0]["val_loss"])
        self.assertIsNone(result[0]["val_accuracy"])
        self.assertIsNone(result[0]["lr"])

    def test_rejects_invalid_epoch(self):
        for epoch in (0, -1, 1.5, "wrong", None, True):
            with self.subTest(epoch=epoch):
                with self.assertRaises(ValueError):
                    clean_history([{"epoch": epoch}])

    def test_reads_csv_with_bom_and_semicolon(self):
        path = self.root / "history.csv"
        path.write_text(
            "epoch;loss;val_loss\n"
            "1;0.8;\n"
            "2;0.4;0.5\n",
            encoding="utf-8-sig",
        )

        result = read_history(path)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["epoch"], 2)
        self.assertEqual(result[1]["val_loss"], 0.5)
        self.assertIsNone(result[0]["val_loss"])

    def test_reads_history_from_project_json(self):
        path = self.root / "project.json"
        path.write_text(
            '{"history": [{"epoch": 1, "loss": 0.2}]}',
            encoding="utf-8",
        )

        result = read_history(path)

        self.assertEqual(result[0]["epoch"], 1)
        self.assertEqual(result[0]["loss"], 0.2)

    def test_rejects_json_without_history(self):
        path = self.root / "other.json"
        path.write_text('{"settings": {}}', encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "history"):
            read_history(path)

    def test_rejects_csv_without_epoch_column(self):
        path = self.root / "other.csv"
        path.write_text(
            "loss,accuracy\n0.2,0.9\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "epoch"):
            read_history(path)

    def test_does_not_modify_original_rows(self):
        original = [{"epoch": "1", "loss": "0.5"}]

        clean_history(original)

        self.assertEqual(
            original,
            [{"epoch": "1", "loss": "0.5"}],
        )


if __name__ == "__main__":
    unittest.main()
