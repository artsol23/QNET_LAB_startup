"""Проверки открытия проекта: папка не должна запускать загрузку.

На Linux нативный GTK-диалог мог вернуть папку вместо project.json.
Фоновая задача тогда падала с IsADirectoryError, а окно оставалось
заблокированным. Теперь выбор проверяется до запуска задачи, а диалог
не использует нативный выбор файла.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# Сохраняем предусмотренный приложением порядок загрузки Qt и PyTorch.
from main import preload_system_icu

preload_system_icu()

import mainwindow
from mainwindow import MainWindow


class OpenProjectTests(unittest.TestCase):
    def make_state(self):
        calls = []

        def run(title, work, done, **kwargs):
            calls.append(("run", title))

        state = SimpleNamespace(
            _busy=False,
            project_dir=None,
            service=SimpleNamespace(device="cpu"),
            _error=lambda error: calls.append(("error", str(error))),
            _run=run,
            _project_opened=lambda result: None,
        )
        return state, calls

    def open_with(self, state, selected):
        """Вызвать обработчик с подменённым диалогом выбора файла."""
        calls = []

        class FakeDialog:
            Option = mainwindow.QFileDialog.Option

            @staticmethod
            def getOpenFileName(*args, **kwargs):
                calls.append((args, kwargs))
                return selected

        with mock.patch.object(mainwindow, "QFileDialog", FakeDialog):
            MainWindow._open_project_after_confirm(state)
        return calls

    def test_folder_does_not_start_loading(self):
        state, calls = self.make_state()
        self.open_with(state, ("/tmp", ""))
        self.assertEqual([name for name, *_ in calls], ["error"])
        self.assertIn("папка", calls[0][1].lower())
        self.assertIn("project.json", calls[0][1])

    def test_missing_file_reports_error(self):
        state, calls = self.make_state()
        self.open_with(state, ("/tmp/qnet_missing_project.json", ""))
        self.assertEqual([name for name, *_ in calls], ["error"])
        self.assertIn("не найден", calls[0][1])

    def test_file_starts_loading_with_qt_dialog(self):
        state, calls = self.make_state()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "project.json"
            path.write_text("{}", encoding="utf-8")
            dialog_calls = self.open_with(state, (str(path), ""))

        self.assertEqual([name for name, *_ in calls], ["run"])
        self.assertEqual(calls[0][1], "Открытие проекта")

        args, kwargs = dialog_calls[0]
        self.assertEqual(args[1], "Открыть проект")
        self.assertEqual(args[3], "Проект Q-Net Lab (*.json)")
        if sys.platform.startswith("linux"):
            option = mainwindow.QFileDialog.Option.DontUseNativeDialog
            self.assertTrue(kwargs["options"] & option)


if __name__ == "__main__":
    unittest.main()
