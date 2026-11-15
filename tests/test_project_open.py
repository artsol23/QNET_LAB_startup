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
            PROJECT_FORMAT="qnetlab-project",
            PROJECT_VERSION=2,
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

    def test_open_returns_manifest_name(self):
        state, calls = self.make_state()

        def run(title, work, done, **kwargs):
            calls.append(("run", work))

        state._run = run
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "my_project.json"
            path.write_text('{"format": "qnetlab-project", "version": 2}', encoding="utf-8")
            self.open_with(state, (str(path), ""))
            restored = calls[0][1]()

        self.assertEqual(restored["manifest_name"], "my_project.json")
        self.assertEqual(restored["folder"], Path(folder).resolve())

    def test_save_writes_opened_manifest(self):
        written = []
        state = SimpleNamespace(
            project_dir=Path("/tmp/qnet_project"),
            manifest_name="my_project.json",
            service=SimpleNamespace(is_fitted=False),
            dirty=False,
            current_weights=None,
            _require_project=lambda: None,
            _manifest_now=lambda: {"format": "qnetlab-project", "version": 2},
            _write_json=lambda path, payload: written.append(path),
            _set_dirty=lambda value: None,
        )
        state._manifest_path = lambda: state.project_dir / state.manifest_name

        MainWindow._save_project_now(state)

        self.assertEqual([path.name for path in written], ["my_project.json"])

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
