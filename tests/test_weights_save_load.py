"""Проверка сохранения и загрузки весов без создания окна.

Проверяются две ошибки интерфейса:

* кнопка «Сохранить веса» не должна запускать сохранение дважды:
  фоновое сохранение и синхронный экспорт в главном потоке писали
  один и тот же файл весов и один и тот же project.json, а запись
  в реестр терялась;
* загрузка весов должна ставить в очередь одну задачу: второй вызов
  _run() поднимал RuntimeError «Дождитесь завершения текущей
  операции.», который показывался пользователю как ошибка.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

# Сохраняем предусмотренный приложением порядок загрузки Qt и PyTorch.
from main import preload_system_icu

preload_system_icu()

from mainwindow import MainWindow


class SaveWeightsTests(unittest.TestCase):
    def test_save_weights_asks_for_name_and_saves_once(self):
        calls = []

        class FakeService:
            is_fitted = True

        state = SimpleNamespace(
            _busy=False,
            service=FakeService(),
            _require_project=lambda: None,
            _guard=lambda function: function(),
            _ask_weights_path=lambda: Path("/tmp/project/weights/model.pt"),
            _save_project_async=lambda **kwargs: calls.append(("async", kwargs)),
            _export_weights_now=lambda *args: calls.append(("export_now", ())),
            _save_project_now=lambda *args: calls.append(("save_now", ())),
            _error=lambda message: calls.append(("error", message)),
        )

        MainWindow.save_weights(state)

        self.assertEqual([call[0] for call in calls], ["async"])
        self.assertEqual(calls[0][1], {
            "force_weights": True,
            "weights_path": Path("/tmp/project/weights/model.pt"),
        })

    def test_save_weights_without_name_does_nothing(self):
        calls = []

        class FakeService:
            is_fitted = True

        state = SimpleNamespace(
            _busy=False,
            service=FakeService(),
            _require_project=lambda: None,
            _guard=lambda function: function(),
            _ask_weights_path=lambda: None,
            _save_project_async=lambda **kwargs: calls.append(("async", kwargs)),
            _error=lambda message: calls.append(("error", message)),
        )

        MainWindow.save_weights(state)

        self.assertEqual(calls, [])

    def test_save_weights_reports_busy_state(self):
        calls = []

        state = SimpleNamespace(
            _busy=True,
            _save_project_async=lambda **kwargs: calls.append(("async", kwargs)),
            _error=lambda message: calls.append(("error", message)),
        )

        MainWindow.save_weights(state)

        self.assertEqual([call[0] for call in calls], ["error"])
        self.assertIn("Дождитесь завершения", calls[0][1])


class LoadWeightsTests(unittest.TestCase):
    def make_state(self):
        started = []

        class FakeService:
            def load_weights(self, path):
                return {"metadata": {}, "path": path}

        state = SimpleNamespace(
            _busy=False,
            service=FakeService(),
            _run=lambda title, work, done: started.append((title, work, done)),
            _weights_loaded=lambda payload: None,
        )
        return state, started

    def test_load_weights_starts_single_task(self):
        state, started = self.make_state()

        MainWindow._load_weights_after_confirm(state, Path("/tmp/model.pt"), False)

        self.assertEqual(len(started), 1)
        self.assertEqual(started[0][0], "Загрузка весов")

    def test_load_weights_task_returns_loaded_file(self):
        state, started = self.make_state()

        MainWindow._load_weights_after_confirm(state, Path("/tmp/model.pt"), False)

        target, info = started[0][1]()
        self.assertEqual(target, Path("/tmp/model.pt"))
        self.assertEqual(info["path"], Path("/tmp/model.pt"))


if __name__ == "__main__":
    unittest.main()
