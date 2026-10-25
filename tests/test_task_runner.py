"""Проверки фоновых задач без Qt и PyTorch."""

import time
import unittest
from threading import Event

from task_runner import TaskRunner


class TaskRunnerTests(unittest.TestCase):
    TIMEOUT = 5.0

    def setUp(self):
        self.runner = TaskRunner()
        self.addCleanup(self.runner.shutdown)

    def wait_message(self):
        """Дождаться сообщения с ограничением времени."""
        deadline = time.monotonic() + self.TIMEOUT

        while time.monotonic() < deadline:
            message = self.runner.take_message()

            if message is not None:
                return message

            time.sleep(0.005)

        self.fail("Рабочая задача не вернула сообщение вовремя.")

    def test_result_is_delivered_without_calling_ui_callback(self):
        """Рабочий поток возвращает результат, но не вызывает done."""
        received = []

        def done(result):
            received.append(result)

        self.runner.submit(lambda: 42, done)

        kind, callback, result, error = self.wait_message()

        self.assertEqual(kind, "done")
        self.assertIs(callback, done)
        self.assertEqual(result, 42)
        self.assertIsNone(error)

        # Обработчик интерфейса ещё не вызывался.
        self.assertEqual(received, [])

        # Имитируем обработку сообщения главным потоком.
        callback(result)
        self.assertEqual(received, [42])

    def test_epoch_messages_arrive_before_completion(self):
        """Сообщения эпох не теряются и идут перед завершением."""
        def work():
            row = {"epoch": 1, "loss": 0.5}
            self.runner.publish_epoch(row)

            # Проверяем, что publish_epoch копирует словарь.
            row["epoch"] = 999

            self.runner.publish_epoch({
                "epoch": 2,
                "loss": 0.25,
            })
            return "finished"

        self.runner.submit(work, lambda result: None)

        first = self.wait_message()
        second = self.wait_message()
        final = self.wait_message()

        self.assertEqual(first, (
            "epoch",
            {"epoch": 1, "loss": 0.5},
        ))
        self.assertEqual(second, (
            "epoch",
            {"epoch": 2, "loss": 0.25},
        ))
        self.assertEqual(final[0], "done")
        self.assertEqual(final[2], "finished")
        self.assertIsNone(final[3])

    def test_failure_does_not_block_next_task(self):
        """После ошибки можно выполнить следующую операцию."""
        expected_error = ValueError("Повреждённый файл")

        def failing_work():
            raise expected_error

        self.runner.submit(
            failing_work,
            lambda result: None,
        )

        message = self.wait_message()

        self.assertEqual(message[0], "done")
        self.assertIsNone(message[2])
        self.assertIs(message[3], expected_error)

        self.runner.submit(
            lambda: "next task",
            lambda result: None,
        )

        next_message = self.wait_message()

        self.assertEqual(next_message[2], "next task")
        self.assertIsNone(next_message[3])

    def test_cancellation_is_reset_for_next_task(self):
        """Отмена текущей задачи не отменяет следующую."""
        started = Event()

        def cancellable_work():
            started.set()
            return self.runner.cancel_event.wait(self.TIMEOUT)

        self.runner.submit(
            cancellable_work,
            lambda result: None,
        )

        self.assertTrue(
            started.wait(self.TIMEOUT),
            "Задача не запустилась.",
        )

        self.runner.request_cancel()

        message = self.wait_message()

        self.assertTrue(message[2])
        self.assertIsNone(message[3])

        self.runner.submit(
            lambda: self.runner.cancel_event.is_set(),
            lambda result: None,
        )

        next_message = self.wait_message()

        self.assertFalse(next_message[2])
        self.assertIsNone(next_message[3])

    def test_new_task_waits_until_completion_message_is_consumed(self):
        """Нельзя заменить задачу до обработки её результата."""
        def work():
            self.runner.publish_epoch({"epoch": 1})
            return "finished"

        self.runner.submit(work, lambda result: None)

        # Получили сообщение эпохи, но ещё не забрали done.
        self.assertEqual(self.wait_message()[0], "epoch")

        with self.assertRaises(RuntimeError):
            self.runner.submit(
                lambda: "too early",
                lambda result: None,
            )

        self.assertEqual(self.wait_message()[0], "done")

        self.runner.submit(
            lambda: "allowed",
            lambda result: None,
        )

        self.assertEqual(self.wait_message()[2], "allowed")

    def test_shutdown_is_idempotent_and_rejects_new_tasks(self):
        """Повторное завершение допустимо, новые задачи запрещены."""
        self.runner.shutdown()
        self.runner.shutdown()

        with self.assertRaises(RuntimeError):
            self.runner.submit(
                lambda: 1,
                lambda result: None,
            )


if __name__ == "__main__":
    unittest.main()
