"""Фоновые задачи без зависимости от Qt и PyTorch."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import Event
from typing import Callable


class TaskRunner:
    """Выполняет одну задачу за раз.

    submit(), take_message() и shutdown() вызываются главным потоком.
    publish_epoch() вызывается рабочим потоком.

    Обработчик done не вызывается в рабочем потоке:
    он возвращается вместе с результатом через очередь сообщений.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="qnetlab",
        )
        self._messages = Queue()
        self._cancel_event = Event()
        self._active = False
        self._closed = False

    @property
    def cancel_event(self) -> Event:
        """Сигнал для кооперативной отмены вычислений."""
        return self._cancel_event

    def submit(self, work: Callable, done: Callable) -> None:
        """Запустить work(), передав результат через очередь."""
        if self._closed:
            raise RuntimeError("Обработчик фоновых задач уже закрыт.")

        if self._active:
            raise RuntimeError(
                "Дождитесь завершения текущей операции."
            )

        self._cancel_event.clear()
        self._active = True

        def job():
            try:
                result = work()
            except Exception as exc:
                self._messages.put(("done", done, None, exc))
            else:
                self._messages.put(("done", done, result, None))

        try:
            self._executor.submit(job)
        except Exception:
            self._active = False
            raise

    def publish_epoch(self, row: dict) -> None:
        """Передать независимую копию метрик эпохи в главный поток."""
        self._messages.put(("epoch", dict(row)))

    def take_message(self) -> tuple | None:
        """Забрать одно сообщение без ожидания."""
        try:
            message = self._messages.get_nowait()
        except Empty:
            return None

        if message[0] == "done":
            self._active = False

        return message

    def request_cancel(self) -> None:
        """Запросить остановку на ближайшей проверке cancel_event."""
        self._cancel_event.set()

    def shutdown(self) -> None:
        """Запретить новые задачи и запросить остановку текущей."""
        if self._closed:
            return

        self._closed = True
        self.request_cancel()
        self._executor.shutdown(
            wait=False,
            cancel_futures=True,
        )
