# -*- coding: utf-8 -*-
"""
main.py — точка входа Q-Net Lab. В PyCharm запускайте именно этот файл
(правый клик → Run 'main'). Собственных классов здесь нет: файл только
связывает три класса проекта.

Порядок запуска
===============
1. Создаётся QApplication и показывается заставка: картинка, нарисованная
   средствами Qt (отдельный файл не нужен), и полоса завершённости загрузки.
2. :class:`license_manager.LicenseManager` проверяет RSA-подпись и срок
   license.json. Полоса показывает реальные этапы проверки.
3. Если лицензия недействительна — сообщение с причиной и возможностью
   выбрать другой файл лицензии. Без действительной лицензии главное окно
   НЕ создаётся, тяжёлые библиотеки (PyTorch) даже не загружаются.
4. При успехе загружаются PyTorch и :class:`mainwindow.MainWindow`,
   создаётся :class:`neural_service.NeuralService`, окно показывается.

Команды издателя лицензий (без GUI)::

    python main.py --generate-keys --private-key ../issuer/private_key.pem
    python main.py --issue-license --private-key ../issuer/private_key.pem --subject "Лаборатория" --days 365
    python main.py --verify-license [--license путь/к/license.json]

Закрытый ключ храните вне папки приложения: владелец закрытого ключа может
выпускать лицензии.
"""
from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

from license_manager import LicenseManager

#: Папка приложения: от неё ищутся public_key.pem и license.json, а не от
#: текущего рабочего каталога (в PyCharm он может быть другим).
APP_DIR = Path(__file__).resolve().parent


def preload_system_icu() -> None:
    """Загрузить системную ICU Windows до импорта PyQt6 (нужно для Python из Anaconda).

    Qt 6.10+ на Windows использует системную библиотеку ICU
    ``C:\\Windows\\System32\\icuuc.dll`` с неверсионными именами функций
    (``ucnv_open`` …). Python из Anaconda добавляет в поиск DLL свою папку
    ``Library\\bin``, где лежит ДРУГАЯ ``icuuc.dll`` (ICU conda, имена с номером
    версии: ``ucnv_open_78``). Qt находит её раньше системной, и импорт падает
    с «DLL load failed … Не найдена указанная процедура». Если заранее загрузить
    системную библиотеку по полному пути, Windows использует уже загруженный
    модуль ``icuuc.dll``. На других системах и Python функция ничего не делает.
    """
    if sys.platform != "win32":
        return
    import ctypes
    import os
    system_icu = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "icuuc.dll"
    if system_icu.is_file():
        try:
            ctypes.WinDLL(str(system_icu))
        except OSError:
            pass  # старые Windows без системной ICU: Qt тогда использует свою


def parse_arguments(argv=None) -> argparse.Namespace:
    """Разобрать параметры командной строки. Обычному пользователю они не нужны."""
    parser = argparse.ArgumentParser(description="Q-Net Lab — распознавание состояний кубитов и кудитов")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--generate-keys", action="store_true", help="создать пару RSA-ключей издателя")
    group.add_argument("--issue-license", action="store_true", help="выпустить подписанную лицензию")
    group.add_argument("--verify-license", action="store_true", help="проверить лицензию без запуска GUI")
    parser.add_argument("--license", type=Path, default=APP_DIR / "license.json", help="путь к license.json")
    parser.add_argument("--private-key", type=Path, help="закрытый ключ издателя (только для издателя)")
    parser.add_argument("--public-key", type=Path, default=APP_DIR / "public_key.pem",
                        help="куда записать открытый ключ при --generate-keys")
    parser.add_argument("--output", type=Path, default=APP_DIR / "license.json", help="файл новой лицензии")
    parser.add_argument("--subject", help="владелец лицензии")
    parser.add_argument("--days", type=int, default=365, help="срок действия лицензии, дней")
    parser.add_argument("--overwrite", action="store_true", help="разрешить перезапись license.json")
    args = parser.parse_args(argv)
    if (args.generate_keys or args.issue_license) and args.private_key is None:
        parser.error("укажите --private-key")
    if args.issue_license and not args.subject:
        parser.error("укажите --subject")
    return args


def make_splash(app):
    """Создать заставку: рисунок (IQ-трасса, «пятна» и узлы FNN) и полосу загрузки.

    Рисунок строится QPainter-ом, поэтому для запуска не нужны файлы картинок.
    Возвращает (splash, progress_bar, label).
    """
    from PyQt6.QtCore import QPointF, QRectF, Qt
    from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QPixmap, QRadialGradient
    from PyQt6.QtWidgets import QLabel, QProgressBar, QSplashScreen

    width, height = 760, 420
    picture = QPixmap(width, height)
    painter = QPainter(picture)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    background = QLinearGradient(0, 0, width, height)
    background.setColorAt(0, QColor("#0d1f33"))
    background.setColorAt(1, QColor("#16485a"))
    painter.fillRect(picture.rect(), background)
    # Три размытых «пятна» IQ — облака состояний |0⟩, |1⟩, |2⟩.
    for (cx, cy, color) in ((560, 120, "#4fc3f7"), (640, 210, "#ffb74d"), (545, 255, "#e57373")):
        glow = QRadialGradient(QPointF(cx, cy), 46)
        c = QColor(color)
        c.setAlpha(170)
        glow.setColorAt(0, c)
        c.setAlpha(0)
        glow.setColorAt(1, c)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(QRectF(cx - 46, cy - 46, 92, 92))
    # Затухающая осциллирующая трасса считывания (I-квадратура).
    painter.setPen(QPen(QColor("#7fe0d0"), 2.2))
    previous = None
    for i in range(380):
        point = QPointF(44 + i, 250 + 34 * math.exp(-i / 210) * math.sin(i / 11))
        if previous is not None:
            painter.drawLine(previous, point)
        previous = point
    # Узлы маленькой FNN.
    layers = (5, 4, 3)
    for li, count in enumerate(layers):
        for ni in range(count):
            x, y = 470 + li * 95, 300 + (ni - (count - 1) / 2) * 18
            if li + 1 < len(layers):
                nxt = layers[li + 1]
                painter.setPen(QPen(QColor(127, 224, 208, 60), 1))
                for nj in range(nxt):
                    painter.drawLine(QPointF(x, y), QPointF(x + 95, 300 + (nj - (nxt - 1) / 2) * 18))
            painter.setPen(QPen(QColor("#a6f0e6"), 1.2))
            painter.setBrush(QColor("#1f6f7c"))
            painter.drawEllipse(QPointF(x, y), 5, 5)
    painter.setPen(QColor("#ffffff"))
    painter.setFont(QFont("Segoe UI", 34, QFont.Weight.Bold))
    painter.drawText(44, 110, "Q-Net Lab")
    painter.setFont(QFont("Segoe UI", 12))
    painter.setPen(QColor("#b7d8e4"))
    painter.drawText(46, 145, "Нейросетевое распознавание состояний кубитов и кудитов")
    painter.drawText(46, 167, "при дисперсионном считывании")
    painter.end()

    splash = QSplashScreen(picture)
    splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
    bar = QProgressBar(splash)
    bar.setGeometry(44, 352, width - 88, 14)
    bar.setRange(0, 100)
    bar.setTextVisible(False)
    bar.setStyleSheet("QProgressBar{background:#0b2133;border:0;border-radius:7px;}"
                      "QProgressBar::chunk{background:#7fe0d0;border-radius:7px;}")
    label = QLabel("Запуск…", splash)
    label.setGeometry(44, 372, width - 88, 24)
    label.setStyleSheet("color:#d4e7ef;font-size:12px;background:transparent;")
    splash.show()
    app.processEvents()
    return splash, bar, label


def run_gui(license_path: Path) -> int:
    """Показать заставку, проверить лицензию и (при успехе) открыть главное окно."""
    preload_system_icu()
    try:
        from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
    except ImportError as exc:
        print("Не удалось загрузить PyQt6: " + str(exc) + "\n"
              "Установите зависимости: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Q-Net Lab")
    splash, bar, label = make_splash(app)

    def progress(percent: int, text: str) -> None:
        """Обновить полосу загрузки; processEvents сразу перерисовывает заставку."""
        bar.setValue(percent)
        label.setText(text)
        app.processEvents()

    manager = LicenseManager(APP_DIR / "public_key.pem")
    while True:
        result = manager.verify(license_path, progress=progress)
        if result["valid"]:
            break
        splash.hide()
        box = QMessageBox(QMessageBox.Icon.Warning, "Q-Net Lab — лицензия", result["message"])
        box.setInformativeText("Для работы нужна лицензия, подписанная издателем приложения.")
        choose = box.addButton("Выбрать файл лицензии…", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Выход", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is not choose:
            return 1
        selected, _ = QFileDialog.getOpenFileName(None, "Файл лицензии", str(APP_DIR), "Лицензия (*.json)")
        if not selected:
            return 1
        license_path = Path(selected)
        splash.show()

    try:
        progress(85, "Загрузка PyTorch и нейросетевого сервиса…")
        # Тяжёлые импорты — только после успешной проверки лицензии.
        # ПОРЯДОК ВАЖЕН: сначала все модули Qt (mainwindow → matplotlib → QtSvg),
        # потом PyTorch. В venv на Python из Anaconda torch при импорте добавляет
        # в поиск DLL папку Library\bin базовой Anaconda, где лежит ДРУГАЯ сборка
        # Qt 6; загруженный после этого QtSvg подхватил бы чужой Qt6Svg.dll и упал
        # с «Не найдена указанная процедура». Уже загруженные модули Qt не меняются.
        from PyQt6 import QtSvg  # noqa: F401 — нужен matplotlib; загружаем до torch
        from mainwindow import MainWindow
        from neural_service import NeuralService
        progress(95, "Создание главного окна…")
        window = MainWindow(NeuralService(), license_info=result)
        progress(100, "Готово")
        window.show()
        splash.finish(window)
        return app.exec()
    except Exception as exc:  # noqa: BLE001 — показать пользователю любую ошибку запуска
        splash.hide()
        QMessageBox.critical(None, "Q-Net Lab — ошибка запуска",
                             f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=3)}\n"
                             "Проверьте зависимости: python -m pip install -r requirements.txt")
        return 1


def main(argv=None) -> int:
    """Точка входа: команды издателя или обычный запуск GUI."""
    args = parse_arguments(argv)
    try:
        if args.generate_keys:
            LicenseManager.generate_keys(args.private_key, args.public_key)
            print(f"Открытый ключ: {args.public_key.resolve()}\nЗакрытый ключ: {args.private_key.resolve()}\n"
                  "Закрытый ключ не включайте в поставку приложения!")
            return 0
        if args.issue_license:
            payload = LicenseManager.issue(args.private_key, args.output, args.subject, args.days, args.overwrite)
            print(f"Лицензия записана: {args.output.resolve()}\nДействует до {payload['expires_at']}")
            return 0
        if args.verify_license:
            result = LicenseManager(APP_DIR / "public_key.pem").verify(args.license)
            print(result["message"])
            return 0 if result["valid"] else 1
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return run_gui(args.license)


if __name__ == "__main__":
    raise SystemExit(main())
