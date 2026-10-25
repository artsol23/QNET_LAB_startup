# -*- coding: utf-8 -*-
"""
mainwindow.py — главное окно Q-Net Lab. В файле ровно один класс:
:class:`MainWindow`.

Разделение ответственности
==========================
* neural_service.NeuralService — данные, фильтры, сеть, обучение,
  метрики, загрузка и сохранение весов.
* MainWindow — интерфейс, состояние текущей сессии и координация
  пользовательских действий.
* project_store.ProjectStore — пути проекта, запись JSON,
  контроль целостности и восстановление файлов данных.
* task_runner.TaskRunner — выполнение фоновых задач, очередь сообщений
  и сигнал отмены.
* plotting — построение графиков по готовым данным.

Структура папки проекта::

    <проект>/
        project.json        — манифест: данные, разбиение, реестр весов, история, настройки
        logs.txt            — системный журнал операций (ТЗ, п. 4.4)
        data/               — копии импортированных данных и исправленные метки
        weights/            — наборы весов  weights_ГГГГММДД_ЧЧММСС.pt
        logs/               — журналы обучения по эпохам  training_….csv
        reports/            — экспорт: предсказания CSV, отчёты JSON, графики PNG

Интерфейс
=========
Кнопка в левом верхнем углу переключает два режима.

**Режим «Обучение»**: слева вкладки «Обучение» (данные, разбиение,
архитектура FNN, фильтры, оптимизатор — поля ввода, переключатели и
слайдеры) и «Дообучение» (реестр сохранённых весов, загрузка, дообучение).
Справа — графики метрик обучения (loss, score/accuracy, assignment fidelity,
macro-F1, XEB, матрица ошибок). Каждый график включается флажком; кнопка
«Только этот» оставляет один выбранный.

**Режим «Валидация»**: слева вкладки «Фильтры» (загрузка экспериментальных
данных, применение фильтров модели, гистограммы выходов фильтров) и
«Распознавание» (валидация по меткам, классификация без меток, экспорт).
Справа — графики эффективности классификатора: пятна IQ, матрицы ошибок,
бенчмарки (NN против центроида/LDA/QDA), ошибки распада и «температуры»,
гистограммы и ядра фильтров, кривые IQ. Одновременно показывается не более
пяти атомов — пользователь сам отмечает, какие.

Длительные операции (загрузка, обучение, валидация) выполняются в одном
рабочем потоке. Рабочий поток не трогает виджеты: он кладёт сообщения в
очередь, а таймер главного потока их разбирает (так требует Qt). Сообщение
о завершении задачи идёт через ту же очередь, поэтому последняя эпоха
никогда не обрабатывается позже завершения.
"""
from __future__ import annotations

from app_version import APP_VERSION

import csv
import hashlib
import json
import logging
import shutil
import traceback
from datetime import datetime
from pathlib import Path
import copy

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QStackedWidget, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)

import plotting
from neural_service import NeuralService
from project_store import ProjectStore
from task_runner import TaskRunner
from training_history import HISTORY_KEYS, clean_history, read_history
from result_export import write_predictions_csv, write_report_json


class MainWindow(QMainWindow):
    """Главное окно: проект, два режима (обучение / валидация), графики."""

    PROJECT_FORMAT = "qnetlab-project"
    PROJECT_VERSION = 2
    MAX_ATOMS_SHOWN = 5

    #: Метрики обучения: ключ истории → (подпись, ключ train, ключ validation).
    TRAIN_CHARTS = {
        "loss": ("Loss (cross-entropy)", "loss", "val_loss"),
        "accuracy": ("Score / accuracy регистра", "accuracy", "val_accuracy"),
        "fidelity": ("Assignment fidelity F_nQ", None, "val_assignment_fidelity"),
        "f1": ("Macro-F1", None, "val_macro_f1"),
        "xeb": ("XEB", None, "val_xeb"),
        "confusion": ("Матрица ошибок (validation)", None, None),
    }
    HISTORY_KEYS = HISTORY_KEYS
    VALIDATION_VIEWS = ("Пятна IQ", "Матрицы ошибок", "Бенчмарки", "Распад и температура",
                        "Гистограммы фильтров", "Ядра фильтров", "Кривые IQ")

    # ==================================================================
    # Создание окна
    # ==================================================================
    def __init__(self, service: NeuralService | None = None, license_info: dict | None = None):
        """
        :param service: экземпляр сервиса нейросети (создаётся в main.py).
        :param license_info: результат проверки лицензии — для строки состояния.
        """
        super().__init__()
        self.service = service or NeuralService()
        self.license_info = license_info or {}
        # --- данные и состояние, которыми владеет окно ---
        self.project_dir: Path | None = None
        self.manifest: dict = {}
        self.train_data: dict | None = None       # обучающий набор (весь)
        self.eval_data: dict | None = None        # экспериментальный набор
        self.split: dict | None = None            # индексы train/validation/test
        self.history: list[dict] = []             # история эпох текущей модели
        self.train_eval: dict | None = None       # validate() на validation после обучения
        self.result: dict | None = None           # последний validate()/infer()
        self.result_metadata: dict | None = None
        self.result_source = ""                   # описание набора для отчётов
        self.diagnostics: dict | None = None      # filter_diagnostics()
        self.leakage: dict | None = None          # предложение find_leakage()
        self.current_weights: str | None = None   # имя файла весов в проекте
        self.model_fingerprint: str | None = None  # отпечаток train, на котором обучены текущие веса
        self.dirty = False                        # есть несохранённые изменения модели
        self._pending_log: list[tuple[int, str]] = []   # журнал до создания проекта
        self._logger: logging.Logger | None = None
        self._run_log: str | None = None          # CSV эпох текущего запуска
        # --- фоновые задачи ---
        self._tasks = TaskRunner()
        self._busy = False

        self._close_after_task = False
        self._close_confirmed = False
        self._redraw_pending = False

        self.setWindowTitle(f"Q-Net Lab {APP_VERSION} — распознавание состояний кубитов и кудитов")
        self.resize(1480, 940)
        self.setMinimumSize(1100, 720)
        self._build_ui()
        self._apply_style()
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self._set_mode(False)
        self._update_header()
        self._log(f"Запуск Q-Net Lab. {self.license_info.get('message', '')}")

        QTimer.singleShot(0, self.change_device)

    # ------------------------------------------------------------------
    # Компоновка
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 8)

        # Верхняя строка: кнопка режима (левый верхний угол) + кнопки проекта.
        top = QHBoxLayout()
        self.mode_button = QPushButton()
        self.mode_button.setObjectName("modeButton")
        self.mode_button.setCheckable(True)
        self.mode_button.setMinimumWidth(220)
        self.mode_button.setToolTip("Переключить режим: обучение ⇄ валидация")
        self.mode_button.toggled.connect(self._set_mode)
        top.addWidget(self.mode_button)
        self.project_bar = QWidget()
        bar = QHBoxLayout(self.project_bar)
        bar.setContentsMargins(8, 0, 0, 0)
        for text, slot, tip in (
            ("Новый проект", self.new_project, "Выбрать папку и создать проект"),
            ("Открыть проект", self.open_project, "Открыть project.json существующего проекта"),
            ("Сохранить проект", self.save_project, "Сохранить сессию и текущие веса"),
            ("Загрузить веса", self.load_weights_dialog, "Загрузить набор весов (.pt) в нейросеть"),
            ("Сохранить веса", self.save_weights, "Сохранить текущие веса в папку проекта"),
            ("Загрузить журнал", self.load_history_dialog, "Загрузить журнал обучения (CSV/JSON)"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bar.addWidget(button)

        top.addWidget(self.project_bar)
        top.addStretch()
        self.project_label = QLabel()
        top.addWidget(self.project_label)
        root.addLayout(top)

        self.device_bar = QWidget()
        device_layout = QHBoxLayout(self.device_bar)
        device_layout.setContentsMargins(0, 0, 0, 0)

        device_layout.addWidget(QLabel("Вычисления:"))

        self.device_combo = QComboBox()
        self.device_combo.addItem("Автоматически", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("GPU NVIDIA", "cuda:0")
        device_layout.addWidget(self.device_combo)

        self.device_button = QPushButton("Применить")
        self.device_button.clicked.connect(self.change_device)
        device_layout.addWidget(self.device_button)

        self.device_status = QLabel(
            f"Используется: {self.service.device}"
        )
        device_layout.addWidget(self.device_status)
        device_layout.addStretch()

        root.addWidget(self.device_bar)

        self._applied_device_mode = str(self.service.device)
        if self._applied_device_mode == "cuda":
            self._applied_device_mode = "cuda:0"

        self.pages = QStackedWidget()
        self.pages.addWidget(self._training_page())
        self.pages.addWidget(self._validation_page())
        root.addWidget(self.pages, 1)

        bottom = QHBoxLayout()
        self.status = QLabel("Создайте или откройте проект.")
        bottom.addWidget(self.status, 1)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(280)
        bottom.addWidget(self.progress)
        self.stop_button = QPushButton("Остановить")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._request_cancel)
        bottom.addWidget(self.stop_button)
        root.addLayout(bottom)

        help_action = QAction("Справка", self)
        help_action.triggered.connect(self._show_help)
        self.menuBar().addAction(help_action)

    def change_device(self):
        """Переключить устройство в рабочем потоке."""
        if self._busy:
            return

        requested = self.device_combo.currentData()
        service = self.service

        def work():
            try:
                actual = service.set_device(requested)
                return requested, actual, service.device_note, None
            except Exception as exc:
                return requested, str(service.device), "", exc

        self._run(
            "Выбор вычислительного устройства",
            work,
            self._device_changed,
        )

    def _device_changed(self, payload):
        """Обновить интерфейс в главном потоке."""
        requested, actual, note, error = payload

        self.device_status.setText(f"Используется: {actual}")

        if error is not None:
            index = self.device_combo.findData(
                self._applied_device_mode
            )
            self.device_combo.setCurrentIndex(max(index, 0))
            self._error(error, "Не удалось переключить устройство")
            return

        self._applied_device_mode = requested
        self._log(note)

    @staticmethod
    def _scroll(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # не обрезать поля на узком экране
        return area

    @staticmethod
    def _spin(low, high, value, step=1) -> QSpinBox:
        box = QSpinBox()
        box.setRange(low, high)
        box.setValue(value)
        box.setSingleStep(step)
        return box

    @staticmethod
    def _dspin(low, high, value, step, decimals) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setDecimals(decimals)
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    @staticmethod
    def _slider(low, high, value) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        return slider

    def _canvas_panel(self):
        """Правая часть: подпись-сводка, панель matplotlib и холст."""
        figure = Figure(figsize=(9, 6), facecolor="white")
        canvas = FigureCanvasQTAgg(figure)
        toolbar = NavigationToolbar2QT(canvas, self)
        # Штатная кнопка «Save» matplotlib сохраняет куда угодно; в приложении все
        # файлы пишутся только в проект, поэтому она перенаправлена в reports/.
        for action in toolbar.actions():
            if action.text() == "Save":
                action.triggered.disconnect()
                action.triggered.connect(self.export_figure)
        return figure, canvas, toolbar

    # ----------------------------- режим обучения ---------------------
    def _training_page(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.train_tabs = QTabWidget()
        self.train_tabs.addTab(self._scroll(self._tab_training()), "Обучение")
        self.train_tabs.addTab(self._scroll(self._tab_finetune()), "Дообучение")
        self.train_tabs.setMinimumWidth(500)
        splitter.addWidget(self.train_tabs)

        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(10, 0, 0, 0)
        toggles = QHBoxLayout()
        toggles.addWidget(QLabel("Графики:"))
        self.chart_checks: dict[str, QCheckBox] = {}
        for key, (title, *_) in self.TRAIN_CHARTS.items():
            check = QCheckBox(title.split(" (")[0])
            check.setChecked(key in ("loss", "accuracy", "fidelity", "xeb"))
            check.toggled.connect(self._schedule_redraw)
            self.chart_checks[key] = check
            toggles.addWidget(check)
        toggles.addStretch()
        layout.addLayout(toggles)
        single = QHBoxLayout()
        single.addWidget(QLabel("Оставить один:"))
        self.single_combo = QComboBox()
        for key, (title, *_) in self.TRAIN_CHARTS.items():
            self.single_combo.addItem(title, key)
        single.addWidget(self.single_combo)
        only = QPushButton("Только этот")
        only.clicked.connect(self._only_one_chart)
        single.addWidget(only)
        show_all = QPushButton("Показать все")
        show_all.clicked.connect(lambda: [c.setChecked(True) for c in self.chart_checks.values()])
        single.addWidget(show_all)
        single.addStretch()
        layout.addLayout(single)
        self.train_summary = QLabel("Метрики появятся после первой эпохи.")
        self.train_summary.setObjectName("summary")
        self.train_summary.setWordWrap(True)
        layout.addWidget(self.train_summary)
        self.train_figure, self.train_canvas, toolbar = self._canvas_panel()
        layout.addWidget(toolbar)
        layout.addWidget(self.train_canvas, 1)
        splitter.addWidget(right)
        splitter.setSizes([520, 960])
        return splitter

    def _tab_training(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # 1. Данные
        box = QGroupBox("1. Обучающие данные (CSV / NPZ)")
        form = QFormLayout(box)
        load = QPushButton("Загрузить обучающие данные…")
        load.clicked.connect(lambda: self.load_data_dialog("train"))
        form.addRow(load)
        self.copy_data_check = QCheckBox("Копировать файл данных в проект")
        self.copy_data_check.setChecked(True)
        self.copy_data_check.setToolTip("Иначе в проекте хранится путь к файлу и его SHA-256")
        form.addRow(self.copy_data_check)
        self.train_info = QLabel("Данные не загружены")
        self.train_info.setWordWrap(True)
        form.addRow(self.train_info)
        self.val_fraction = self._dspin(0.05, 0.45, 0.15, 0.05, 2)
        self.test_fraction = self._dspin(0.05, 0.6, 0.25, 0.05, 2)
        self.seed_spin = self._spin(0, 2_000_000_000, 42)
        form.addRow("Доля validation", self.val_fraction)
        form.addRow("Доля отложенного test", self.test_fraction)
        form.addRow("Seed", self.seed_spin)
        split_button = QPushButton("Разделить на train / validation / test")
        split_button.clicked.connect(self.make_split)
        form.addRow(split_button)
        self.split_info = QLabel("Разделение не выполнено")
        self.split_info.setWordWrap(True)
        form.addRow(self.split_info)
        leak_row = QHBoxLayout()
        self.leak_atom = self._spin(0, 255, 0)
        leak_row.addWidget(QLabel("Атом"))
        leak_row.addWidget(self.leak_atom)
        leak_button = QPushButton("Поиск утечки в |2⟩ (спектр. кластеризация)")
        leak_button.setToolTip("§V-A статьи: найти естественную утечку по MTV и предложить метки уровня 2")
        leak_button.clicked.connect(self.find_leakage)
        leak_row.addWidget(leak_button)
        form.addRow(leak_row)
        layout.addWidget(box)

        # 2. Архитектура
        box = QGroupBox("2. Регистр и архитектура FNN")
        form = QFormLayout(box)
        self.atoms_edit = QLineEdit("0, 1")
        self.atoms_edit.setToolTip("Номера каналов (атомов) данных, используемых для классификации")
        self.levels_edit = QLineEdit("2, 3")
        self.levels_edit.setToolTip("Число уровней каждого выбранного атома; одно число — для всех")
        form.addRow("Атомы (каналы)", self.atoms_edit)
        form.addRow("Уровни атомов", self.levels_edit)
        self.model_combo = QComboBox()
        self.model_combo.addItem("Фильтры QMF/RMF/EMF + FNN (статья)", "matched")
        self.model_combo.addItem("FNN по временным I/Q", "raw")
        form.addRow("Модель", self.model_combo)
        self.per_atom_check = QCheckBox("Отдельная сеть на каждый атом (как в статье)")
        self.per_atom_check.setChecked(True)
        form.addRow(self.per_atom_check)
        self.auto_layers = QCheckBox("Скрытые слои автоматически: ⌊P/2⌋, ⌊P/4⌋")
        self.auto_layers.setChecked(True)
        self.auto_layers.toggled.connect(self._auto_layers_toggled)
        form.addRow(self.auto_layers)
        self.layers_slider = self._slider(0, 8, 2)
        self.layers_label = QLabel("2")
        row = QHBoxLayout()
        row.addWidget(self.layers_slider)
        row.addWidget(self.layers_label)
        form.addRow("Число скрытых слоёв", row)
        self.width_slider = self._slider(2, 1024, 64)
        self.width_label = QLabel("64")
        row = QHBoxLayout()
        row.addWidget(self.width_slider)
        row.addWidget(self.width_label)
        form.addRow("Размер первого слоя", row)
        self.hidden_edit = QLineEdit("64, 32")
        self.hidden_edit.setToolTip("Точные размеры слоёв через запятую (слайдеры заполняют это поле)")
        form.addRow("Размеры слоёв", self.hidden_edit)
        self.layers_slider.valueChanged.connect(self._layers_from_sliders)
        self.width_slider.valueChanged.connect(self._layers_from_sliders)
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(list(NeuralService.ACTIVATIONS))
        form.addRow("Активация", self.activation_combo)
        self.dropout_slider = self._slider(0, 80, 0)
        self.dropout_label = QLabel("0.00")
        self.dropout_slider.valueChanged.connect(lambda v: self.dropout_label.setText(f"{v / 100:.2f}"))
        row = QHBoxLayout()
        row.addWidget(self.dropout_slider)
        row.addWidget(self.dropout_label)
        form.addRow("Dropout", row)
        layout.addWidget(box)
        self._auto_layers_toggled(True)

        # 3. Фильтры (калибруются по train при обучении)
        box = QGroupBox("3. Фильтры")
        form = QFormLayout(box)
        self.window_start = self._spin(0, 1_000_000, 0)
        self.window_stop = self._spin(0, 1_000_000, 0)
        self.window_stop.setToolTip("0 — до конца трассы")
        row = QHBoxLayout()
        row.addWidget(self.window_start)
        row.addWidget(QLabel("…"))
        row.addWidget(self.window_stop)
        form.addRow("Окно, отсчёты", row)
        self.conv_combo = QComboBox()
        for text, key in (("нет", "none"), ("прямоугольное (boxcar)", "boxcar"), ("гауссово", "gauss"),
                          ("экспоненциальное", "exp"), ("своё ядро", "custom")):
            self.conv_combo.addItem(text, key)
        form.addRow("Свёрточный фильтр", self.conv_combo)
        self.conv_slider = self._slider(1, 101, 5)
        self.conv_label = QLabel("5")
        self.conv_slider.valueChanged.connect(lambda v: self.conv_label.setText(str(v)))
        row = QHBoxLayout()
        row.addWidget(self.conv_slider)
        row.addWidget(self.conv_label)
        form.addRow("Ширина ядра", row)
        self.kernel_edit = QLineEdit()
        self.kernel_edit.setPlaceholderText("своё ядро: 0.25, 0.5, 0.25")
        form.addRow("Коэффициенты", self.kernel_edit)
        self.qmf_check = QCheckBox("QMF — пары состояний")
        self.rmf_check = QCheckBox("RMF — фильтры распада")
        self.emf_check = QCheckBox("EMF — фильтры возбуждения (заброса)")
        for check in (self.qmf_check, self.rmf_check, self.emf_check):
            check.setChecked(True)
            form.addRow(check)
        self.event_combo = QComboBox()
        self.event_combo.addItem("ближайший центроид MTV (статья)", "centroid")
        self.event_combo.addItem("ранний/поздний сегмент", "segments")
        form.addRow("Поиск трасс-событий", self.event_combo)
        self.margin_spin = self._dspin(0, 10, 0, 0.1, 2)
        self.margin_spin.setToolTip("Запас уверенности: событие, если d(свой центр) > (1+m)·d(чужой)")
        form.addRow("Запас уверенности", self.margin_spin)
        self.reg_spin = self._dspin(1e-6, 10, 1e-3, 1e-3, 6)
        form.addRow("Регуляризация MF λ", self.reg_spin)
        self.min_events_spin = self._spin(2, 1_000_000, 5)
        form.addRow("Мин. событий для RMF/EMF", self.min_events_spin)
        layout.addWidget(box)

        # 4. Оптимизатор и запуск
        box = QGroupBox("4. Оптимизатор и обучение")
        form = QFormLayout(box)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(list(NeuralService.OPTIMIZERS))
        form.addRow("Оптимизатор", self.optimizer_combo)
        self.lr_slider = self._slider(-60, -10, -25)
        self.lr_spin = self._dspin(1e-6, 0.1, 3e-3, 1e-4, 6)
        self.lr_slider.valueChanged.connect(lambda v: self.lr_spin.setValue(10 ** (v / 10)))
        row = QHBoxLayout()
        row.addWidget(self.lr_slider)
        row.addWidget(self.lr_spin)
        form.addRow("Learning rate", row)
        self.wd_spin = self._dspin(0, 1, 0, 1e-4, 6)
        form.addRow("Weight decay", self.wd_spin)
        self.momentum_spin = self._dspin(0, 0.999, 0.9, 0.05, 3)
        form.addRow("Momentum (SGD/RMSprop)", self.momentum_spin)
        self.epochs_spin = self._spin(1, 100_000, 40)
        self.batch_spin = self._spin(1, 65536, 256)
        form.addRow("Эпохи", self.epochs_spin)
        form.addRow("Размер батча", self.batch_spin)
        train = QPushButton("Создать сеть и обучить")
        train.setObjectName("primary")
        train.clicked.connect(lambda: self.start_training(False))
        form.addRow(train)
        test = QPushButton("Проверить на отложенном test")
        test.clicked.connect(self.validate_test_split)
        form.addRow(test)
        layout.addWidget(box)
        layout.addStretch()
        return page

    def _tab_finetune(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("Наборы весов проекта")
        form = QVBoxLayout(box)
        self.weights_list = QListWidget()
        self.weights_list.setMinimumHeight(220)
        form.addWidget(self.weights_list)
        row = QHBoxLayout()
        load = QPushButton("Загрузить выбранные")
        load.clicked.connect(self.load_selected_weights)
        row.addWidget(load)
        other = QPushButton("Загрузить из файла…")
        other.clicked.connect(self.load_weights_dialog)
        row.addWidget(other)
        form.addLayout(row)
        layout.addWidget(box)
        box = QGroupBox("Дообучение текущих весов")
        form = QFormLayout(box)
        self.model_info = QLabel("Модель не обучена")
        self.model_info.setWordWrap(True)
        form.addRow(self.model_info)
        self.ft_epochs = self._spin(1, 100_000, 10)
        self.ft_lr = self._dspin(1e-7, 0.1, 3e-4, 1e-4, 7)
        form.addRow("Эпохи дообучения", self.ft_epochs)
        form.addRow("Learning rate", self.ft_lr)
        note = QLabel("Используются обучающие данные и разбиение с вкладки «Обучение» и выбранный там "
                      "оптимизатор. Архитектура и фильтры заморожены: они — часть загруженной модели.")
        note.setWordWrap(True)
        form.addRow(note)
        finetune = QPushButton("Дообучить")
        finetune.setObjectName("primary")
        finetune.clicked.connect(lambda: self.start_training(True))
        form.addRow(finetune)
        layout.addWidget(box)
        layout.addStretch()
        return page

    # ----------------------------- режим валидации --------------------
    def _validation_page(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.val_tabs = QTabWidget()
        self.val_tabs.addTab(self._scroll(self._tab_filters()), "Фильтры")
        self.val_tabs.addTab(self._scroll(self._tab_recognition()), "Распознавание")
        self.val_tabs.setMinimumWidth(420)
        splitter.addWidget(self.val_tabs)

        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(10, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("График:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(self.VALIDATION_VIEWS)
        self.view_combo.currentIndexChanged.connect(self._schedule_redraw)
        row.addWidget(self.view_combo)
        self.color_combo = QComboBox()
        self.color_combo.addItem("цвет: предсказание", "prediction")
        self.color_combo.addItem("цвет: метка", "label")
        self.color_combo.currentIndexChanged.connect(self._schedule_redraw)
        row.addWidget(self.color_combo)
        self.filter_combo = QComboBox()
        self.filter_combo.setToolTip("Какой фильтр показывать на гистограммах")
        self.filter_combo.currentIndexChanged.connect(self._schedule_redraw)
        row.addWidget(self.filter_combo)
        row.addStretch()
        layout.addLayout(row)
        atoms_row = QHBoxLayout()
        atoms_row.addWidget(QLabel(f"Атомы на графиках (до {self.MAX_ATOMS_SHOWN}):"))
        self.atom_list = QListWidget()
        self.atom_list.setFlow(QListWidget.Flow.LeftToRight)
        self.atom_list.setWrapping(True)
        self.atom_list.setMaximumHeight(58)
        self.atom_list.itemChanged.connect(self._atom_checked)
        atoms_row.addWidget(self.atom_list, 1)
        layout.addLayout(atoms_row)
        self.val_summary = QLabel("Загрузите данные и выполните валидацию или классификацию.")
        self.val_summary.setObjectName("summary")
        self.val_summary.setWordWrap(True)
        layout.addWidget(self.val_summary)
        self.val_figure, self.val_canvas, toolbar = self._canvas_panel()
        layout.addWidget(toolbar)
        layout.addWidget(self.val_canvas, 1)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(140)
        layout.addWidget(self.details)
        splitter.addWidget(right)
        splitter.setSizes([440, 1040])
        return splitter

    def _tab_filters(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("1. Экспериментальные данные")
        form = QFormLayout(box)
        load = QPushButton("Загрузить экспериментальные данные…")
        load.clicked.connect(lambda: self.load_data_dialog("eval"))
        form.addRow(load)
        self.eval_info = QLabel("Данные не загружены")
        self.eval_info.setWordWrap(True)
        form.addRow(self.eval_info)
        layout.addWidget(box)
        box = QGroupBox("2. Применение фильтров модели")
        form = QFormLayout(box)
        self.filter_status = QLabel("Фильтры появятся после обучения или загрузки весов.")
        self.filter_status.setWordWrap(True)
        form.addRow(self.filter_status)
        apply = QPushButton("Применить фильтры к данным")
        apply.setObjectName("primary")
        apply.setToolTip("Окно → свёртка → QMF/RMF/EMF модели; гистограммы выходов по классам")
        apply.clicked.connect(self.apply_filters)
        form.addRow(apply)
        kernels = QPushButton("Показать ядра фильтров")
        kernels.clicked.connect(lambda: self.view_combo.setCurrentText("Ядра фильтров"))
        form.addRow(kernels)
        note = QLabel("Фильтры откалиброваны на train и являются частью модели: применяются ровно те, "
                      "с которыми сеть обучена. Новые параметры фильтров задаются в режиме «Обучение» "
                      "и требуют обучения новой сети.")
        note.setWordWrap(True)
        form.addRow(note)
        layout.addWidget(box)
        layout.addStretch()
        return page

    def _tab_recognition(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("Распознавание")
        form = QFormLayout(box)
        validate = QPushButton("Валидировать (данные с метками)")
        validate.setObjectName("primary")
        validate.clicked.connect(self.validate_external)
        form.addRow(validate)
        classify = QPushButton("Классифицировать новые данные")
        classify.clicked.connect(self.classify_external)
        form.addRow(classify)
        self.f01_edit = QLineEdit()
        self.f01_edit.setPlaceholderText("напр. 5.1, 4.9 (ГГц) — для T_eff")
        self.f01_edit.setToolTip("Частоты переходов 0→1 атомов; без них температура не оценивается")
        form.addRow("f01 атомов, ГГц", self.f01_edit)
        layout.addWidget(box)
        box = QGroupBox("Экспорт в папку проекта")
        form = QVBoxLayout(box)
        for text, slot in (("Результаты распознавания (CSV)", self.export_predictions),
                           ("Отчёт с метриками (JSON)", self.export_report),
                           ("Текущий график (PNG)", self.export_figure)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            form.addWidget(button)
        layout.addWidget(box)
        layout.addStretch()
        return page

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget { font-size: 13px; color: #17253c; }
            QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget { background: #f3f6fa; }
            QGroupBox { background: white; border: 1px solid #d9e1ec; border-radius: 8px;
                        margin-top: 12px; padding: 14px 8px 8px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QGroupBox QLabel, QGroupBox QCheckBox { font-weight: 400; }
            QPushButton { background: white; border: 1px solid #c3cfdf; border-radius: 6px; padding: 6px 10px; }
            QPushButton:hover { background: #e8f0fc; }
            QPushButton:disabled { color: #9aa6b6; background: #eef1f5; }
            QPushButton#primary { background: #1f5bb4; color: white; border-color: #1f5bb4; }
            QPushButton#modeButton { background: #0f3d6e; color: white; font-weight: 700; padding: 8px 14px; }
            QPushButton#modeButton:checked { background: #7a3e9d; }
            QLabel#summary { background: #e8f0fc; border-radius: 8px; padding: 10px; }
            QTabWidget::pane { border: 0; }
        """)

    # ==================================================================
    # Вспомогательное: журнал, ошибки, режимы
    # ==================================================================
    def _log(self, text: str, level=logging.INFO):
        """Записать в logs.txt проекта (до создания проекта — в буфер) и в строку состояния."""
        self.status.setText(text.splitlines()[0][:200] if text else "")
        if self._logger is None:
            self._pending_log.append((level, text))
            self._pending_log = self._pending_log[-500:]
        else:
            self._logger.log(level, text)

    def _error(self, error: Exception | str, title="Ошибка"):
        """Показать ошибку пользователю и записать её в журнал с traceback."""
        message = str(error) or type(error).__name__
        detail = "" if isinstance(error, str) else "".join(traceback.format_exception(error))
        self._log(f"{title}: {message}\n{detail}".strip(), logging.ERROR)
        QMessageBox.critical(self, title, message)

    def _guard(self, function):
        """Выполнить обработчик кнопки, превращая исключения в окно ошибки."""
        try:
            return function()
        except Exception as exc:  # noqa: BLE001 — ТЗ 4.4: ошибки ввода не роняют приложение
            self._error(exc)
            return None

    def _set_mode(self, validation: bool):
        """Переключить режим «Обучение» / «Валидация»."""
        self.pages.setCurrentIndex(1 if validation else 0)
        self.mode_button.setText("Режим: ВАЛИДАЦИЯ  ⇄" if validation else "Режим: ОБУЧЕНИЕ  ⇄")
        self._schedule_redraw()

    def _update_header(self):
        if self.project_dir is None:
            self.project_label.setText("Проект не выбран")
        else:
            self.project_label.setText(f"Проект: {self.project_dir.name}" + ("  •  есть несохранённые изменения" if self.dirty else ""))
            self.project_label.setToolTip(str(self.project_dir))
        info = self.service.describe()
        if info["fitted"]:
            cfg = info["config"]
            text = (f"Модель: {'фильтры+FNN' if cfg['mode'] == 'matched' else 'FNN'}; атомы {cfg['atoms']}, "
                    f"уровни {cfg['levels']}; скрытые слои {info['hidden']}; параметров {info['parameters']:,}; "
                    f"эпох {info['epochs']}; веса: {self.current_weights or 'не сохранены'}")
        else:
            text = "Модель не обучена"
        self.model_info.setText(text)
        self._update_filter_status()

    def _update_filter_status(self):
        if not self.service.is_fitted:
            self.filter_status.setText("Фильтры появятся после обучения или загрузки весов.")
            self.filter_combo.clear()
            return
        cfg = self.service.config
        fc = cfg["filter_config"]
        parts = [f"окно {fc['window'] or 'вся трасса'}", f"свёртка: {fc['convolution']}"]
        names = []
        if cfg["mode"] == "matched":
            order = {"QMF": 0, "RMF": 1, "EMF": 2}
            unique = {(order.get(d["kind"], 9), d["source"], d["target"]): NeuralService._kernel_name(d).replace(" (н/д)", "")
                      for d in self.service.filters["descriptions"]}
            names = [unique[key] for key in sorted(unique)]
            available = sum(d["available"] for d in self.service.filters["descriptions"])
            parts.append(f"фильтров {len(self.service.filters['descriptions'])} (откалибровано {available})")
        else:
            parts.append("банк MF не используется (модель FNN по I/Q)")
        self.filter_status.setText("Фильтры модели: " + "; ".join(parts) + ".")
        current = self.filter_combo.currentText()
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItems(names)
        if current in names:
            self.filter_combo.setCurrentText(current)
        self.filter_combo.blockSignals(False)

    def _set_dirty(self, value=True):
        self.dirty = value
        self._update_header()

    def _require_project(self):
        if self.project_dir is None:
            raise ValueError("Сначала создайте или откройте проект: все файлы хранятся в его папке.")

    def _unique(self, folder: str, prefix: str, suffix: str) -> Path:
        return ProjectStore(self.project_dir).unique_path(
            folder,
            prefix,
            suffix,
        )

    def _relative(self, path: Path) -> str:
        return ProjectStore(self.project_dir).relative_path(path)

    def _inside(self, relative: str) -> Path:
        return ProjectStore(self.project_dir).inside(relative)

    _sha256 = staticmethod(ProjectStore.sha256)
    _jsonable = staticmethod(ProjectStore.jsonable)

    def _write_json(self, path: Path, payload):
        ProjectStore(self.project_dir).write_json(path, payload)

    # ==================================================================
    # Фоновые задачи
    # ==================================================================
    def _set_task_ui(
        self,
        busy: bool,
        *,
        cancellable: bool = False,
        steps: int = 0,
    ):
        """Обновить элементы интерфейса при запуске/завершении задачи."""
        self._busy = busy

        panels = (
            self.train_tabs,
            self.val_tabs,
            self.project_bar,
            self.mode_button,
            self.device_bar,
        )

        for widget in panels:
            widget.setEnabled(not busy)

        self.stop_button.setEnabled(busy and cancellable)

        if busy:
            self.progress.setRange(0, steps if steps else 0)
            self.progress.setValue(0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(100)

    def _run(
        self,
        title: str,
        work,
        done,
        *,
        cancellable=False,
        steps=0,
    ):
        """Запустить вычисление; обработать результат в главном потоке."""
        if self._busy:
            raise RuntimeError(
                "Дождитесь завершения текущей операции."
            )

        self._set_task_ui(
            True,
            cancellable=cancellable,
            steps=steps,
        )
        self._log(f"{title}…")

        try:
            self._tasks.submit(work, done)
        except Exception as exc:
            # Ошибка самого запуска также должна разблокировать окно.
            self._set_task_ui(False)
            self.progress.setValue(0)
            self._error(exc)

    def _request_cancel(self):
        if not self._busy:
            return

        self._tasks.request_cancel()
        self.stop_button.setEnabled(False)
        self._log("Запрошена остановка текущей операции…")

    def _poll(self):
        """Обработать сообщения рабочего потока в главном потоке Qt."""
        redraw = False

        while True:
            message = self._tasks.take_message()

            if message is None:
                break

            kind = message[0]

            if kind == "epoch":
                self._guard(
                    lambda row=message[1]: self._on_epoch(row)
                )
                redraw = True

            elif kind == "done":
                _, done, result, error = message

                self._set_task_ui(False)

                if error is not None:
                    self._error(error)
                else:
                    self._guard(
                        lambda callback=done, value=result: callback(value)
                    )

                if self._close_after_task:
                    self._close_after_task = False
                    self.close()
                    return

                redraw = True

        if redraw or self._redraw_pending:
            self._redraw_pending = False
            self._redraw()

    def _schedule_redraw(self, *_):
        self._redraw_pending = True

    # ==================================================================
    # Проект
    # ==================================================================
    def _confirm_discard(self, on_continue):
        """Продолжить действие после проверки несохранённых изменений.

        При выборе «Сохранить» продолжение вызывается только
        после успешной фоновой записи проекта.
        """
        if self._busy:
            return

        if not self.dirty or self.project_dir is None:
            self._guard(on_continue)
            return

        answer = QMessageBox.question(
            self,
            "Несохранённые изменения",
            "Текущая модель изменена и не сохранена. "
            "Сохранить проект?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )

        if answer == QMessageBox.StandardButton.Cancel:
            return

        if answer == QMessageBox.StandardButton.Save:
            self._save_project_async(
                on_success=on_continue,
            )
            return

        if answer == QMessageBox.StandardButton.Discard:
            self._guard(on_continue)

    def _attach_logger(self, folder: Path):
        """Подключить logs.txt проекта и записать в него буфер ранних сообщений."""
        if self._logger is not None:
            for handler in list(self._logger.handlers):
                handler.close()
                self._logger.removeHandler(handler)
        logger = logging.getLogger(f"qnetlab.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(folder / "logs.txt", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        logger.addHandler(handler)
        self._logger = logger
        for level, text in self._pending_log:
            logger.log(level, "[до открытия проекта] " + text)
        self._pending_log.clear()

    def _reset_state(self):
        self.train_data = self.eval_data = self.split = None
        self.history, self.train_eval, self.result, self.diagnostics, self.leakage = [], None, None, None, None
        self.current_weights = None
        self.model_fingerprint = None
        self.service = NeuralService(device=str(self.service.device))
        self.train_info.setText("Данные не загружены")
        self.eval_info.setText("Данные не загружены")
        self.split_info.setText("Разделение не выполнено")
        self.weights_list.clear()
        self._refresh_atoms()
        self.result_metadata = None

    def new_project(self):
        """Запросить создание проекта с проверкой текущих изменений."""
        self._confirm_discard(self._new_project_after_confirm)

    def _new_project_after_confirm(self):
        """Создать проект: пользователь выбирает (или создаёт) пустую папку."""
        if self._busy:
            return
        folder = QFileDialog.getExistingDirectory(self, "Папка нового проекта (создайте пустую папку)")
        if not folder:
            return

        def create():
            path = Path(folder).resolve()
            if (path / "project.json").exists():
                raise ValueError("В этой папке уже есть проект. Используйте «Открыть проект».")
            for sub in ("data", "weights", "logs", "reports"):
                (path / sub).mkdir(parents=True, exist_ok=True)
            self._reset_state()
            self.project_dir = path
            self._attach_logger(path)
            self.manifest = {"format": self.PROJECT_FORMAT, "version": self.PROJECT_VERSION,
                             "created": datetime.now().astimezone().isoformat(timespec="seconds"),
                             "datasets": {}, "weights": [], "history": [], "split": None}
            self._save_project_now(save_weights=False)
            self._set_dirty(False)
            self._log(f"Создан проект {path}")
        self._guard(create)

    def _manifest_now(self) -> dict:
        """Собрать project.json из текущего состояния окна."""
        manifest = dict(self.manifest)
        manifest.update(format=self.PROJECT_FORMAT, version=self.PROJECT_VERSION,
                        saved=datetime.now().astimezone().isoformat(timespec="seconds"),
                        current_weights=self.current_weights, history=self.history,
                        split=self.split and {k: (v if isinstance(v, (int, float, str)) else np.asarray(v).tolist())
                                              for k, v in self.split.items()},
                        settings=self._form_settings())
        return manifest

    def _save_project_now(self, save_weights=True):
        """Сохранить сессию: при несохранённой модели — новый набор весов, затем project.json."""
        self._require_project()
        if save_weights and self.service.is_fitted and (self.dirty or not self.current_weights):
            self._export_weights_now(kind="session")
        self.manifest = self._manifest_now()
        self._write_json(self.project_dir / "project.json", self.manifest)
        self._set_dirty(False)

    def _save_project_async(self, *, force_weights=False, on_success=None):
        """Сохранить проект в рабочем потоке.

        force_weights=True:
            всегда создать новый файл весов.

        force_weights=False:
            создать файл весов, если модель изменена
            или ещё не имеет сохранённого файла.
        """
        if self._busy:
            return

        def prepare():
            self._require_project()

            service = self.service

            if force_weights and not service.is_fitted:
                raise ValueError(
                    "Нет обученной модели для сохранения."
                )

            save_weights = (
                service.is_fitted
                and (
                    force_weights
                    or self.dirty
                    or not self.current_weights
                )
            )

            # Виджеты читаются только в главном потоке.
            # Рабочий поток получит независимую копию манифеста.
            manifest = copy.deepcopy(self._manifest_now())
            project_dir = self.project_dir
            store = ProjectStore(project_dir)

            weights_path = None
            weights_entry = None
            metadata = None

            if save_weights:
                weights_path = store.unique_path(
                    "weights",
                    "weights",
                    ".pt",
                )

                last_metrics = (
                    copy.deepcopy(self.history[-1])
                    if self.history
                    else {}
                )

                metadata = {
                    "created": datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                    "kind": "manual" if force_weights else "session",
                    "epochs": service.trained_epochs,
                    "metrics": last_metrics,
                    "train_data_sha256": (
                        self.manifest
                        .get("datasets", {})
                        .get("train", {})
                        .get("sha256")
                    ),
                    "split_seed": (
                        None
                        if self.split is None
                        else self.split.get("seed")
                    ),
                    "train_fingerprint": self.model_fingerprint,
                }

                weights_entry = store.jsonable({
                    "file": store.relative_path(weights_path),
                    **metadata,
                    "config": copy.deepcopy(service.config),
                })

            return {
                "service": service,
                "store": store,
                "project_dir": project_dir,
                "manifest": manifest,
                "weights_path": weights_path,
                "weights_entry": weights_entry,
                "metadata": metadata,
            }

        prepared = self._guard(prepare)

        if prepared is None:
            return

        def work():
            service = prepared["service"]
            store = prepared["store"]
            manifest = prepared["manifest"]
            weights_path = prepared["weights_path"]
            weights_entry = prepared["weights_entry"]

            if weights_path is not None:
                service.export_weights(
                    weights_path,
                    metadata=prepared["metadata"],
                )

                manifest.setdefault("weights", []).append(
                    weights_entry
                )
                manifest["current_weights"] = weights_entry["file"]

            # Сначала полностью записываем веса.
            # Затем обновляем манифест, который ссылается на них.
            store.write_json(
                prepared["project_dir"] / "project.json",
                manifest,
            )

            return manifest, weights_entry

        def done(payload):
            manifest, weights_entry = payload

            self.manifest = manifest
            self.current_weights = manifest.get("current_weights")

            self._refresh_weights_list()
            self._set_dirty(False)

            if weights_entry is not None:
                self._log(
                    "Проект и веса сохранены: "
                    f"{weights_entry['file']}"
                )
            else:
                self._log("Проект сохранён.")

            # Если во время сохранения пользователь закрыл окно,
            # закрытие обработает _poll(). Не запускаем другое действие.
            if (
                    on_success is not None
                    and not self._close_after_task
            ):
                on_success()

        title = (
            "Сохранение проекта и весов"
            if prepared["weights_path"] is not None
            else "Сохранение проекта"
        )

        self._run(
            title,
            work,
            done,
            cancellable=False,
        )

    def save_project(self):
        """Сохранить проект и при необходимости текущие веса."""
        self._save_project_async(force_weights=False)

    def open_project(self):
        """Запросить открытие проекта с проверкой текущих изменений."""
        self._confirm_discard(self._open_project_after_confirm)

    def _open_project_after_confirm(self):
        """Открыть существующий проект (project.json) и восстановить сессию."""
        if self._busy:
            return
        selected, _ = QFileDialog.getOpenFileName(self, "Открыть проект", "", "Проект Q-Net Lab (project.json)")
        if not selected:
            return

        device_name = str(self.service.device)

        def read():
            path = Path(selected).resolve()
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("format") != self.PROJECT_FORMAT or manifest.get("version") != self.PROJECT_VERSION:
                raise ValueError("Это не проект Q-Net Lab этой версии.")
            folder = path.parent
            restored = {"folder": folder, "manifest": manifest, "service": NeuralService(device=device_name), "data": {}, "warnings": []}
            for role, entry in manifest.get("datasets", {}).items():
                try:
                    restored["data"][role] = (
                        ProjectStore(folder).read_dataset(
                            entry,
                            restored["warnings"],
                            restored["service"].load_data,
                        )
                    )
                except (ValueError, OSError) as exc:
                    restored["warnings"].append(f"Набор данных «{role}» не загружен: {exc}")
            weights = manifest.get("current_weights")
            if weights:
                file = ProjectStore(folder).inside(weights)
                if folder not in file.parents:
                    raise ValueError("Путь к весам выходит за пределы проекта.")
                try:
                    restored["weights_info"] = restored["service"].load_weights(file)
                except ValueError as exc:
                    restored["warnings"].append(f"Текущие веса {weights} не загружены: {exc} "
                                                "Проект открыт без модели; выберите веса на вкладке «Дообучение».")
                    manifest["current_weights"] = None
            split = manifest.get("split")
            train = restored["data"].get("train")
            if split and train is not None:
                parts = [np.asarray(split.get(k, []), np.int64) for k in ("train", "validation", "test")]
                joined = np.concatenate(parts)
                if manifest.get("datasets", {}).get("train", {}).get("sha_mismatch"):
                    manifest["split"] = None
                elif len(np.unique(joined)) != len(joined) or (len(joined) and (joined.min() < 0 or joined.max() >= len(train["iq"]))):
                    restored["warnings"].append("Разделение в project.json повреждено — выполните его заново.")
                    manifest["split"] = None
            return restored
        self._run("Открытие проекта", read, self._project_opened)

    def _project_opened(self, restored: dict):
        self._reset_state()
        self.project_dir = restored["folder"]
        self._attach_logger(self.project_dir)
        self.manifest = restored["manifest"]
        self.service = restored["service"]
        self.train_data = restored["data"].get("train")
        self.eval_data = restored["data"].get("eval")
        self.current_weights = self.manifest.get("current_weights")
        self.history = self._clean_history(self.manifest.get("history", []))
        split = self.manifest.get("split")
        if split and self.train_data is not None:
            self.split = {k: (np.asarray(v, np.int64) if k in ("train", "validation", "test") else v)
                          for k, v in split.items()}
            self.split_info.setText(self._split_text())
        self._apply_settings(self.manifest.get("settings", {}))
        if self.service.is_fitted:
            self._form_from_config(self.service.config)
            meta = (restored.get("weights_info") or {}).get("metadata") or {}
            self.model_fingerprint = meta.get("train_fingerprint")
        self.train_info.setText(self._describe_data(self.train_data, "train"))
        self.eval_info.setText(self._describe_data(self.eval_data, "eval"))
        self._refresh_weights_list()
        self._refresh_atoms()
        self._set_dirty(False)
        for warning in restored["warnings"]:
            QMessageBox.warning(self, "Проверка данных проекта", warning)
            self._log(warning, logging.WARNING)
        self._log(f"Открыт проект {self.project_dir}")

    # ------------------------------------------------------------------
    # Данные
    # ------------------------------------------------------------------
    def load_data_dialog(self, role: str):
        """Загрузить обучающие (role='train') или экспериментальные ('eval') данные."""
        if self._busy:
            return
        if not self._guard(lambda: self._require_project() or True):
            return
        selected, _ = QFileDialog.getOpenFileName(self, "Кривые квадратур I и Q", "",
                                                  "Данные (*.npz *.csv);;NumPy (*.npz);;CSV (*.csv)")
        if not selected:
            return
        copy_file = self.copy_data_check.isChecked()

        def work():
            source = Path(selected).resolve()
            data = self.service.load_data(source)      # проверка схемы до копирования
            digest = self._sha256(source)
            if copy_file:
                target = self._unique("data", f"{role}_{source.stem}", source.suffix.lower())
                shutil.copy2(source, target)
                entry = {"file": self._relative(target), "copied": True}
            else:
                entry = {"file": str(source), "copied": False}
            entry.update(sha256=digest, size=source.stat().st_size, original=str(source),
                         imported=datetime.now().astimezone().isoformat(timespec="seconds"))
            return role, data, entry
        self._run("Загрузка данных", work, self._data_loaded)

    def _data_loaded(self, payload):
        role, data, entry = payload
        self.manifest.setdefault("datasets", {})[role] = entry
        if role == "train":
            self.train_data, self.split, self.leakage = data, None, None
            self.split_info.setText("Новые данные: выполните разделение.")
            self.train_info.setText(self._describe_data(data, "train"))
            n_atoms = data["iq"].shape[1]
            if data["labels"] is not None and not self.service.is_fitted:
                levels = np.maximum(2, data["labels"].max(axis=0) + 1)
                self.atoms_edit.setText(", ".join(map(str, range(n_atoms))))
                self.levels_edit.setText(", ".join(map(str, levels)))
            self.make_split(quiet=True)
        else:
            self.eval_data, self.result, self.diagnostics = data, None, None
            self.result_metadata = None
            self.eval_info.setText(self._describe_data(data, "eval"))
        self._refresh_atoms()
        self._write_json(self.project_dir / "project.json", self._manifest_now())
        self._log(f"Загружены данные ({role}): {entry['original']} — {self._describe_data(data, role)}")

    @staticmethod
    def _describe_data(data, role) -> str:
        if data is None:
            return "Данные не загружены"
        n, a, t = data["iq"].shape
        text = f"{n} измерений × {a} атомов × {t} отсчётов"
        if data.get("time_ns") is not None:
            text += f", шаг {np.median(np.diff(data['time_ns'])):.3g} нс"
        if data["labels"] is not None:
            levels = [int(v) + 1 for v in data["labels"].max(axis=0)]
            text += f"; метки есть (уровней по атомам: {levels})"
        else:
            text += "; без меток" + (" — только классификация" if role == "eval" else "")
        return text

    def make_split(self, quiet=False):
        """Разделить обучающий набор на непересекающиеся train / validation / test.

        Разделение стратифицировано по состоянию регистра (сочетанию меток
        атомов), чтобы каждое состояние было представлено во всех частях.
        Фильтры и нормировка потом калибруются ТОЛЬКО по train.
        """
        def work():
            if self.train_data is None or self.train_data["labels"] is None:
                raise ValueError("Для разделения нужны обучающие данные с метками.")
            val, test = self.val_fraction.value(), self.test_fraction.value()
            if val + test >= 0.9:
                raise ValueError("Сумма долей validation и test должна быть меньше 0.9.")
            rng = np.random.default_rng(self.seed_spin.value())
            labels = self.train_data["labels"]
            _, group = np.unique(labels, axis=0, return_inverse=True)
            parts = {"train": [], "validation": [], "test": []}
            for g in np.unique(group):
                idx = rng.permutation(np.flatnonzero(group == g))
                n_val, n_test = int(round(len(idx) * val)), int(round(len(idx) * test))
                if len(idx) - n_val - n_test < 1:
                    n_val = n_test = 0
                parts["validation"].append(idx[:n_val])
                parts["test"].append(idx[n_val:n_val + n_test])
                parts["train"].append(idx[n_val + n_test:])
            split = {k: np.sort(np.concatenate(v)).astype(np.int64) for k, v in parts.items()}
            if not len(split["validation"]) or not len(split["test"]):
                # Состояний регистра больше, чем измерений на каждое (большой регистр):
                # стратификация по совместным меткам невозможна. Тогда — случайное
                # разбиение с проверкой, что в train есть ≥ 2 примера каждого уровня атома.
                order = rng.permutation(len(labels))
                n_val, n_test = int(round(len(order) * val)), int(round(len(order) * test))
                split = {"validation": np.sort(order[:n_val]), "test": np.sort(order[n_val:n_val + n_test]),
                         "train": np.sort(order[n_val + n_test:])}
                train_labels = labels[split["train"]]
                for atom in range(labels.shape[1]):
                    for level in np.unique(labels[:, atom]):
                        if np.count_nonzero(train_labels[:, atom] == level) < 2:
                            raise ValueError(f"Не удаётся разделить данные: в train меньше двух трасс уровня "
                                             f"{level} атома {atom}. Уменьшите доли validation/test.")
            if not len(split["validation"]) or not len(split["test"]) or len(split["train"]) < 2:
                raise ValueError("Слишком мало измерений для непустых train, validation и test.")
            self.split = split
            self.split.update(seed=self.seed_spin.value(), validation_fraction=val, test_fraction=test,
                              data_sha256=self.manifest.get("datasets", {}).get("train", {}).get("sha256"))
            self.split_info.setText(self._split_text())
            self.train_eval = None   # история текущей модели сохраняется
            if self.project_dir is not None:
                self._write_json(self.project_dir / "project.json", self._manifest_now())
            self._log(self._split_text())
        if quiet:
            try:
                work()
            except ValueError:
                pass
        else:
            self._guard(work)

    def _train_fingerprint(self) -> str | None:
        """Отпечаток обучающей части: SHA-256 файла данных + номера train + исправленные метки.

        Сохраняется вместе с весами. Перед проверкой на отложенном test он
        сравнивается с текущим разбиением: если test мог пересекаться с train
        этих весов, оценка была бы завышена.
        """
        if self.split is None or self.train_data is None:
            return None
        digest = hashlib.sha256()
        entry = self.manifest.get("datasets", {}).get("train", {})
        # Фактический SHA прочитанного файла (если файл менялся после импорта —
        # именно его содержимое), а не только SHA, записанный при импорте.
        digest.update(str(entry.get("actual_sha256") or entry.get("sha256")).encode())
        # Содержимое меток train (в т.ч. псевдометок утечки), а не только имя файла.
        digest.update(np.ascontiguousarray(self.train_data["labels"][self.split["train"]]).tobytes())
        digest.update(np.asarray(self.split["train"], np.int64).tobytes())
        return digest.hexdigest()

    def _split_text(self) -> str:
        return (f"Train {len(self.split['train'])} · validation {len(self.split['validation'])} · "
                f"test {len(self.split['test'])} (seed {self.split.get('seed')})")

    def _check_split(self):
        n = len(self.train_data["iq"])
        joined = np.concatenate([self.split[k] for k in ("train", "validation", "test")])
        if len(np.unique(joined)) != len(joined) or joined.min() < 0 or joined.max() >= n:
            raise ValueError("Разделение в проекте повреждено — выполните его заново.")

    def find_leakage(self):
        """§V-A статьи: предложить метки утечки в |2⟩ для атома и спросить пользователя."""
        if self._busy:
            return

        atom, seed = self.leak_atom.value(), self.seed_spin.value()   # виджеты читаются в главном потоке

        def work():
            if self.train_data is None or self.split is None:
                raise ValueError("Загрузите обучающие данные с метками и выполните разделение.")
            # Кластеры подбираются ТОЛЬКО по train (по подготовленным меткам, если
            # разметка уже применялась). Validation и test размечаются по найденным
            # центрам без повторного подбора — они не участвуют в предобработке.
            data = dict(self.train_data)
            if data.get("prepared_labels") is not None:
                data["labels"] = data["prepared_labels"]
            proposal = self.service.find_leakage(data, atom, indices=self.split["train"], seed=seed)
            proposal["other_indices"] = np.concatenate([self.split["validation"], self.split["test"]])
            proposal["other_labels"] = self.service.leakage_labels(data, proposal, proposal["other_indices"])
            return proposal
        self._run("Поиск утечки (спектральная кластеризация)", work, self._leakage_found)

    def _leakage_found(self, result: dict):
        self.leakage = result
        atom = result["atom"]
        self.view_combo.setCurrentText("Пятна IQ")
        self._draw_leakage(result, atom)
        text = (f"Атом {atom}: размеры кластеров {result['cluster_sizes']}; сопоставление кластер→уровень "
                f"{result['mapping']}; кандидатов утечки в |{result['leaked_level']}⟩: {result['leaked_count']}; "
                f"согласие остальных кластеров с подготовкой: {self._pct(result['agreement'])}.")
        if result["warnings"]:
            text += "\n\nПРЕДУПРЕЖДЕНИЯ: " + " ".join(result["warnings"])
        self._log(text)
        answer = QMessageBox.question(
            self, "Поиск утечки",
            text + "\n\nЭто гипотеза по кластерам MTV, а не измерение. Применить предложенные метки? "
            "Кластеры ищутся только по train; validation и test размечаются по найденным центрам кластеров "
            "без повторной кластеризации. Подготовленные метки сохраняются отдельно (для физической диагностики). Исправленные метки будут сохранены в папке проекта.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.train_data.get("prepared_labels") is None:
            self.train_data["prepared_labels"] = self.train_data["labels"]
        labels = self.train_data["labels"].copy()
        labels[result["indices"], atom] = result["labels"]
        labels[result["other_indices"], atom] = result["other_labels"]
        self.train_data["labels"] = labels
        path = self._unique("data", "labels_leakage", ".npy")
        np.save(path, labels)
        self.manifest["datasets"]["train"]["labels_override"] = self._relative(path)
        self._write_json(self.project_dir / "project.json", self._manifest_now())
        self.train_info.setText(self._describe_data(self.train_data, "train"))
        levels = [int(v) + 1 for v in labels.max(axis=0)]
        self.levels_edit.setText(", ".join(map(str, np.maximum(levels, 2))))
        self._log(f"Метки утечки применены (train, validation, test) и сохранены: {path.name}; "
                  "подготовленные метки сохранены отдельно.")

    # ------------------------------------------------------------------
    # Конфигурация из формы
    # ------------------------------------------------------------------
    def _auto_layers_toggled(self, auto: bool):
        for widget in (self.layers_slider, self.width_slider, self.hidden_edit):
            widget.setEnabled(not auto)

    def _layers_from_sliders(self):
        """Слайдеры «число слоёв» и «размер первого слоя» → размеры, каждый следующий вдвое меньше."""
        count, width = self.layers_slider.value(), self.width_slider.value()
        self.layers_label.setText(str(count))
        self.width_label.setText(str(width))
        self.hidden_edit.setText(", ".join(str(max(width >> i, 2)) for i in range(count)))

    @staticmethod
    def _ints(text: str, name: str) -> list[int]:
        try:
            return [int(v) for v in text.replace(";", ",").split(",") if v.strip()]
        except ValueError as exc:
            raise ValueError(f"{name}: введите целые числа через запятую.") from exc

    def _filter_config(self) -> dict:
        start, stop = self.window_start.value(), self.window_stop.value()
        window = None if start == 0 and stop == 0 else [start, stop if stop else self._samples()]
        kernel = []
        if self.conv_combo.currentData() == "custom":
            try:
                kernel = [float(v) for v in self.kernel_edit.text().replace(";", ",").split(",") if v.strip()]
            except ValueError as exc:
                raise ValueError("Коэффициенты ядра: числа через запятую.") from exc
        return dict(window=window, convolution=self.conv_combo.currentData(), conv_width=self.conv_slider.value(),
                    conv_kernel=kernel, qmf=self.qmf_check.isChecked(), rmf=self.rmf_check.isChecked(),
                    emf=self.emf_check.isChecked(), event_rule=self.event_combo.currentData(),
                    event_margin=self.margin_spin.value(), regularization=self.reg_spin.value(),
                    min_events=self.min_events_spin.value())

    def _samples(self) -> int:
        return 0 if self.train_data is None else self.train_data["iq"].shape[2]

    def _config_from_form(self) -> dict:
        atoms = self._ints(self.atoms_edit.text(), "Атомы")
        levels = self._ints(self.levels_edit.text(), "Уровни")
        if len(levels) == 1 and len(atoms) > 1:
            levels = levels * len(atoms)
        hidden = None if self.auto_layers.isChecked() else self._ints(self.hidden_edit.text(), "Размеры слоёв")
        return self.service.configure(levels, hidden, atoms=atoms, mode=self.model_combo.currentData(),
                                      per_atom_networks=self.per_atom_check.isChecked(),
                                      activation=self.activation_combo.currentText(),
                                      dropout=self.dropout_slider.value() / 100,
                                      filter_config=self._filter_config(), seed=self.seed_spin.value())

    def _form_from_config(self, cfg: dict):
        """Показать в форме конфигурацию загруженной модели."""
        self.atoms_edit.setText(", ".join(map(str, cfg["atoms"])))
        self.levels_edit.setText(", ".join(map(str, cfg["levels"])))
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(cfg["mode"])))
        self.per_atom_check.setChecked(cfg["per_atom_networks"])
        self.auto_layers.setChecked(cfg["hidden_layers"] is None)
        if cfg["hidden_layers"] is not None:
            self.hidden_edit.setText(", ".join(map(str, cfg["hidden_layers"])))
        self.activation_combo.setCurrentText(cfg["activation"])
        self.dropout_slider.setValue(int(round(cfg["dropout"] * 100)))
        fc = cfg["filter_config"]
        self.window_start.setValue(fc["window"][0] if fc["window"] else 0)
        self.window_stop.setValue(fc["window"][1] if fc["window"] else 0)
        self.conv_combo.setCurrentIndex(max(0, self.conv_combo.findData(fc["convolution"])))
        self.conv_slider.setValue(fc["conv_width"])
        self.kernel_edit.setText(", ".join(map(str, fc["conv_kernel"])))
        self.qmf_check.setChecked(fc["qmf"])
        self.rmf_check.setChecked(fc["rmf"])
        self.emf_check.setChecked(fc["emf"])
        self.event_combo.setCurrentIndex(max(0, self.event_combo.findData(fc["event_rule"])))
        self.margin_spin.setValue(fc["event_margin"])
        self.reg_spin.setValue(fc["regularization"])
        self.min_events_spin.setValue(fc["min_events"])

    def _form_settings(self) -> dict:
        """Параметры интерфейса, которые сохраняются в project.json."""
        return dict(atoms=self.atoms_edit.text(), levels=self.levels_edit.text(),
                    optimizer=self.optimizer_combo.currentText(), lr=self.lr_spin.value(),
                    weight_decay=self.wd_spin.value(), momentum=self.momentum_spin.value(),
                    epochs=self.epochs_spin.value(), batch=self.batch_spin.value(),
                    val_fraction=self.val_fraction.value(), test_fraction=self.test_fraction.value(),
                    seed=self.seed_spin.value(), charts=[k for k, c in self.chart_checks.items() if c.isChecked()],
                    view=self.view_combo.currentText(), f01=self.f01_edit.text(),
                    mode="validation" if self.mode_button.isChecked() else "training",
                    model=self.model_combo.currentData(), per_atom=self.per_atom_check.isChecked(),
                    auto_layers=self.auto_layers.isChecked(), hidden=self.hidden_edit.text(),
                    activation=self.activation_combo.currentText(), dropout=self.dropout_slider.value(),
                    window=[self.window_start.value(), self.window_stop.value()],
                    convolution=self.conv_combo.currentData(), conv_width=self.conv_slider.value(),
                    kernel=self.kernel_edit.text(), qmf=self.qmf_check.isChecked(), rmf=self.rmf_check.isChecked(),
                    emf=self.emf_check.isChecked(), event_rule=self.event_combo.currentData(),
                    margin=self.margin_spin.value(), regularization=self.reg_spin.value(),
                    min_events=self.min_events_spin.value(), ft_epochs=self.ft_epochs.value(), ft_lr=self.ft_lr.value(),
                    copy_data=self.copy_data_check.isChecked())

    def _apply_settings(self, s: dict):
        try:
            self.atoms_edit.setText(s.get("atoms", self.atoms_edit.text()))
            self.levels_edit.setText(s.get("levels", self.levels_edit.text()))
            self.optimizer_combo.setCurrentText(s.get("optimizer", "Adam"))
            self.lr_spin.setValue(float(s.get("lr", 1e-3)))
            self.wd_spin.setValue(float(s.get("weight_decay", 0)))
            self.momentum_spin.setValue(float(s.get("momentum", 0.9)))
            self.epochs_spin.setValue(int(s.get("epochs", 40)))
            self.batch_spin.setValue(int(s.get("batch", 256)))
            self.val_fraction.setValue(float(s.get("val_fraction", 0.15)))
            self.test_fraction.setValue(float(s.get("test_fraction", 0.25)))
            self.seed_spin.setValue(int(s.get("seed", 42)))
            charts = set(s.get("charts", self.chart_checks))
            for key, check in self.chart_checks.items():
                check.setChecked(key in charts)
            self.view_combo.setCurrentText(s.get("view", self.VALIDATION_VIEWS[0]))
            self.f01_edit.setText(s.get("f01", ""))
            if "model" in s:
                self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(s["model"])))
                self.per_atom_check.setChecked(bool(s.get("per_atom", True)))
                self.auto_layers.setChecked(bool(s.get("auto_layers", True)))
                self.hidden_edit.setText(str(s.get("hidden", "")))
                self.activation_combo.setCurrentText(s.get("activation", "relu"))
                self.dropout_slider.setValue(int(s.get("dropout", 0)))
                self.window_start.setValue(int(s.get("window", [0, 0])[0]))
                self.window_stop.setValue(int(s.get("window", [0, 0])[1]))
                self.conv_combo.setCurrentIndex(max(0, self.conv_combo.findData(s.get("convolution", "none"))))
                self.conv_slider.setValue(int(s.get("conv_width", 5)))
                self.kernel_edit.setText(str(s.get("kernel", "")))
                self.qmf_check.setChecked(bool(s.get("qmf", True)))
                self.rmf_check.setChecked(bool(s.get("rmf", True)))
                self.emf_check.setChecked(bool(s.get("emf", True)))
                self.event_combo.setCurrentIndex(max(0, self.event_combo.findData(s.get("event_rule", "centroid"))))
                self.margin_spin.setValue(float(s.get("margin", 0)))
                self.reg_spin.setValue(float(s.get("regularization", 1e-3)))
                self.min_events_spin.setValue(int(s.get("min_events", 5)))
                self.ft_epochs.setValue(int(s.get("ft_epochs", 10)))
                self.ft_lr.setValue(float(s.get("ft_lr", 3e-4)))
                self.copy_data_check.setChecked(bool(s.get("copy_data", True)))
            self.mode_button.setChecked(s.get("mode") == "validation")
        except (TypeError, ValueError):
            self._log("Часть настроек интерфейса в project.json повреждена и пропущена.", logging.WARNING)

    # ------------------------------------------------------------------
    # Обучение и дообучение
    # ------------------------------------------------------------------
    def start_training(self, fine_tune: bool):
        """Запустить обучение новой сети (fine_tune=False) или дообучение текущей."""
        if self._busy:
            return

        def prepare():
            self._require_project()
            if self.train_data is None or self.split is None:
                raise ValueError("Загрузите обучающие данные с метками и выполните разделение.")
            if fine_tune:
                if not self.service.is_fitted:
                    raise ValueError("Для дообучения загрузите веса или обучите сеть.")
                config, epochs, lr = None, self.ft_epochs.value(), self.ft_lr.value()
            else:
                config, epochs, lr = self._config_from_form(), self.epochs_spin.value(), self.lr_spin.value()
            return config, epochs, lr
        prepared = self._guard(prepare)
        if prepared is None:
            return
        config, epochs, lr = prepared
        if not fine_tune and self.service.is_fitted and self.dirty:
            answer = QMessageBox.question(self, "Новая сеть", "Текущая модель не сохранена и будет заменена новой. Продолжить?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        data, split = self.train_data, self.split
        parameters = dict(epochs=epochs, batch_size=self.batch_spin.value(), optimizer=self.optimizer_combo.currentText(),
                          learning_rate=lr, weight_decay=self.wd_spin.value(), momentum=self.momentum_spin.value())
        self._run_log = self._relative(self._unique("logs", "finetune" if fine_tune else "training", ".csv"))
        if not fine_tune:
            self.history = []
        self.train_eval = None
        self._epoch_offset = self.service.trained_epochs if fine_tune else 0
        self._training_fine_tune = fine_tune

        steps_before = self.service.update_steps
        model_id_before = id(self.service.model)   # не ссылка: старая модель не должна удерживаться в памяти

        def work():
            try:
                report = self.service.train(data, train_indices=split["train"], val_indices=split["validation"],
                                            config=config, fine_tune=fine_tune, cancel_event=self._tasks.cancel_event,
                                            callback=self._tasks.publish_epoch, **parameters)
                evaluation = None

                should_validate = (
                    self.service.is_fitted
                    and len(split["validation"]) > 0
                    and not report["cancelled"]
                    and not self._tasks.cancel_event.is_set()
                )

                if should_validate:
                    evaluation = self.service.validate(
                        data,
                        split["validation"],
                        batch_size=parameters["batch_size"] * 4,
                    )
                return report, evaluation, None
            except Exception as exc:  # noqa: BLE001 — модель могла измениться до ошибки
                changed = (id(self.service.model) != model_id_before or self.service.update_steps != steps_before)
                return None, None, (exc, changed)
        self._run("Дообучение" if fine_tune else "Обучение новой сети", work, self._training_done,
                  cancellable=True, steps=epochs)
        self._log(f"Параметры: {json.dumps(self._jsonable(parameters), ensure_ascii=False)}; "
                  f"конфигурация: {json.dumps(self._jsonable(config or self.service.config), ensure_ascii=False)}")

    def _on_epoch(self, row: dict):
        """Обработать метрики эпохи (в главном потоке): история, CSV, прогресс."""
        row = {k: row.get(k) for k in self.HISTORY_KEYS}
        self.history.append(row)
        self.progress.setValue(int(row["epoch"]) - self._epoch_offset)
        self.train_summary.setText(self._epoch_text(row))
        if self.project_dir is not None and self._run_log:
            try:
                path = self._inside(self._run_log)
                new = not path.exists()
                with path.open("a", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=self.HISTORY_KEYS)
                    if new:
                        writer.writeheader()
                    writer.writerow(row)
            except OSError as exc:
                # История остаётся в памяти и попадёт в project.json; обучение не прерывается.
                self._log(f"Не удалось записать журнал эпох {self._run_log}: {exc}", logging.WARNING)
                self._run_log = None

    def _epoch_text(self, row) -> str:
        def fmt(v, pct=False):
            return "н/д" if v is None else (f"{100 * v:.2f}%" if pct else f"{v:.4g}")
        return (f"Эпоха {row['epoch']}: loss {fmt(row['loss'])} / val {fmt(row.get('val_loss'))}; "
                f"score {fmt(row['accuracy'], True)} / val {fmt(row.get('val_accuracy'), True)}; "
                f"F_nQ {fmt(row.get('val_assignment_fidelity'), True)}; XEB {fmt(row.get('val_xeb'))}")

    def _training_done(self, payload):
        report, evaluation, failure = payload
        # Прежние результаты валидации/классификации относятся к другой модели.
        self.result, self.diagnostics = None, None
        self.result_metadata = None
        self.val_summary.setText("Модель изменена: выполните валидацию или классификацию заново.")
        self.details.clear()
        if failure is not None:
            error, changed = failure
            if changed:
                # Веса успели обновиться: модель отличается от сохранённого файла.
                self.current_weights = None
                self.model_fingerprint = "mixed"
                self._set_dirty(True)
            self._error(error, "Ошибка обучения")
            return
        self.train_eval = evaluation
        fingerprint = self._train_fingerprint()
        # Дообучение на другом train смешивает выборки: независимость test неизвестна.
        if self._training_fine_tune and self.model_fingerprint != fingerprint:
            self.model_fingerprint = "mixed"
        else:
            self.model_fingerprint = fingerprint
        self.current_weights = None
        self._set_dirty(True)
        self._refresh_atoms()
        self._form_from_config(self.service.config)
        for warning in report["warnings"]:   # сначала предупреждения, итог — последней строкой состояния
            self._log(warning, logging.WARNING)
        state = "остановлено" if report["cancelled"] else "завершено"
        self._log(f"Обучение {state}: эпох {report['epochs_completed']}, всего {report['total_epochs']}; "
                  f"последние метрики {json.dumps(self._jsonable(report['last']), ensure_ascii=False)}")
        if evaluation is not None:
            self.train_summary.setText(self.train_summary.text() + "\nValidation после обучения: "
                                       + self._metrics_text(evaluation))
        self._write_json(self.project_dir / "project.json", self._manifest_now() | {"current_weights": None})

    def _run_evaluation(
        self,
        title,
        work,
        *,
        source,
        dataset_role,
        data,
        indices=None,
        f01=None,
    ):
        """Запустить расчёт с фиксацией происхождения результата."""
        if self._busy:
            return

        def prepare():
            dataset_entry = copy.deepcopy(
                self.manifest
                .get("datasets", {})
                .get(dataset_role)
            )

            if dataset_entry is None:
                raise ValueError(
                    "В проекте отсутствует описание набора данных "
                    f"{dataset_role!r}."
                )

            if indices is None:
                selection = {
                    "kind": "all",
                    "count": len(data["iq"]),
                }
            else:
                # Явный порядок байтов делает отпечаток независимым
                # от архитектуры компьютера.
                selected = np.ascontiguousarray(
                    indices,
                    dtype="<i8",
                )

                selection = {
                    "kind": "indices",
                    "count": len(selected),
                    "sha256": hashlib.sha256(
                        selected.tobytes()
                    ).hexdigest(),
                    "encoding": "int64-little-endian",
                }

            weights_saved = (
                self.current_weights is not None
                and not self.dirty
            )

            return {
                "started": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "source": source,
                "application_version": APP_VERSION,
                "project": (
                    str(self.project_dir)
                    if self.project_dir is not None
                    else None
                ),
                "weights": (
                    self.current_weights
                    if weights_saved
                    else None
                ),
                "weights_saved": weights_saved,
                "model": copy.deepcopy(self.service.describe()),
                "model_train_fingerprint": self.model_fingerprint,
                "data": {
                    dataset_role: dataset_entry,
                },
                "dataset_role": dataset_role,
                "selection": selection,
                "evaluation_parameters": {
                    "f01_ghz": copy.deepcopy(f01),
                },
            }

        metadata = self._guard(prepare)

        if metadata is None:
            return

        def calculate():
            result = work()

            metadata["completed"] = (
                datetime.now()
                .astimezone()
                .isoformat(timespec="seconds")
            )

            return result, metadata

        def done(payload):
            result, result_metadata = payload

            self._validation_done(
                result,
                source=result_metadata["source"],
                metadata=result_metadata,
            )

        self._run(title, calculate, done)

    def validate_test_split(self):
        """Оценить модель на отложенной test-части."""
        if self._busy:
            return

        def check():
            if (
                self.train_data is None
                or self.split is None
                or not len(self.split["test"])
            ):
                raise ValueError(
                    "Нет отложенного test: загрузите данные "
                    "и выполните разделение."
                )

            if not self.service.is_fitted:
                raise ValueError(
                    "Сначала обучите сеть или загрузите веса."
                )

            self._check_split()

            return {
                "f01": self._f01(),
                "fingerprint": self._train_fingerprint(),
            }

        checked = self._guard(check)

        if checked is None:
            return

        if self.model_fingerprint != checked["fingerprint"]:
            answer = QMessageBox.question(
                self,
                "Независимость test не подтверждена",
                "Текущие веса обучены не на train текущего "
                "разбиения либо происхождение весов неизвестно. "
                "Test может пересекаться с обучающими данными "
                "этих весов, и оценка будет завышена. "
                "Всё равно проверить?",
            )

            if answer != QMessageBox.StandardButton.Yes:
                return

        service = self.service
        data = self.train_data
        indices = self.split["test"].copy()
        f01 = checked["f01"]

        self._run_evaluation(
            "Проверка на test",
            lambda: service.validate(
                data,
                indices,
                f01_ghz=f01,
            ),
            source="отложенный test",
            dataset_role="train",
            data=data,
            indices=indices,
            f01=f01,
        )

    # ------------------------------------------------------------------
    # Веса
    # ------------------------------------------------------------------
    def _export_weights_now(self, kind="manual") -> str:
        """Записать текущие веса в weights/ и добавить запись в реестр проекта."""
        self._require_project()
        if not self.service.is_fitted:
            raise ValueError("Нет обученной модели для сохранения.")
        path = self._unique("weights", "weights", ".pt")
        last = self.history[-1] if self.history else {}
        metadata = dict(created=datetime.now().astimezone().isoformat(timespec="seconds"), kind=kind,
                        epochs=self.service.trained_epochs, metrics=last,
                        train_data_sha256=self.manifest.get("datasets", {}).get("train", {}).get("sha256"),
                        split_seed=None if self.split is None else self.split.get("seed"),
                        train_fingerprint=self.model_fingerprint)
        self.service.export_weights(path, metadata=metadata)
        entry = dict(file=self._relative(path), **metadata, config=self.service.config)
        self.manifest.setdefault("weights", []).append(self._jsonable(entry))
        self.current_weights = entry["file"]
        self._refresh_weights_list()
        self._log(f"Сохранены веса {entry['file']}")
        return entry["file"]

    def save_weights(self):
        """Сохранить новый набор весов и обновить проект."""
        self._save_project_async(force_weights=True)

        def save():
            self._export_weights_now()
            self._save_project_now(save_weights=False)
        self._guard(save)

    def _refresh_weights_list(self):
        self.weights_list.clear()
        for entry in reversed(self.manifest.get("weights", [])):
            metrics = entry.get("metrics") or {}
            acc = metrics.get("val_accuracy")
            text = (f"{entry.get('created', '')[:19].replace('T', ' ')} · {entry['file']} · эпох {entry.get('epochs')}"
                    + ("" if acc is None else f" · val score {100 * acc:.2f}%"))
            item = QListWidgetItem(text)
            item.setToolTip(text)
            item.setData(Qt.ItemDataRole.UserRole, entry["file"])
            if entry["file"] == self.current_weights:
                item.setText("● " + text)
            self.weights_list.addItem(item)

    def load_selected_weights(self):
        item = self.weights_list.currentItem()
        if item is None:
            self._error("Выберите набор весов в списке.")
            return
        self._load_weights(self._inside(item.data(Qt.ItemDataRole.UserRole)), copy_into_project=False)

    def load_weights_dialog(self):
        """Кнопка «Загрузить веса»: выбрать .pt; файл извне копируется в weights/ проекта."""
        if self._busy:
            return
        if not self._guard(lambda: self._require_project() or True):
            return
        selected, _ = QFileDialog.getOpenFileName(self, "Набор весов", str(self.project_dir / "weights"),
                                                  "Веса Q-Net Lab (*.pt)")
        if selected:
            path = Path(selected).resolve()
            self._load_weights(path, copy_into_project=self.project_dir not in path.parents)

    def _load_weights(self, path: Path, copy_into_project: bool):
        """Проверить изменения перед заменой текущей модели."""
        self._confirm_discard(
            lambda: self._load_weights_after_confirm(
                path,
                copy_into_project,
            )
        )

    def _load_weights_after_confirm(
            self,
            path: Path,
            copy_into_project: bool,
    ):
        """Загрузить веса после подтверждения пользователя."""
        if self._busy:
            return

        def work():
            target = path

            if copy_into_project:
                target = self._unique(
                    "weights",
                    "imported",
                    ".pt",
                )
                shutil.copy2(path, target)

            try:
                info = self.service.load_weights(target)
            except Exception:
                if copy_into_project:
                    target.unlink(missing_ok=True)
                raise

            return target, info

        self._run(
            "Загрузка весов",
            work,
            self._weights_loaded,
        )

        def work():
            target = path
            if copy_into_project:
                target = self._unique("weights", "imported", ".pt")
                shutil.copy2(path, target)
            try:
                info = self.service.load_weights(target)
            except Exception:
                if copy_into_project:
                    target.unlink(missing_ok=True)
                raise
            return target, info
        self._run("Загрузка весов", work, self._weights_loaded)

    def _weights_loaded(self, payload):
        target, info = payload
        relative = self._relative(target)
        if not any(e["file"] == relative for e in self.manifest.get("weights", [])):
            meta = info.get("metadata") or {}
            self.manifest.setdefault("weights", []).append(self._jsonable(dict(
                file=relative, created=meta.get("created") or datetime.now().astimezone().isoformat(timespec="seconds"),
                kind="imported", epochs=info["trained_epochs"], metrics=meta.get("metrics", {}), config=info["config"])))
        self.current_weights = relative
        self.model_fingerprint = (info.get("metadata") or {}).get("train_fingerprint")
        self.history = self._clean_history([])
        self.result = self.train_eval = self.diagnostics = None
        self.result_metadata = None
        self._form_from_config(info["config"])
        self._refresh_weights_list()
        self._refresh_atoms()
        self._save_project_now(save_weights=False)
        self._log(f"Загружены веса {relative}: {info['config']['mode']}, уровни {info['config']['levels']}, "
                  f"эпох {info['trained_epochs']}")

    # ------------------------------------------------------------------
    # Журналы обучения
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_history(rows) -> list[dict]:
        return clean_history(rows)

    def load_history_dialog(self):
        """Выбрать журнал и загрузить его в рабочем потоке."""
        if self._busy:
            return

        if not self._guard(lambda: self._require_project() or True):
            return

        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Журнал обучения",
            str(self.project_dir / "logs"),
            "Журнал (*.csv *.json)",
        )

        if not selected:
            return

        path = Path(selected).resolve()
        project_dir = self.project_dir
        store = ProjectStore(project_dir)

        def work():
            # Сначала проверяем файл. Некорректный журнал
            # не заменяет текущую историю и не копируется в проект.
            history = read_history(path)

            if project_dir not in path.parents:
                target = store.unique_path(
                    "logs",
                    "imported",
                    path.suffix.lower(),
                )
                shutil.copy2(path, target)

            return history

        def done(history):
            self.history = history
            self.mode_button.setChecked(False)
            self._schedule_redraw()

            self._log(
                f"Загружен журнал {path.name}: "
                f"{len(history)} эпох."
            )

        self._run(
            "Загрузка журнала обучения",
            work,
            done,
        )

    # ------------------------------------------------------------------
    # Валидация, классификация, фильтры
    # ------------------------------------------------------------------
    def _f01(self):
        text = self.f01_edit.text().strip()
        if not text:
            return None
        try:
            values = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
        except ValueError as exc:
            raise ValueError("f01: числа в ГГц через запятую.") from exc
        return values[0] if len(values) == 1 else values

    def _need_eval(self, labels=False):
        if not self.service.is_fitted:
            raise ValueError("Сначала обучите сеть или загрузите веса.")
        if self.eval_data is None:
            raise ValueError("Загрузите экспериментальные данные на вкладке «Фильтры».")
        if labels and self.eval_data["labels"] is None:
            raise ValueError("В данных нет меток. Для данных без меток используйте «Классифицировать».")

    def validate_external(self):
        """Валидация на внешних данных с метками."""
        if self._busy:
            return

        def check():
            self._need_eval(labels=True)
            return {"f01": self._f01()}

        checked = self._guard(check)

        if checked is None:
            return

        service = self.service
        data = self.eval_data
        f01 = checked["f01"]

        self._run_evaluation(
            "Валидация",
            lambda: service.validate(
                data,
                f01_ghz=f01,
            ),
            source="экспериментальные данные",
            dataset_role="eval",
            data=data,
            f01=f01,
        )

    def classify_external(self):
        """Классификация экспериментальных данных."""
        if self._busy:
            return

        if self._guard(lambda: self._need_eval() or True) is None:
            return

        service = self.service
        data = self.eval_data

        self._run_evaluation(
            "Классификация",
            lambda: service.infer(data),
            source="классификация без меток",
            dataset_role="eval",
            data=data,
        )

    def _validation_done(
        self,
        result: dict,
        *,
        source: str,
        metadata: dict,
    ):
        self.result = result
        self.result_source = source
        self.result_metadata = copy.deepcopy(metadata)

        self.mode_button.setChecked(True)
        self._refresh_atoms()

        self.val_summary.setText(
            f"{source}: " + self._metrics_text(result)
        )
        self.details.setPlainText(
            self._details_text(result)
        )

        self._log(
            f"Готово ({source}): {self._metrics_text(result)}"
        )

    def apply_filters(self):
        """Применить фильтры модели к экспериментальным данным и показать гистограммы."""
        if self._busy or self._guard(lambda: self._need_eval() or True) is None:
            return
        data, atoms = self.eval_data, self._selected_atoms()

        def done(result):
            self.diagnostics = result
            self.mode_button.setChecked(True)
            self.view_combo.setCurrentText("Гистограммы фильтров")
            lines = []
            for atom, entry in result.items():
                ef = entry.get("event_fraction")
                if ef:
                    lines.append(f"Атом {atom}: помечено как распад {self._pct(ef['relaxation'])}, "
                                 f"как возбуждение {self._pct(ef['excitation'])} (правило {ef['rule']}).")
            self.details.setPlainText("\n".join(lines) or "Фильтры применены.")
            self._log("Фильтры применены к экспериментальным данным.")
        self._run("Применение фильтров", lambda: self.service.filter_diagnostics(data, atoms=atoms or None), done)

    @staticmethod
    def _pct(value):
        return "н/д" if value is None else f"{100 * value:.2f}%"

    def _metrics_text(self, r: dict) -> str:
        if r.get("accuracy") is None:
            text = f"классифицировано {len(r['predictions'])} измерений"
            if r.get("xeb_linear") is not None:
                text += f"; linear XEB {r['xeb_linear']:.4f}"
            return text
        b = r.get("benchmarks", {})
        return (f"score (все атомы верно) {self._pct(r['accuracy'])}; F_nQ {self._pct(r['assignment_fidelity'])}; "
                f"macro-F1 {r['macro_f1']:.4f}; XEB по меткам {r['xeb_labels']:.4f}"
                + ("" if r.get("xeb_linear") is None else f"; linear XEB {r['xeb_linear']:.4f}")
                + (f" | бенчмарки на MTV: центроид {self._pct(b.get('centroid'))}, LDA {self._pct(b.get('lda'))}, "
                   f"QDA {self._pct(b.get('qda'))}" if b else ""))

    def _details_text(self, r: dict) -> str:
        lines = []
        for p in r.get("per_atom", []):
            sep = ", ".join(f"{i}|{j}: SNR {s['snr']:.2f}" for s in p.get("separation", []) if s["snr"] is not None
                            for i, j in [s["pair"]])
            t = p.get("effective_temperature_mK")
            lines.append(f"Атом {p['atom']}: точность {self._pct(p['accuracy'])}, fidelity {self._pct(p['assignment_fidelity'])}; "
                         f"ошибка распада по уровням {[self._pct(v) for v in p['decay_error']]}; "
                         f"возбуждения {[self._pct(v) for v in p['excitation_error']]}; "
                         f"наблюдаемая доля |0⟩, рано попавших в облако выше (диагностика) "
                         f"{self._pct(p.get('thermal_population'))}, за вычетом перекрытия облаков в том же окне "
                         f"{self._pct(p.get('thermal_corrected'))}"
                         + ("" if t is None else f", T_eff ≈ {t:.1f} мК (по наблюдаемой доле, без коррекции ошибок считывания)") + (f"; {sep}" if sep else ""))
        lines += [str(w) for w in r.get("warnings", [])]
        if r.get("excluded"):
            lines.append(f"Исключено измерений с уровнями выше заданных: {r['excluded']}.")
        lines.append("Ошибки распада/возбуждения и доля ранних возбуждений включают ошибки подготовки (SPAM) и "
                     "перекрытие облаков: без отдельной калибровки считывания они не разделяют термальную "
                     "населённость и ошибки назначения: это диагностика, а не оценка сверху или снизу. T_eff — только при заданной f01. "
                     "XEB по меткам — accuracy регистра с поправкой на угадывание (D·acc−1)/(D−1), а не "
                     "независимая квантовая fidelity; linear XEB считается только при ideal_probs в данных. "
                     "* На графике «Распад и температура» T_eff — по наблюдаемой доле, без коррекции ошибок считывания.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Экспорт
    # ------------------------------------------------------------------
    def export_predictions(self):
        """Экспортировать результаты распознавания в рабочем потоке."""
        if self._busy:
            return

        def prepare():
            self._require_project()

            if self.result is None:
                raise ValueError(
                    "Сначала выполните валидацию или классификацию."
                )

            return {
                "result": self.result,
                "service": self.service,
                "path": self._unique(
                    "reports",
                    "predictions",
                    ".csv",
                ),
            }

        prepared = self._guard(prepare)

        if prepared is None:
            return

        def work():
            result = prepared["result"]

            # Вычисление состояния регистра остаётся
            # ответственностью NeuralService.
            register_states = prepared["service"].register_state(
                result["predictions"]
            )

            write_predictions_csv(
                prepared["path"],
                result=result,
                register_states=register_states,
            )

            return prepared["path"]

        def done(path):
            self._log(
                "Результаты распознавания сохранены: "
                f"{self._relative(path)}"
            )

        self._run(
            "Экспорт результатов распознавания",
            work,
            done,
            cancellable=False,
        )

    def export_report(self):
        """Экспортировать результат вместе с его исходными метаданными."""
        if self._busy:
            return

        def prepare():
            self._require_project()

            if self.result is None:
                raise ValueError(
                    "Сначала выполните валидацию или классификацию."
                )

            if self.result_metadata is None:
                raise ValueError(
                    "Для результата отсутствуют сведения "
                    "о происхождении. Повторите расчёт."
                )

            metadata = copy.deepcopy(self.result_metadata)

            # created — время создания отчёта.
            # started/completed — время самого расчёта.
            metadata["created"] = (
                datetime.now()
                .astimezone()
                .isoformat(timespec="seconds")
            )

            return {
                "result": self.result,
                "metadata": metadata,
                "path": self._unique(
                    "reports",
                    "report",
                    ".json",
                ),
            }

        prepared = self._guard(prepare)

        if prepared is None:
            return

        def work():
            write_report_json(
                prepared["path"],
                result=prepared["result"],
                metadata=prepared["metadata"],
            )

            return prepared["path"]

        def done(path):
            self._log(
                f"Отчёт сохранён: {self._relative(path)}"
            )

        self._run(
            "Экспорт отчёта",
            work,
            done,
            cancellable=False,
        )

    def export_figure(self):
        """Текущий график (активного режима) в reports/figure_….png."""
        def export():
            self._require_project()
            figure = self.val_figure if self.mode_button.isChecked() else self.train_figure
            path = self._unique("reports", "figure", ".png")
            figure.savefig(path, dpi=160, bbox_inches="tight")
            self._log(f"График сохранён: {self._relative(path)}")
        self._guard(export)

    # ==================================================================
    # Атомы на графиках (не более пяти)
    # ==================================================================
    def _refresh_atoms(self):
        """Заполнить список атомов: атомы модели, иначе — каналы загруженных данных."""
        if self.service.is_fitted:
            atoms = list(self.service.config["atoms"])
        else:
            source = self.eval_data or self.train_data
            atoms = [] if source is None else list(range(source["iq"].shape[1]))
        previous = set(self._selected_atoms())
        self.atom_list.blockSignals(True)
        self.atom_list.clear()
        checked = 0
        for atom in atoms:
            item = QListWidgetItem(f"атом {atom}")
            item.setData(Qt.ItemDataRole.UserRole, atom)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            on = (atom in previous) if previous & set(atoms) else checked < self.MAX_ATOMS_SHOWN
            on = on and checked < self.MAX_ATOMS_SHOWN
            checked += on
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self.atom_list.addItem(item)
        self.atom_list.blockSignals(False)
        self._schedule_redraw()

    def _atom_checked(self, item: QListWidgetItem):
        """Не дать отметить больше пяти атомов: снимается именно последний отмеченный."""
        if item.checkState() == Qt.CheckState.Checked and len(self._selected_atoms()) > self.MAX_ATOMS_SHOWN:
            self.atom_list.blockSignals(True)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.atom_list.blockSignals(False)
            self.status.setText(f"Одновременно показывается не более {self.MAX_ATOMS_SHOWN} атомов.")
        self._schedule_redraw()

    def _selected_atoms(self) -> list[int]:
        return [self.atom_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.atom_list.count())
                if self.atom_list.item(i).checkState() == Qt.CheckState.Checked]

    # ==================================================================
    # Графики
    # ==================================================================
    def _only_one_chart(self):
        key = self.single_combo.currentData()
        for k, check in self.chart_checks.items():
            check.setChecked(k == key)

    def _redraw(self):
        try:
            if self.mode_button.isChecked():
                self._draw_validation()
            else:
                self._draw_training()
        except Exception as exc:  # noqa: BLE001 — ошибка рисования не должна ронять окно
            self._log(f"Ошибка построения графика: {exc}\n{traceback.format_exc()}", logging.ERROR)

    @staticmethod
    def _grid(count: int):
        return plotting.grid_shape(count)

    @staticmethod
    def _message(figure, canvas, text):
        plotting.draw_message(figure, text)
        canvas.draw_idle()

    def _draw_training(self):
        chart_keys = [
            key
            for key, checkbox in self.chart_checks.items()
            if checkbox.isChecked()
        ]

        plotting.draw_training(
            self.train_figure,
            history=self.history,
            evaluation=self.train_eval,
            chart_keys=chart_keys,
            chart_specs=self.TRAIN_CHARTS,
            selected_atoms=self._selected_atoms(),
        )

        self.train_canvas.draw_idle()

    def _draw_confusions(self, ax, result):
        plotting.draw_confusions(
            ax,
            result,
            self._selected_atoms(),
        )

    @staticmethod
    def _finish(fig, canvas):
        plotting.finish_figure(fig)
        canvas.draw_idle()

    @staticmethod
    def _heatmap(ax, entry, compact=False):
        plotting.draw_heatmap(
            ax,
            entry,
            compact=compact,
        )

    def _draw_validation(self):
        fig, canvas = self.val_figure, self.val_canvas
        view = self.view_combo.currentText()
        atoms = self._selected_atoms()
        if not atoms:
            self._message(fig, canvas, "Отметьте от одного до пяти атомов сверху.")
            return
        if view == "Ядра фильтров":
            return self._draw_kernels(atoms)
        if view == "Гистограммы фильтров":
            return self._draw_histograms(atoms)
        if view == "Кривые IQ":
            return self._draw_traces(atoms)
        if self.result is None:
            if view == "Пятна IQ" and self.leakage is not None:
                return self._draw_leakage(self.leakage, self.leakage["atom"])
            self._message(fig, canvas, "Нажмите «Валидировать» или «Классифицировать» на вкладке «Распознавание».")
            return
        r = self.result
        used = r["atoms"]
        atoms = [a for a in atoms if a in used]
        if not atoms:
            self._message(fig, canvas, "Выбранные атомы не используются моделью.")
            return
        if view in ("Матрицы ошибок", "Бенчмарки", "Распад и температура") and "per_atom" not in r:
            self._message(fig, canvas, "Для этого графика нужны данные с метками (валидация).")
            return
        fig.clear()
        rows, cols = self._grid(len(atoms))
        for position, channel in enumerate(atoms, 1):
            a = used.index(channel)
            ax = fig.add_subplot(rows, cols, position)
            if view == "Пятна IQ":
                self._draw_spots(ax, r, a, channel)
            elif view == "Матрицы ошибок":
                self._heatmap(ax, r["per_atom"][a], compact=len(atoms) > 2)
            elif view == "Бенчмарки":
                self._draw_benchmarks(ax, r["per_atom"][a])
            else:
                self._draw_physics(ax, r["per_atom"][a])
        self._finish(fig, canvas)

    def _draw_spots(self, ax, result, atom_index, channel):
        plotting.draw_spots(
            ax,
            result,
            atom_index,
            channel,
            color_by=self.color_combo.currentData(),
        )

    def _draw_benchmarks(self, ax, entry):
        plotting.draw_benchmarks(ax, entry)

    def _draw_physics(self, ax, entry):
        plotting.draw_physics(ax, entry)

    def _draw_kernels(self, atoms):
        """Подготовить данные модели для отображения ядер."""
        service = self.service

        if (
                not service.is_fitted
                or service.config["mode"] != "matched"
        ):
            self._message(
                self.val_figure,
                self.val_canvas,
                "Ядра есть только у модели «фильтры + FNN» "
                "после обучения или загрузки весов.",
            )
            return

        used = service.config["atoms"]
        selected_atoms = [
            atom for atom in atoms if atom in used
        ]

        filters = service.filters
        kernels = []

        for kernel, description in zip(
                filters["kernels"],
                filters["descriptions"],
        ):
            if not description["available"]:
                continue

            # description["atom"] — индекс внутри списка
            # выбранных моделью каналов, а не номер канала данных.
            channel = used[description["atom"]]

            if channel not in selected_atoms:
                continue

            kernels.append({
                "channel": channel,
                "name": NeuralService._kernel_name(description),
                "values": kernel,
            })

        filter_names = [
            self.filter_combo.itemText(index)
            for index in range(self.filter_combo.count())
        ]

        plotting.draw_kernels(
            self.val_figure,
            atoms=selected_atoms,
            kernels=kernels,
            filter_names=filter_names,
        )
        self.val_canvas.draw_idle()

    def _draw_histograms(self, atoms):
        plotting.draw_histograms(
            self.val_figure,
            atoms=atoms,
            diagnostics=self.diagnostics,
            filter_name=self.filter_combo.currentText(),
        )
        self.val_canvas.draw_idle()

    def _draw_traces(self, atoms):
        data = (
            self.eval_data
            if self.eval_data is not None
            else self.train_data
        )

        plotting.draw_traces(
            self.val_figure,
            atoms=atoms,
            data=data,
        )
        self.val_canvas.draw_idle()

    def _draw_leakage(self, result, atom):
        # Переключение режима — обязанность окна, не plotting.
        self.mode_button.setChecked(True)

        plotting.draw_leakage(
            self.val_figure,
            result=result,
            atom=atom,
        )
        self.val_canvas.draw_idle()

    # ==================================================================
    # Справка и закрытие
    # ==================================================================
    def _show_help(self):
        QMessageBox.information(self, "Q-Net Lab — как начать", (
            "1. «Новый проект» — выберите пустую папку. Все файлы (журналы, веса, отчёты) хранятся в ней.\n"
            "2. Режим «Обучение», вкладка «Обучение»: загрузите размеченные IQ (CSV/NPZ), выполните разделение.\n"
            "3. Задайте атомы и уровни, модель (фильтры+FNN или FNN), слои, фильтры, оптимизатор; обучите.\n"
            "4. Справа графики метрик; флажками можно оставить один график.\n"
            "5. «Сохранить веса» — новый файл весов с датой; вкладка «Дообучение» — загрузка и дообучение.\n"
            "6. Режим «Валидация»: загрузите эксперимент, примените фильтры, валидируйте или классифицируйте.\n"
            "7. Экспорт результатов, отчётов и графиков — в папку reports/ проекта.\n\n"
            "Форматы данных описаны в README.txt и UserGuide.html."))

    def _approve_close(self):
        """Закрыть окно после подтверждения или успешного сохранения."""
        self._close_confirmed = True
        self.close()

    def closeEvent(self, event: QCloseEvent):
        """Закрыть окно после завершения задачи и сохранения изменений."""
        if self._busy:
            self._close_after_task = True
            self._tasks.request_cancel()
            self.stop_button.setEnabled(False)

            self._log(
                "Завершение текущей операции перед закрытием…"
            )
            event.ignore()
            return

        if not self._close_confirmed and self.dirty:
            # Сначала отклоняем текущее закрытие.
            # После подтверждения _approve_close() вызовет close() снова.
            event.ignore()
            self._confirm_discard(self._approve_close)
            return

        self._timer.stop()
        self._tasks.shutdown()

        if self._logger is not None:
            self._logger.info("Окно закрыто.")

            for handler in list(self._logger.handlers):
                handler.close()
                self._logger.removeHandler(handler)

        event.accept()
