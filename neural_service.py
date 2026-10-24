# -*- coding: utf-8 -*-
"""
neural_service.py — нейросетевой backend Q-Net Lab. В файле ровно один класс:
:class:`NeuralService`.

Назначение
==========
Класс отвечает за всё, что связано с нейросетью для распознавания состояний
кудитов и кубитных регистров по кривым дисперсионного считывания (временные
трассы квадратур I и Q):

* чтение экспериментальных данных (CSV двух видов и NPZ) — :meth:`load_data`;
* задание архитектуры FNN «снаружи» — :meth:`configure`;
* фильтры: временное окно, свёрточный фильтр с настраиваемым ядром,
  согласованные фильтры состояний (QMF), распада (RMF) и возбуждения (EMF)
  по статье Mude et al. — :meth:`apply_filters`, :meth:`filter_diagnostics`;
* поиск естественной утечки в |2⟩ спектральной кластеризацией —
  :meth:`find_leakage`;
* обучение и дообучение с настройкой оптимизатора — :meth:`train`;
* валидация с бенчмарками — :meth:`validate`;
* классификация новых данных — :meth:`infer`;
* классификация с построением «пятен» IQ — :meth:`classify_and_plot`;
* экспорт и загрузка весов — :meth:`export_weights`, :meth:`load_weights`.

Что класс ХРАНИТ и чего НЕ хранит
==================================
Требование задания: «класс хранит только текущие параметры нейросети, ничего
никуда дополнительно не сохраняется и не копируется». Поэтому в объекте живут
только:

* ``config``      — текущая конфигурация (уровни, атомы, слои, фильтры …);
* ``model``       — текущая сеть PyTorch (одна);
* ``filters``     — откалиброванные на train ядра фильтров и центроиды;
* ``normalizer``  — среднее и масштаб признаков (часть входного слоя);
* ``optimizer_state`` — состояние оптимизатора для дообучения;
* ``trained_epochs`` и формы входа.

Датасеты, история эпох, старые наборы весов, рисунки и пути проекта в объекте
НЕ хранятся: данные передаются в методы аргументами и возвращаются вызывающему
коду (окну), а на диск класс пишет только в :meth:`export_weights` и только по
явно переданному пути. Где хранить веса, журналы и отчёты, решает MainWindow.
Обучение берёт из общего массива данных только очередной batch по индексам,
поэтому train/validation не копируются целиком.

Формат данных в памяти (словарь ``data``)
=========================================
``iq``          complex64 [N, A, T] — N измерений (shots), A каналов (атомов),
                T временных отсчётов; I = Re, Q = Im.
``labels``      int64 [N, A] или None — подготовленный уровень каждого атома.
``time_ns``     float64 [T] или None — временная сетка в наносекундах.
``shot_ids``    int64 [N] — исходные номера измерений (для экспорта).
``ideal_probs`` float64 [D] или [N, D] или отсутствует — идеальное распределение
                для linear XEB, D = prod(levels).

«Атом» здесь — физический элемент регистра (трансмон-кубит или кудит).
Атом 0 — первый канал. При записи номера состояния регистра одним числом
атом 0 — старший разряд: для levels=[2,3] номер = 3*q0 + q1.

Связь со статьёй (arXiv:2405.08982v2, Mude, Maurya, Lienhard, Tannu)
====================================================================
* MTV (mean trace value, §V-A): временное среднее трассы, одна точка IQ на
  трассу. Это и есть «пятна» на графиках.
* Спектральная кластеризация MTV для поиска естественной утечки (§V-A).
* MF: K = (μ1 − μ0) / (σ1² − σ0²) (§V-B). ОТЛИЧИЕ: при σ0 ≈ σ1 разность
  дисперсий близка к нулю и меняет знак (ядро «взрывается»), поэтому здесь
  используется регуляризованная СУММА дисперсий, отдельно для I и для Q
  (диагональное приближение LDA). Название варианта записывается в веса.
* QMF для каждой пары уровней, RMF для переходов вниз, EMF для переходов
  вверх (табл. III). Трасса с меткой s, MTV которой ближе к центроиду другого
  состояния j, считается трассой с ошибкой (§V-B): j<s — распад, j>s —
  возбуждение. Дополнительно доступна эвристика «ранний/поздний сегмент».
* Сеть: по умолчанию, как в статье, отдельная маленькая FNN на каждый атом;
  вход — выходы всех фильтров всех атомов (P признаков), скрытые слои
  ⌊P/2⌋ и ⌊P/4⌋, выход — k уровней атома (§V-C, рис. 4). Для 5 кутритов
  P = 45, сеть 45-22-11-3. Можно выбрать и общую сеть с отдельными softmax.
* Метрика F_nQ = (F1·…·Fn)^(1/n) — геометрическое среднее assignment
  fidelity атомов (табл. II, IV).
* Бенчмарки LDA/QDA на MTV (табл. V) и классификатор ближайшего центроида.

Режимы модели (ТЗ: «2 различных модели нейросетей: FNN, фильтрация+FNN»)
* ``raw``     — FNN по развёрнутым временным I и Q (базовая FNN [1] статьи);
* ``matched`` — банк фильтров QMF/RMF/EMF + FNN (предложенная статьёй схема).
"""
from __future__ import annotations

import copy
import csv
import itertools
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from scipy.cluster.vq import kmeans2
from scipy.ndimage import convolve1d
from scipy.optimize import linear_sum_assignment
from scipy.special import ndtr
from torch import nn


class NeuralService:
    """Сервис текущей нейросети: данные → фильтры → FNN → метрики.

    Все методы синхронные: длительные операции (обучение, валидация, загрузка)
    окно запускает в рабочем потоке. ``callback`` метода :meth:`train`
    вызывается из этого рабочего потока, поэтому GUI не должен трогать виджеты
    прямо в callback — только положить результат в очередь.

    Пример использования без GUI::

        service = NeuralService()
        data = service.load_data("demo_training.npz")
        service.configure(levels=[2, 3], hidden_layers=None, mode="matched")
        idx = np.random.default_rng(0).permutation(len(data["iq"]))
        service.train(data, train_indices=idx[:500], val_indices=idx[500:],
                      epochs=30, optimizer="Adam", learning_rate=3e-3)
        report = service.validate(data, indices=idx[500:])
        service.export_weights("weights.pt")
    """

    #: Версия формата файла весов. Меняется при несовместимых изменениях.
    BUNDLE_FORMAT = "QNetLab-weights"
    BUNDLE_VERSION = 2

    #: Поддерживаемые оптимизаторы и их параметры по умолчанию.
    OPTIMIZERS = ("Adam", "AdamW", "SGD", "RMSprop")

    #: Поддерживаемые функции активации скрытых слоёв.
    ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "elu": nn.ELU}

    #: Параметры фильтров по умолчанию (см. :meth:`configure`).
    DEFAULT_FILTERS = {
        "window": None,            # [start, stop) в отсчётах; None — вся трасса
        "convolution": "none",     # none | boxcar | gauss | exp | custom
        "conv_width": 5,           # ширина ядра (отсчёты) для пресетов
        "conv_kernel": [],         # коэффициенты для custom
        "qmf": True,               # фильтры пар состояний
        "rmf": True,               # фильтры распада (релаксации)
        "emf": True,               # фильтры возбуждения
        "event_rule": "centroid",  # centroid (как в статье) | segments
        "event_margin": 0.0,       # запас уверенности для пометки трассы как события
        "regularization": 1e-3,    # λ в знаменателе MF (доля среднего масштаба)
        "min_events": 5,           # минимум трасс-событий для RMF/EMF
    }

    # ------------------------------------------------------------------
    # Создание и конфигурация
    # ------------------------------------------------------------------
    def __init__(self, device: str = "cpu") -> None:
        """Создать пустой сервис.

        :param device: ``"cpu"`` (по умолчанию) или ``"cuda"``, если в системе
            есть видеокарта с CUDA и установлена соответствующая сборка PyTorch.
        :raises ValueError: если запрошена CUDA, но она недоступна.
        """
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA недоступна. Выберите устройство cpu.")
        # Текущее состояние классификатора. Ничего другого объект не хранит.
        self.config: dict[str, Any] = {}
        self.model: nn.Module | None = None
        self.filters: dict[str, Any] = {}
        self.normalizer: dict[str, np.ndarray] = {}
        self.optimizer_name: str | None = None
        self.optimizer_state: dict | None = None
        self.input_channels: int | None = None   # A в данных обучения
        self.input_samples: int | None = None    # T в данных обучения
        self.feature_count: int | None = None    # P — размер входа FNN
        self.trained_epochs = 0
        #: Счётчик шагов оптимизатора за время жизни объекта. Позволяет окну
        #: понять, изменились ли веса, даже если обучение прервалось ошибкой.
        self.update_steps = 0
        self._fitted = False
        self.configure(levels=[2], hidden_layers=None)

    @property
    def is_fitted(self) -> bool:
        """True, если в сервисе есть обученные или загруженные веса."""
        return self._fitted and self.model is not None

    def configure(self, levels, hidden_layers=None, *, atoms=None, mode="matched",
                  per_atom_networks=True, activation="relu", dropout=0.0,
                  filter_config=None, seed=42) -> dict:
        """Проверить и вернуть конфигурацию новой сети (ничего не обучая).

        Метод только ПРОВЕРЯЕТ параметры и строит словарь конфигурации.
        Если сервис ещё не обучен, конфигурация сразу становится текущей.
        Если же в сервисе есть обученная сеть, она НЕ уничтожается: новую
        конфигурацию нужно передать в :meth:`train` (``config=...``), и
        замена произойдёт только после успешной калибровки на новых данных.
        Так ошибка в данных не оставляет пользователя без прежней модели.

        :param levels: число уровней каждого используемого атома, например
            ``[2, 3]`` — кубит и кутрит. Длина списка — число атомов,
            используемых для классификации.
        :param hidden_layers: размеры скрытых слоёв, например ``[64, 32]``;
            число слоёв = длина списка. ``[]`` — линейный классификатор.
            ``None`` — автоматически, как в статье: ``[⌊P/2⌋, ⌊P/4⌋]``,
            где P — размер входа (вычисляется при обучении).
        :param atoms: номера каналов данных, которые используются (по одному
            на элемент ``levels``). ``None`` — первые ``len(levels)`` каналов.
            Позволяет, например, классифицировать только атомы 0 и 3 из 5.
        :param mode: ``"raw"`` — FNN по временным I/Q; ``"matched"`` — банк
            фильтров QMF/RMF/EMF + FNN.
        :param per_atom_networks: True — отдельная FNN на каждый атом (как в
            статье); False — одна общая FNN с отдельным softmax на атом.
        :param activation: ``relu`` | ``gelu`` | ``tanh`` | ``elu``.
        :param dropout: вероятность dropout после каждого скрытого слоя, [0, 1).
        :param filter_config: словарь параметров фильтров, ключи как в
            :attr:`DEFAULT_FILTERS`. Неизвестные ключи — ошибка.
        :param seed: зерно генераторов случайных чисел (инициализация и
            перемешивание батчей) — для воспроизводимости.
        :returns: копия проверенной конфигурации.
        :raises ValueError: при недопустимых параметрах.
        """
        levels = self._int_list(levels, "Уровни атомов", minimum=2)
        if not levels:
            raise ValueError("Укажите хотя бы один атом (список уровней не пуст).")
        if len(levels) > 256 or max(levels) > 64:
            raise ValueError("Поддерживается до 256 атомов и до 64 уровней на атом.")
        if hidden_layers is not None:
            hidden_layers = self._int_list(hidden_layers, "Скрытые слои", minimum=1)
            if len(hidden_layers) > 16 or any(v > 65536 for v in hidden_layers):
                raise ValueError("Слишком большая архитектура FNN (до 16 слоёв по 65536).")
        atoms = list(range(len(levels))) if atoms is None else self._int_list(atoms, "Номера атомов", 0)
        if len(atoms) != len(levels) or len(set(atoms)) != len(atoms):
            raise ValueError("Номера атомов должны быть различны, по одному на каждый элемент уровней.")
        mode = str(mode).lower()
        if mode not in ("raw", "matched"):
            raise ValueError("mode должен быть 'raw' (FNN) или 'matched' (фильтры + FNN).")
        activation = str(activation).lower()
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"Активация: {', '.join(self.ACTIVATIONS)}.")
        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout должен лежать в [0, 1).")
        fc = self._check_filter_config(filter_config, mode)
        seed = self._int(seed, "seed", 0)
        config = dict(levels=levels, hidden_layers=hidden_layers, atoms=atoms, mode=mode,
                      per_atom_networks=bool(per_atom_networks), activation=activation,
                      dropout=dropout, filter_config=fc, seed=seed)
        if not self.is_fitted:
            self.config = config
        return copy.deepcopy(config)

    def _check_filter_config(self, supplied, mode) -> dict:
        """Объединить параметры фильтров с умолчаниями и проверить их."""
        fc = copy.deepcopy(self.DEFAULT_FILTERS)
        supplied = dict(supplied or {})
        unknown = set(supplied) - set(fc)
        if unknown:
            raise ValueError(f"Неизвестные параметры фильтров: {sorted(unknown)}")
        fc.update(supplied)
        if fc["window"] is not None:
            window = self._int_list(fc["window"], "Окно", 0)
            if len(window) != 2 or window[1] - window[0] < 2:
                raise ValueError("Окно задаётся как [начало, конец) отсчётов, минимум 2 отсчёта.")
            fc["window"] = window
        if fc["convolution"] not in ("none", "boxcar", "gauss", "exp", "custom"):
            raise ValueError("Свёртка: none, boxcar, gauss, exp или custom.")
        fc["conv_width"] = self._int(fc["conv_width"], "Ширина ядра", 1)
        kernel = np.asarray(fc["conv_kernel"], dtype=float).ravel()
        if fc["convolution"] == "custom":
            if kernel.size == 0 or kernel.size > 4097 or not np.isfinite(kernel).all() or not np.any(kernel):
                raise ValueError("Пользовательское ядро свёртки: 1…4097 конечных чисел, не все нули.")
        fc["conv_kernel"] = kernel.tolist()
        for key in ("qmf", "rmf", "emf"):
            fc[key] = bool(fc[key])
        if mode == "matched" and not (fc["qmf"] or fc["rmf"] or fc["emf"]):
            raise ValueError("Для режима фильтры+FNN включите хотя бы один из QMF, RMF, EMF.")
        if fc["event_rule"] not in ("centroid", "segments"):
            raise ValueError("Правило поиска событий: centroid или segments.")
        fc["event_margin"] = float(fc["event_margin"])
        if not 0.0 <= fc["event_margin"] <= 10.0:
            raise ValueError("Запас уверенности событий должен лежать в [0, 10].")
        fc["regularization"] = float(fc["regularization"])
        if not (0.0 < fc["regularization"] <= 10.0):
            raise ValueError("Регуляризация MF должна быть в (0, 10].")
        fc["min_events"] = self._int(fc["min_events"], "Минимум событий", 2)
        return fc

    # ------------------------------------------------------------------
    # Небольшие проверки типов
    # ------------------------------------------------------------------
    @staticmethod
    def _int(value, name, minimum=0) -> int:
        """Целое число ≥ minimum; bool и дробные значения отвергаются."""
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name}: требуется целое число.")
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name}: требуется целое число.") from exc
        if result != value or result < minimum:
            raise ValueError(f"{name}: требуется целое число ≥ {minimum}.")
        return result

    @classmethod
    def _int_list(cls, values, name, minimum=0) -> list[int]:
        """Список целых чисел (строка вида '64, 32' тоже допускается)."""
        if isinstance(values, str):
            values = [v for v in values.replace(";", ",").split(",") if v.strip()]
            values = [float(v) for v in values]
        return [cls._int(v, name, minimum) for v in values]

    # ------------------------------------------------------------------
    # Загрузка данных
    # ------------------------------------------------------------------
    def load_data(self, path) -> dict:
        """Прочитать файл с кривыми квадратур I и Q и вернуть словарь данных.

        Данные НЕ сохраняются в объекте — словарь принадлежит вызывающему коду.

        Поддерживаемые форматы (ТЗ: «импорт из файлов .csv, .npz»):

        **NPZ** (``numpy.savez``), ключи:

        * ``iq`` complex [N, A, T] — комплексные трассы, **или**
          ``iq`` real [N, A, T, 2] (последняя ось [I, Q]), **или**
          пара ``I`` и ``Q`` real [N, A, T]. Для одного атома допускается
          форма [N, T].
        * ``labels`` int [N, A] (или [N] для одного атома) — необязательно.
        * ``time_ns`` [T], ``shot_ids`` [N], ``ideal_probs`` [D] или [N, D] —
          необязательно.

        **CSV, «длинный» формат** — одна строка на временной отсчёт::

            shot,atom,sample,I,Q[,label][,time_ns]

        **CSV, «широкий» формат** — одна строка на трассу одного атома::

            shot,atom[,label],I_0,…,I_{T-1},Q_0,…,Q_{T-1}

        Разделитель (запятая, точка с запятой, табуляция) и вид CSV
        определяются автоматически по заголовку. Десятичный разделитель — точка.
        Номера ``atom`` — 0…A−1 без пропусков, ``sample`` — 0…T−1. Пропуски,
        NaN, повторы и неполные трассы считаются ошибкой файла.

        :param path: путь к .csv или .npz.
        :returns: словарь ``{"iq", "labels", "time_ns", "shot_ids", ...}``.
        :raises ValueError: с понятным описанием, если файл не соответствует схеме.
        """
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Файл не найден: {path}")
        suffix = path.suffix.lower()
        if suffix == ".npz":
            data = self._read_npz(path)
        elif suffix == ".csv":
            data = self._read_csv(path)
        else:
            raise ValueError("Поддерживаются только файлы .csv и .npz.")
        return self._check_data(data)

    @staticmethod
    def _read_npz(path: Path) -> dict:
        """Разобрать NPZ без pickle (allow_pickle=False защищает от чужого кода)."""
        try:
            with np.load(path, allow_pickle=False) as source:
                keys = set(source.files)
                if "iq" in keys:
                    iq = source["iq"]
                    if not np.iscomplexobj(iq):
                        if iq.ndim < 3 or iq.shape[-1] != 2:
                            raise ValueError("Вещественный массив iq должен иметь последнюю ось [I, Q].")
                        iq = iq[..., 0] + 1j * iq[..., 1]
                elif {"I", "Q"} <= keys:
                    i_part, q_part = source["I"], source["Q"]
                    if i_part.shape != q_part.shape:
                        raise ValueError("Массивы I и Q должны иметь одинаковую форму.")
                    iq = i_part.astype(np.float32) + 1j * q_part.astype(np.float32)
                else:
                    raise ValueError("NPZ должен содержать массив iq либо пару массивов I и Q.")
                data = {"iq": iq}
                for key in ("labels", "time_ns", "shot_ids", "ideal_probs"):
                    if key in keys:
                        data[key] = source[key]
        except (OSError, KeyError, TypeError, EOFError) as exc:
            raise ValueError(f"Не удалось прочитать NPZ: {exc}") from exc
        return data

    def _read_csv(self, path: Path) -> dict:
        """Разобрать CSV в «длинном» или «широком» формате (см. load_data)."""
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                head = handle.read(65536)
            if not head.strip():
                raise ValueError("CSV пуст.")
            try:
                delimiter = csv.Sniffer().sniff(head.splitlines()[0], delimiters=",;\t").delimiter
            except csv.Error:
                delimiter = ","
            frame = pd.read_csv(path, sep=delimiter, encoding="utf-8-sig")
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(f"Не удалось прочитать CSV: {exc}") from exc
        frame.columns = [str(c).strip() for c in frame.columns]
        if frame.empty:
            raise ValueError("CSV не содержит строк с данными.")
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame.isna().any().any():
            bad = [c for c in frame.columns if frame[c].isna().any()]
            raise ValueError(f"CSV: пустые или нечисловые значения в столбцах {bad}.")
        columns = set(frame.columns)
        if {"shot", "atom", "sample", "I", "Q"} <= columns:
            return self._csv_long(frame)
        if {"shot", "atom", "I_0", "Q_0"} <= columns:
            return self._csv_wide(frame)
        raise ValueError("CSV: нужны столбцы shot,atom,sample,I,Q (длинный формат) "
                         "или shot,atom,I_0…,Q_0… (широкий формат).")

    @staticmethod
    def _as_int(frame, column) -> np.ndarray:
        """Целочисленный столбец CSV без потери точности (большие shot id не идут через float)."""
        series = frame[column]
        if np.issubdtype(series.dtype, np.integer):
            return NeuralService._int_array(series.to_numpy(), f"CSV: столбец {column}")
        values = series.to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values != np.round(values)) or np.any(np.abs(values) > 2 ** 53):
            raise ValueError(f"CSV: столбец {column} должен содержать целые числа.")
        return values.astype(np.int64)

    @staticmethod
    def _int_array(values, name) -> np.ndarray:
        """Массив целых чисел: проверка конечности и целочисленности ДО приведения к int64."""
        values = np.asarray(values)
        if np.issubdtype(values.dtype, np.integer):
            # uint64 и др.: значения вне диапазона int64 при приведении молча стали бы отрицательными.
            if values.size and (values.max() > np.iinfo(np.int64).max or values.min() < np.iinfo(np.int64).min):
                raise ValueError(f"{name}: значения не помещаются в int64.")
            return values.astype(np.int64, copy=False)
        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all() \
                or np.any(values != np.round(values)) or np.any(np.abs(values) > 2 ** 53):
            raise ValueError(f"{name} должны быть целыми числами.")
        return values.astype(np.int64)

    def _csv_long(self, frame) -> dict:
        """Длинный CSV: одна строка — один отсчёт одной трассы."""
        shot, atom, sample = (self._as_int(frame, c) for c in ("shot", "atom", "sample"))
        shots = np.unique(shot)
        atoms = np.unique(atom)
        samples = np.unique(sample)
        if not np.array_equal(atoms, np.arange(len(atoms))):
            raise ValueError("CSV: atom должен принимать значения 0…A−1 без пропусков.")
        if not np.array_equal(samples, np.arange(len(samples))):
            raise ValueError("CSV: sample должен принимать значения 0…T−1 без пропусков.")
        shape = (len(shots), len(atoms), len(samples))
        if len(frame) != math.prod(shape):
            raise ValueError("CSV: у всех измерений должен быть одинаковый полный набор атомов и отсчётов.")
        shot_index = np.searchsorted(shots, shot)
        flat = (shot_index * shape[1] + atom) * shape[2] + sample
        if len(np.unique(flat)) != len(flat):
            raise ValueError("CSV: повторяется комбинация shot, atom, sample.")
        iq = np.empty(math.prod(shape), np.complex64)
        iq[flat] = frame["I"].to_numpy(np.float32) + 1j * frame["Q"].to_numpy(np.float32)
        data = {"iq": iq.reshape(shape), "shot_ids": shots}
        if "label" in frame:
            lab = np.empty(math.prod(shape), np.int64)
            lab[flat] = self._as_int(frame, "label")
            lab = lab.reshape(shape)
            if np.any(lab != lab[:, :, :1]):
                raise ValueError("CSV: label должен быть одинаковым во всех отсчётах одной трассы.")
            data["labels"] = lab[:, :, 0]
        if "time_ns" in frame:
            times = np.empty(math.prod(shape), float)
            times[flat] = frame["time_ns"].to_numpy(float)
            times = times.reshape(shape)
            if not np.allclose(times, times[:1, :1, :]):
                raise ValueError("CSV: временная сетка time_ns должна быть общей для всех трасс.")
            data["time_ns"] = times[0, 0]
        return data

    def _csv_wide(self, frame) -> dict:
        """Широкий CSV: одна строка — вся трасса одного атома одного измерения."""
        i_cols = sorted((c for c in frame.columns if c.startswith("I_")), key=lambda c: int(c[2:]))
        q_cols = sorted((c for c in frame.columns if c.startswith("Q_")), key=lambda c: int(c[2:]))
        samples = len(i_cols)
        if [int(c[2:]) for c in i_cols] != list(range(samples)) or [int(c[2:]) for c in q_cols] != list(range(samples)):
            raise ValueError("CSV: столбцы I_0…I_{T−1} и Q_0…Q_{T−1} должны идти без пропусков.")
        shot, atom = self._as_int(frame, "shot"), self._as_int(frame, "atom")
        shots, atoms = np.unique(shot), np.unique(atom)
        if not np.array_equal(atoms, np.arange(len(atoms))):
            raise ValueError("CSV: atom должен принимать значения 0…A−1 без пропусков.")
        if len(frame) != len(shots) * len(atoms):
            raise ValueError("CSV: для каждого shot нужна ровно одна строка на каждый атом.")
        row = np.searchsorted(shots, shot) * len(atoms) + atom
        if len(np.unique(row)) != len(row):
            raise ValueError("CSV: повторяется пара shot, atom.")
        iq = np.empty((len(row), samples), np.complex64)
        iq[row] = frame[i_cols].to_numpy(np.float32) + 1j * frame[q_cols].to_numpy(np.float32)
        data = {"iq": iq.reshape(len(shots), len(atoms), samples), "shot_ids": shots}
        if "label" in frame:
            lab = np.empty(len(row), np.int64)
            lab[row] = self._as_int(frame, "label")
            data["labels"] = lab.reshape(len(shots), len(atoms))
        return data

    @staticmethod
    def _check_data(data) -> dict:
        """Проверить и привести словарь данных к канонической форме.

        Не изменяет переданный словарь: возвращает новый (массивы, уже
        имеющие нужный тип, не копируются).
        """
        if not isinstance(data, dict) or "iq" not in data:
            raise ValueError("Ожидается словарь с ключом iq.")
        iq = np.asarray(data["iq"])
        if iq.ndim == 2:
            iq = iq[:, None, :]
        if iq.ndim != 3 or min(iq.shape) < 1:
            raise ValueError("iq должен иметь форму [N, A, T] (или [N, T] для одного атома).")
        if not np.iscomplexobj(iq):
            raise ValueError("iq должен быть комплексным (I = Re, Q = Im).")
        iq = iq.astype(np.complex64, copy=False)
        if not (np.isfinite(iq.real).all() and np.isfinite(iq.imag).all()):
            raise ValueError("В I/Q есть NaN или бесконечность.")
        n, a, t = iq.shape
        result = {"iq": iq, "labels": None, "time_ns": None}
        labels = data.get("labels")
        if labels is not None:
            labels = np.asarray(labels)
            if labels.ndim == 1 and a == 1:
                labels = labels[:, None]
            if labels.shape != (n, a):
                raise ValueError(f"labels должны иметь форму [N, A] = [{n}, {a}].")
            labels = NeuralService._int_array(labels, "labels")
            if np.any(labels < 0):
                raise ValueError("labels должны быть неотрицательными целыми числами.")
            result["labels"] = labels
        # Подготовленные метки до разметки утечки (если окно её применило):
        # физическая диагностика считается по подготовке, а не по псевдометкам.
        if data.get("prepared_labels") is not None:
            prepared = NeuralService._int_array(data["prepared_labels"], "prepared_labels")
            if prepared.shape != (n, a):
                raise ValueError("prepared_labels должны иметь форму [N, A].")
            result["prepared_labels"] = prepared
        if data.get("time_ns") is not None:
            time_ns = np.asarray(data["time_ns"], dtype=float).ravel()
            if time_ns.shape != (t,) or not np.isfinite(time_ns).all() or np.any(np.diff(time_ns) <= 0):
                raise ValueError("time_ns должен быть строго возрастающим массивом длины T.")
            result["time_ns"] = time_ns
        shot_ids = data.get("shot_ids")
        shot_ids = np.arange(n, dtype=np.int64) if shot_ids is None else \
            NeuralService._int_array(np.asarray(shot_ids).ravel(), "shot_ids")
        if shot_ids.shape != (n,) or len(np.unique(shot_ids)) != n:
            raise ValueError("shot_ids должны быть уникальны, по одному на измерение.")
        result["shot_ids"] = shot_ids
        if data.get("ideal_probs") is not None:
            p = np.asarray(data["ideal_probs"], dtype=float)
            if p.ndim not in (1, 2) or (p.ndim == 2 and len(p) != n) or not np.isfinite(p).all() or np.any(p < 0):
                raise ValueError("ideal_probs должен иметь форму [D] или [N, D] с неотрицательными числами.")
            if not np.allclose(p.sum(axis=-1), 1.0, atol=1e-6):
                raise ValueError("ideal_probs должны быть нормированы (сумма = 1).")
            result["ideal_probs"] = p
        return result

    # ------------------------------------------------------------------
    # Предобработка: окно, свёртка, MTV
    # ------------------------------------------------------------------
    def _conv_kernel(self, fc=None) -> np.ndarray | None:
        """Ядро свёрточного фильтра (ТЗ: «свёрточные фильтры с настраиваемым ядром»).

        * boxcar — скользящее среднее ширины w;
        * gauss  — гауссово окно, σ = w/4, длина w (нечётная);
        * exp    — причинное экспоненциальное сглаживание с постоянной w;
        * custom — коэффициенты пользователя.
        Ядро нормируется на сумму коэффициентов (если она не ноль), чтобы не
        менять амплитуду сигнала.
        """
        fc = fc or self.config["filter_config"]
        kind, width = fc["convolution"], fc["conv_width"]
        if kind == "none":
            return None
        if kind == "boxcar":
            kernel = np.ones(width)
        elif kind == "gauss":
            length = width | 1
            x = np.arange(length) - length // 2
            kernel = np.exp(-0.5 * (x / max(width / 4.0, 0.5)) ** 2)
        elif kind == "exp":
            # convolve1d центрирует ядро; дополнение нулями слева делает его причинным.
            tail = np.exp(-np.arange(width) / max(width / 3.0, 1e-9))
            kernel = np.concatenate([np.zeros(width - 1), tail])
        else:
            kernel = np.asarray(fc["conv_kernel"], dtype=float)
        total = kernel.sum()
        return kernel / total if abs(total) > 1e-12 else kernel

    def _prepare(self, iq: np.ndarray, config=None) -> np.ndarray:
        """Выбрать атомы, вырезать окно и применить свёртку к батчу [B, A, T]."""
        config = config or self.config
        x = iq[:, config["atoms"], :]
        window = config["filter_config"]["window"]
        if window is not None:
            x = x[:, :, window[0]:window[1]]
        kernel = self._conv_kernel(config["filter_config"])
        if kernel is not None:
            x = (convolve1d(x.real, kernel, axis=-1, mode="nearest") +
                 1j * convolve1d(x.imag, kernel, axis=-1, mode="nearest")).astype(np.complex64)
        return x

    @staticmethod
    def mean_trace_value(iq: np.ndarray) -> np.ndarray:
        """MTV статьи (§V-A): временное среднее трассы → комплексное число.

        :param iq: complex [..., T].
        :returns: complex [...] — одна точка IQ на трассу («пятно»).
        """
        return iq.mean(axis=-1)

    @staticmethod
    def _batches(indices: np.ndarray, size: int):
        for start in range(0, len(indices), size):
            yield indices[start:start + size]

    # ------------------------------------------------------------------
    # Фильтры статьи: QMF / RMF / EMF
    # ------------------------------------------------------------------
    def _matched_kernel(self, delta: np.ndarray, var_i: np.ndarray, var_q: np.ndarray, reg: float) -> np.ndarray:
        """Ядро согласованного фильтра для разности средних трасс ``delta``.

        Статья: K = Δμ / (σ1² − σ0²). Здесь (см. docstring модуля) знаменатель
        — регуляризованная СУММА дисперсий, отдельно для I и Q:
        ``K = ΔI/(σ²_I + λS) + i·ΔQ/(σ²_Q + λS)``, S — средняя дисперсия.
        Ядро нормируется на единичную длину; признак трассы x —
        ``Re Σ_t x(t)·conj(K(t)) = Σ_t (K_I·I + K_Q·Q)``.
        """
        scale = max(float(np.mean(var_i + var_q)) / 2.0, 1e-20)
        kernel = delta.real / (var_i + reg * scale) + 1j * delta.imag / (var_q + reg * scale)
        norm = np.linalg.norm(kernel)
        return (kernel / norm if norm > 0 else np.zeros_like(kernel)).astype(np.complex64)

    @staticmethod
    def _moments(traces_iter):
        """Потоково посчитать среднее и дисперсии I и Q по набору трасс [B, T].

        Используется устойчивое объединение моментов по батчам (Chan et al.):
        в float64 и без формулы E[x²] − E[x]², которая теряет точность при
        большом постоянном смещении сигнала.
        """
        count, mean, m2_i, m2_q = 0, None, None, None
        for traces in traces_iter:
            x = traces.astype(np.complex128)
            nb = len(x)
            if not nb:
                continue
            mb = x.mean(axis=0)
            d = x - mb
            bi, bq = (d.real ** 2).sum(axis=0), (d.imag ** 2).sum(axis=0)
            if mean is None:
                count, mean, m2_i, m2_q = nb, mb, bi, bq
                continue
            delta = mb - mean
            total = count + nb
            mean = mean + delta * nb / total
            m2_i = m2_i + bi + delta.real ** 2 * count * nb / total
            m2_q = m2_q + bq + delta.imag ** 2 * count * nb / total
            count = total
        if not count:
            return None
        return mean, m2_i / count, m2_q / count, count

    def _calibrate(self, data, indices, config, batch_size, cancel_event=None) -> dict:
        """Откалибровать центроиды, бенчмарки и банк фильтров ТОЛЬКО по train.

        Возвращает новый словарь ``filters``; текущее состояние сервиса не
        меняется (замену делает :meth:`train` после успешной калибровки).

        Шаги для каждого используемого атома a с k уровнями:
        1. средние трассы μ_s(t) и дисперсии I/Q каждого уровня s;
        2. центроиды MTV c_s = среднее MTV трасс уровня s;
        3. параметры LDA/QDA на MTV (бенчмарки, табл. V статьи);
        4. пометка трасс-событий: правило ``centroid`` (статья, §V-B) —
           трасса уровня s, MTV которой ближе к c_j (j ≠ s), — событие s→j;
           правило ``segments`` — ранний сегмент у c_s, поздний у c_j;
        5. QMF для каждой пары (i<j), RMF для j→i и EMF для i→j: ядро
           строится между средней трассой событий и «чистыми» трассами
           исходного уровня. Если событий меньше ``min_events``, ядро
           нулевое (признак всегда 0) и выдаётся предупреждение —
           искусственных событий не создаётся.
        """
        x_all, y_all = data["iq"], data["labels"]
        levels, fc = config["levels"], config["filter_config"]
        reg = fc["regularization"]
        filters = dict(variant="sum-of-variances, separate I/Q, regularized",
                       atoms=[], means=[], centroids=[], lda=[], qda=[],
                       kernels=[], kernel_atoms=[], descriptions=[], warnings=[],
                       time_ns=None if data.get("time_ns") is None else np.array(data["time_ns"]))

        def traces(sel, a):
            for b in self._batches(sel, batch_size):
                self._raise_if_cancelled(cancel_event)
                yield self._prepare(x_all[b], config)[:, a]

        for a, k in enumerate(levels):
            ya = y_all[indices, config["atoms"][a]]   # метки КАНАЛА атома, а не позиции в списке
            means, var_i, var_q, mtv_stats = [], [], [], []
            for s in range(k):
                sel = indices[ya == s]
                if len(sel) < 2:
                    raise ValueError(
                        f"В обучающей выборке для атома {config['atoms'][a]} нет хотя бы двух трасс уровня {s}. "
                        "Уменьшите число уровней, добавьте данные или выполните «Поиск утечки».")
                mean, vi, vq, _ = self._moments(traces(sel, a))
                means.append(mean.astype(np.complex64))
                var_i.append(vi)
                var_q.append(vq)
            means = np.stack(means)
            centroids = means.mean(axis=-1)
            # Параметры гауссовых бенчмарков на MTV (2D точки I, Q).
            mtv = np.concatenate([self.mean_trace_value(t) for t in traces(indices, a)])
            points = np.stack([mtv.real, mtv.imag], axis=1).astype(float)
            class_means, class_covs, priors = [], [], []
            for s in range(k):
                p = points[ya == s]
                class_means.append(p.mean(axis=0))
                class_covs.append(np.cov(p.T) + 1e-12 * np.eye(2))
                priors.append(len(p) / len(points))
            pooled = sum(c * pr for c, pr in zip(class_covs, priors))
            filters["lda"].append(dict(means=np.array(class_means), cov=pooled, priors=np.array(priors)))
            filters["qda"].append(dict(means=np.array(class_means), covs=np.array(class_covs), priors=np.array(priors)))
            filters["atoms"].append(config["atoms"][a])
            filters["means"].append(means)
            filters["centroids"].append(centroids)
            if config["mode"] != "matched":
                continue
            # Пометка трасс-событий (правило centroid или segments).
            events = self._event_labels(x_all, indices, a, ya, centroids, means, config, batch_size, cancel_event)
            for low, high in itertools.combinations(range(k), 2):
                if fc["qmf"]:
                    kernel = self._matched_kernel(means[high] - means[low], var_i[low] + var_i[high],
                                                  var_q[low] + var_q[high], reg)
                    self._add_kernel(filters, a, kernel, "QMF", low, high, True, None)
                for enabled, kind, source, target in ((fc["rmf"], "RMF", high, low), (fc["emf"], "EMF", low, high)):
                    if not enabled:
                        continue
                    event_sel = indices[(ya == source) & (events == target)]
                    clean_sel = indices[(ya == source) & (events == source)]
                    available = len(event_sel) >= fc["min_events"] and len(clean_sel) >= 2
                    kernel = np.zeros(means.shape[-1], np.complex64)
                    if available:
                        ev_mean, ev_vi, ev_vq, _ = self._moments(traces(event_sel, a))
                        cl_mean, cl_vi, cl_vq, _ = self._moments(traces(clean_sel, a))
                        kernel = self._matched_kernel(ev_mean - cl_mean, ev_vi + cl_vi, ev_vq + cl_vq, reg)
                    else:
                        filters["warnings"].append(
                            f"{kind} атом {config['atoms'][a]}, {source}→{target}: найдено {len(event_sel)} трасс-событий "
                            f"(нужно ≥ {fc['min_events']}); фильтр не откалиброван, его признак равен 0.")
                    self._add_kernel(filters, a, kernel, kind, source, target, available, int(len(event_sel)))
        if config["mode"] == "matched" and not any(d["available"] for d in filters["descriptions"]):
            raise ValueError("Не удалось откалибровать ни одного фильтра. Включите QMF или добавьте данные.")
        return filters

    @staticmethod
    def _add_kernel(filters, atom, kernel, kind, source, target, available, events):
        filters["kernels"].append(kernel)
        filters["kernel_atoms"].append(atom)
        filters["descriptions"].append(dict(atom=atom, kind=kind, source=int(source), target=int(target),
                                            available=bool(available), events=events))

    def _event_labels(self, x_all, indices, a, ya, centroids, means, config, batch_size, cancel_event=None):
        """Для каждой трассы train вернуть «наблюдаемый» уровень атома a.

        Правило ``centroid`` (статья, §V-B): ближайший центроид MTV. Трасса
        уровня s, ближайшая к своему c_s, — «чистая» (значение s). Трасса,
        ближайшая к чужому c_j, — событие s→j, если расстояние до c_s больше
        расстояния до c_j в (1 + margin) раз; иначе она «неопределённая» (−1)
        и не используется ни как событие, ни как чистая. При margin = 0
        правило в точности совпадает с правилом статьи.
        Правило ``segments``: уровень по позднему трети трассы при условии,
        что ранняя треть соответствует подготовленному уровню; иначе −1
        («неопределённо», не используется ни как событие, ни как чистая).
        """
        fc = config["filter_config"]
        margin = 1.0 + fc["event_margin"]
        result = np.empty(len(indices), np.int64)
        position = 0
        third = max(1, means.shape[-1] // 3)
        early_ref = means[:, :third].mean(axis=-1)
        late_ref = means[:, -third:].mean(axis=-1)
        for b in self._batches(indices, batch_size):
            self._raise_if_cancelled(cancel_event)
            tr = self._prepare(x_all[b], config)[:, a]
            ys = ya[position:position + len(b)]
            if fc["event_rule"] == "centroid":
                dist = np.abs(self.mean_trace_value(tr)[:, None] - centroids[None, :])
                nearest = dist.argmin(axis=1)
                own = dist[np.arange(len(b)), ys]
                confident = own > margin * dist[np.arange(len(b)), nearest]
                observed = np.where(nearest == ys, ys, np.where(confident, nearest, -1))
            else:
                early = np.abs(tr[:, :third].mean(axis=-1)[:, None] - early_ref).argmin(axis=1)
                late = np.abs(tr[:, -third:].mean(axis=-1)[:, None] - late_ref).argmin(axis=1)
                observed = np.where(early == ys, late, -1)
            result[position:position + len(b)] = observed
            position += len(b)
        return result

    def _features(self, iq_batch: np.ndarray, config=None, filters=None) -> np.ndarray:
        """Признаки для FNN (до нормировки) для батча [B, A, T].

        raw: развёрнутые I и Q выбранных атомов (окно и свёртка учтены);
        matched: по одному действительному числу на каждый фильтр банка.
        """
        config = config or self.config
        filters = self.filters if filters is None else filters
        x = self._prepare(iq_batch, config)
        if config["mode"] == "raw":
            return np.concatenate([x.real, x.imag], axis=-1).reshape(len(x), -1).astype(np.float32)
        out = np.empty((len(x), len(filters["kernels"])), np.float32)
        for j, (atom, kernel) in enumerate(zip(filters["kernel_atoms"], filters["kernels"])):
            out[:, j] = np.real(x[:, atom] @ np.conjugate(kernel))
        return out

    def apply_filters(self, data, indices=None, *, normalized=False, batch_size=4096) -> np.ndarray:
        """Применить ТЕКУЩИЕ (замороженные) фильтры модели к данным.

        Возвращает матрицу признаков [N, P] — ровно то, что подаётся на вход
        FNN (окно → свёртка → банк QMF/RMF/EMF или развёрнутые I/Q). Фильтры
        не перекалибровываются: они являются частью обученной модели.

        :param data: словарь данных (см. :meth:`load_data`).
        :param indices: номера измерений; None — все.
        :param normalized: True — после стандартизации (как видит сеть).
        """
        self._require_fitted()
        data = self._check_compatible(data)
        indices = self._indices(data, indices)
        parts = []
        for b in self._batches(indices, batch_size):
            f = self._features(data["iq"][b])
            if normalized:
                f = (f - self.normalizer["mean"]) / self.normalizer["scale"]
            parts.append(f)
        return np.concatenate(parts) if parts else np.empty((0, self.feature_count), np.float32)

    def filter_diagnostics(self, data, indices=None, *, atoms=None, max_points=5000, batch_size=4096) -> dict:
        """Данные для панели «Применение фильтров» режима валидации.

        Для выбранных атомов возвращает выходы их фильтров (гистограммы по
        классам), ядра фильтров и долю трасс, которые правило событий
        текущей модели пометило как распад или возбуждение.

        :returns: ``{atom: {"names", "values" [M, F], "labels" [M] | None,
            "kernels" [F, T'], "event_fraction": {...}}}`` — M ≤ max_points.
        """
        self._require_fitted()
        data = self._check_compatible(data)
        indices = self._indices(data, indices)
        if len(indices) > max_points:
            indices = np.sort(np.random.default_rng(0).choice(indices, max_points, replace=False))
        used = self.config["atoms"]
        atoms = used if atoms is None else [a for a in atoms if a in used]
        result = {}
        features = None
        if self.config["mode"] == "matched":
            features = np.concatenate([self._features(data["iq"][b]) for b in self._batches(indices, batch_size)])
        for channel in atoms:
            a = used.index(channel)
            entry = {"labels": None if data["labels"] is None else data["labels"][indices, channel]}
            if features is not None:
                cols = [j for j, d in enumerate(self.filters["descriptions"]) if d["atom"] == a]
                entry["names"] = [self._kernel_name(self.filters["descriptions"][j]) for j in cols]
                entry["values"] = features[:, cols]
                entry["kernels"] = np.array([self.filters["kernels"][j] for j in cols])
                entry["available"] = [self.filters["descriptions"][j]["available"] for j in cols]
            else:
                entry["names"], entry["values"], entry["kernels"], entry["available"] = [], None, None, []
            if data["labels"] is not None:
                ya = data["labels"][indices, channel]
                ok = ya < self.config["levels"][a]
                obs = self._event_labels(data["iq"], indices[ok], a, ya[ok], self.filters["centroids"][a],
                                         self.filters["means"][a], self.config, batch_size)
                entry["event_fraction"] = {
                    "relaxation": float(np.mean(obs[obs >= 0] < ya[ok][obs >= 0])) if np.any(obs >= 0) else None,
                    "excitation": float(np.mean(obs[obs >= 0] > ya[ok][obs >= 0])) if np.any(obs >= 0) else None,
                    "rule": self.config["filter_config"]["event_rule"],
                }
            result[channel] = entry
        return result

    @staticmethod
    def _kernel_name(d) -> str:
        return f"{d['kind']} {d['source']}→{d['target']}" + ("" if d["available"] else " (н/д)")

    # ------------------------------------------------------------------
    # Поиск естественной утечки (§V-A статьи)
    # ------------------------------------------------------------------
    def find_leakage(self, data, atom, *, indices=None, n_clusters=3, max_points=2000, seed=0) -> dict:
        """Найти трассы, вероятно утёкшие в уровень вне вычислительного базиса.

        Метод статьи (§V-A): MTV-точки трасс атома кластеризуются спектральной
        кластеризацией на ``n_clusters`` кластеров. Затем кластеры
        сопоставляются подготовленным состояниям 0…k−2 так, чтобы максимально
        совпадать с метками подготовки (венгерский алгоритм по таблице
        сопряжённости — «маркировка по вероятности утечки при подготовке
        вычислительного состояния»). Оставшийся кластер считается утечкой в
        уровень ``n_clusters − 1`` (обычно |2⟩).

        Это ГИПОТЕЗА, а не измерение (статья: наименьший кластер лишь
        «вероятно» соответствует утечке). Поэтому проверяется правдоподобие:
        кластер утечки самый маленький, ни для одной подготовки не составляет
        большинства и чаще встречается после |1⟩, чем после |0⟩; нарушения
        возвращаются в ``warnings``. Метод возвращает предложенные метки и
        отчёт, окно показывает их пользователю и применяет только после
        подтверждения. Данные и метки на месте не изменяются.

        Реализация спектральной кластеризации (без scikit-learn):
        RBF-аффинность W_ij = exp(−|m_i − m_j|²/(2s²)), s — медиана попарных
        расстояний (параметр в статье не указан — это выбор реализации);
        нормированная матрица D^{-1/2} W D^{-1/2}; k старших собственных
        векторов; нормировка строк; k-means. Чтобы не строить матрицу N×N
        на всех данных, кластеризуется подвыборка ≤ max_points, остальные
        точки относятся к ближайшему центру кластера в плоскости IQ.

        :param atom: номер канала в данных.
        :returns: ``{"labels": новые метки атома [N] (для indices),
            "indices", "cluster_sizes", "mapping", "leaked_level", "warnings",
            "leaked_count", "agreement", "mtv"}``.
        """
        data = self._check_data(data)
        if data["labels"] is None:
            raise ValueError("Для поиска утечки нужны метки подготовленных состояний.")
        atom = self._int(atom, "atom", 0)
        if atom >= data["iq"].shape[1]:
            raise ValueError("Нет такого атома в данных.")
        k = self._int(n_clusters, "Число кластеров", 2)
        indices = self._indices(data, indices)
        mtv = self.mean_trace_value(data["iq"][indices, atom])
        prepared = data["labels"][indices, atom]
        rng = np.random.default_rng(seed)
        sample = indices if len(indices) <= max_points else rng.choice(len(indices), max_points, replace=False)
        sample = np.arange(len(indices)) if len(indices) <= max_points else np.sort(sample)
        pts = np.stack([mtv[sample].real, mtv[sample].imag], axis=1).astype(float)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = (diff ** 2).sum(-1)
        s2 = np.median(d2[np.triu_indices(len(pts), 1)]) or 1.0
        w = np.exp(-d2 / (2 * s2))
        np.fill_diagonal(w, 0.0)
        deg = w.sum(axis=1)
        inv = 1.0 / np.sqrt(np.maximum(deg, 1e-300))
        m = inv[:, None] * w * inv[None, :]
        _, vectors = np.linalg.eigh(m)
        emb = vectors[:, -k:]
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-300)
        _, cluster = kmeans2(emb, k, minit="++", seed=int(seed))
        centers = np.array([pts[cluster == c].mean(axis=0) if np.any(cluster == c) else np.full(2, np.inf)
                            for c in range(k)])
        allp = np.stack([mtv.real, mtv.imag], axis=1)
        assigned = np.argmin(((allp[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        # Сопоставление кластеров подготовленным уровням 0…k−2.
        computational = list(range(k - 1))
        table = np.array([[np.sum((assigned == c) & (prepared == s)) for c in range(k)] for s in computational])
        rows, cols = linear_sum_assignment(-table)
        mapping = {int(c): int(s) for s, c in zip(rows, cols)}
        leaked_clusters = [c for c in range(k) if c not in mapping]
        new = prepared.copy()
        leaked = np.isin(assigned, leaked_clusters)
        new[leaked] = k - 1
        # Проверки правдоподобия (статья: «наименьший кластер, вероятно,
        # утечка»). Нарушение не запрещает предложение, но выводится явно.
        warnings = []
        sizes = [int(np.sum(assigned == c)) for c in range(k)]
        for c in leaked_clusters:
            if sizes[c] != min(sizes):
                warnings.append(f"Кластер утечки {c} не самый маленький ({sizes[c]} из {sizes}).")
            for s in computational:
                prepared_s = prepared == s
                if prepared_s.any() and np.mean(assigned[prepared_s] == c) > 0.5:
                    warnings.append(f"Кластер утечки {c} содержит большинство трасс, подготовленных в |{s}⟩.")
            if k >= 3 and np.any(prepared == 0) and np.any(prepared == 1):
                if np.mean(assigned[prepared == 1] == c) < np.mean(assigned[prepared == 0] == c):
                    warnings.append(f"Кластер утечки {c} чаще встречается после подготовки |0⟩, чем |1⟩ — "
                                    "для утечки из |1⟩ в |2⟩ ожидается обратное.")
        if np.any(prepared >= k - 1):
            warnings.append(f"В метках уже есть уровень ≥ {k - 1}: проверьте, нужен ли поиск утечки.")
        return dict(labels=new, indices=indices, mtv=mtv, clusters=assigned, warnings=warnings,
                    centers=centers, atom=atom,
                    cluster_sizes=[int(np.sum(assigned == c)) for c in range(k)],
                    mapping=mapping, leaked_clusters=leaked_clusters, leaked_level=k - 1,
                    leaked_count=int(leaked.sum()),
                    agreement=float(np.mean(np.array([mapping.get(int(c), -1) for c in assigned])[~leaked]
                                            == prepared[~leaked])) if np.any(~leaked) else None)

    def leakage_labels(self, data, proposal: dict, indices) -> np.ndarray:
        """Применить ГОТОВОЕ предложение :meth:`find_leakage` к другим измерениям.

        Кластеризация заново не выполняется: каждая MTV-точка относится к
        ближайшему центру найденных кластеров (так же, как точки вне
        подвыборки в find_leakage). Нужна, чтобы отложенный test получил ту же
        разметку, что train, и метрики на нём оставались сопоставимыми.

        :returns: новые метки атома для ``indices``.
        """
        data = self._check_data(data)
        indices = self._indices(data, indices)
        atom = proposal["atom"]
        mtv = self.mean_trace_value(data["iq"][indices, atom])
        pts = np.stack([mtv.real, mtv.imag], axis=1)
        assigned = np.argmin(((pts[:, None, :] - proposal["centers"][None]) ** 2).sum(-1), axis=1)
        labels = data["labels"][indices, atom].copy()
        labels[np.isin(assigned, proposal["leaked_clusters"])] = proposal["leaked_level"]
        return labels

    # ------------------------------------------------------------------
    # Сеть
    # ------------------------------------------------------------------
    def _build_model(self, config, feature_count) -> nn.Module:
        """Создать FNN по конфигурации; P = feature_count входов.

        Если hidden_layers = None, слои выбираются как в статье:
        [⌊P/2⌋, ⌊P/4⌋], но не уже [2k+2, k+1]; для общей сети не меньше
        [4·Σk, 2·Σk], иначе узкое горлышко не вместит все выходы.
        per_atom_networks=True: nn.ModuleList из сетей «P → … → k_a»
        (логиты всех атомов затем склеиваются); иначе одна сеть
        «P → … → Σk_a».
        """
        hidden = config["hidden_layers"]
        if hidden is None:
            if config["per_atom_networks"]:
                # Как в статье, но не уже 2k+2 и k+1 нейронов (k — уровней атома):
                # при малом P (один кубит, P = 3) слои ⌊P/2⌋, ⌊P/4⌋ = 1, 0 вырождаются.
                k_max = max(config["levels"])
                hidden = [max(feature_count // 2, 2 * k_max + 2), max(feature_count // 4, k_max + 1)]
            else:
                # Общей сети нужен запас: выходов Σk, а не k одного атома.
                outputs = sum(config["levels"])
                hidden = [max(feature_count // 2, 4 * outputs), max(feature_count // 4, 2 * outputs)]
        act = self.ACTIVATIONS[config["activation"]]

        def mlp(out_size):
            dims = [feature_count, *hidden, out_size]
            layers = []
            for i, (left, right) in enumerate(zip(dims, dims[1:])):
                layers.append(nn.Linear(left, right))
                if i < len(dims) - 2:
                    layers.append(act())
                    if config["dropout"]:
                        layers.append(nn.Dropout(config["dropout"]))
            return nn.Sequential(*layers)

        outputs = list(config["levels"]) if config["per_atom_networks"] else [sum(config["levels"])]
        # Число параметров считается ДО создания слоёв: опечатка вроде 65536×65536
        # не должна сначала занять десятки гигабайт памяти.
        count = 0
        for out_size in outputs:
            dims = [feature_count, *hidden, out_size]
            count += sum((left + 1) * right for left, right in zip(dims, dims[1:]))
        if count > 200_000_000:
            raise ValueError(f"Сеть содержит {count:,} параметров — уменьшите слои или используйте фильтры+FNN.")
        model = nn.ModuleList([mlp(k) for k in outputs])
        return model.to(self.device)

    def _logits(self, features: torch.Tensor) -> torch.Tensor:
        """Прямой проход: склеенные логиты всех атомов [B, Σk]."""
        return torch.cat([net(features) for net in self.model], dim=1)

    def _split_logits(self, logits):
        return torch.split(logits, self.config["levels"], dim=1)

    def _tensor(self, iq_batch) -> torch.Tensor:
        f = (self._features(iq_batch) - self.normalizer["mean"]) / self.normalizer["scale"]
        return torch.as_tensor(f, dtype=torch.float32, device=self.device)

    def _loss(self, logits, labels):
        """Средняя по атомам cross-entropy (масштаб не растёт с числом атомов)."""
        return torch.stack([nn.functional.cross_entropy(head, labels[:, a])
                            for a, head in enumerate(self._split_logits(logits))]).mean()

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------
    def train(self, data, *, train_indices=None, val_indices=None, val_data=None, config=None,
              epochs=50, batch_size=256, optimizer="Adam", learning_rate=1e-3, weight_decay=0.0,
              momentum=0.9, fine_tune=False, callback: Callable[[dict], None] | None = None,
              cancel_event=None) -> dict:
        """Обучить новую сеть или дообучить текущую.

        **Новое обучение** (``fine_tune=False``): по train калибруются
        фильтры и нормировка, строится новая сеть, и только затем она
        заменяет текущую (при ошибке калибровки прежняя модель цела).

        **Дообучение** (``fine_tune=True``): продолжается обучение текущих
        (обученных или загруженных из файла) весов. Фильтры, нормировка и
        архитектура заморожены — иначе изменится входное пространство сети.
        Состояние оптимизатора восстанавливается, если тип оптимизатора тот же.

        Данные не копируются целиком: из ``data["iq"]`` по индексам берётся
        только очередной batch.

        :param data: словарь данных с labels.
        :param train_indices: номера измерений для обучения (None — все).
        :param val_indices: номера измерений validation из ``data``.
        :param val_data: отдельный набор для validation (вместо val_indices).
        :param config: новая конфигурация (результат :meth:`configure`);
            None — текущая.
        :param epochs: число эпох.
        :param batch_size: размер мини-батча.
        :param optimizer: ``Adam`` | ``AdamW`` | ``SGD`` | ``RMSprop``.
        :param learning_rate: шаг обучения.
        :param weight_decay: L2-регуляризация весов.
        :param momentum: момент (для SGD и RMSprop).
        :param callback: функция, получающая словарь метрик после каждой эпохи
            (epoch, loss, accuracy, val_loss, val_accuracy, val_assignment_fidelity,
            val_macro_f1, val_xeb, lr).
        :param cancel_event: ``threading.Event``; если установлен — обучение
            останавливается после текущего батча, достигнутые веса остаются.
        :returns: ``{"cancelled", "epochs_completed", "total_epochs",
            "last", "warnings", "excluded"}``.
        """
        epochs = self._int(epochs, "Эпохи", 1)
        batch_size = self._int(batch_size, "Размер батча", 1)
        learning_rate, weight_decay, momentum = float(learning_rate), float(weight_decay), float(momentum)
        if not (learning_rate > 0 and math.isfinite(learning_rate)):
            raise ValueError("Learning rate должен быть положительным.")
        if weight_decay < 0 or not 0 <= momentum < 1:
            raise ValueError("Weight decay ≥ 0, momentum в [0, 1).")
        names = {n.lower(): n for n in self.OPTIMIZERS}
        if str(optimizer).lower() not in names:
            raise ValueError(f"Оптимизатор: {', '.join(self.OPTIMIZERS)}.")
        optimizer = names[str(optimizer).lower()]
        data = self._check_data(data)
        if data["labels"] is None:
            raise ValueError("Для обучения нужны метки (labels).")
        if fine_tune:
            if not self.is_fitted:
                raise ValueError("Для дообучения сначала обучите сеть или загрузите веса.")
            if config is not None and self._comparable(config) != self._comparable(self.config):
                raise ValueError("При дообучении архитектура и фильтры заморожены. "
                                 "Чтобы их изменить, обучите новую сеть.")
            config = self.config
            data = self._check_compatible(data)
        else:
            config = copy.deepcopy(config or self.config)
            if max(config["atoms"]) >= data["iq"].shape[1]:
                raise ValueError(f"В данных {data['iq'].shape[1]} атомов, а выбран атом {max(config['atoms'])}.")
            window = config["filter_config"]["window"]
            samples = data["iq"].shape[2]
            if window is not None and not (0 <= window[0] < window[1] <= samples):
                raise ValueError(f"Окно {window} выходит за пределы трассы из {samples} отсчётов.")
        train_idx, excluded_train = self._usable(data, self._indices(data, train_indices), config)
        if len(train_idx) < 2:
            raise ValueError("После проверки меток в train осталось меньше двух измерений.")
        if val_data is not None:
            val_data = self._check_data(val_data)
            if val_data["iq"].shape[2] != data["iq"].shape[2] or max(config["atoms"]) >= val_data["iq"].shape[1]:
                raise ValueError("Validation-набор должен иметь ту же длину трасс и нужные атомы, что и train.")
            if (data["time_ns"] is None) != (val_data["time_ns"] is None) or (
                    data["time_ns"] is not None and not np.allclose(data["time_ns"], val_data["time_ns"], rtol=1e-6)):
                raise ValueError("Временные сетки train и validation различаются.")
            val_idx, _ = self._usable(val_data, self._indices(val_data, None), config)
        elif val_indices is not None:
            val_data = data
            val_idx, _ = self._usable(data, self._indices(data, val_indices), config)
            if np.intersect1d(val_idx, train_idx).size:
                raise ValueError("Train и validation пересекаются — передайте непересекающиеся индексы.")
        else:
            val_idx = None
        warnings = []
        if excluded_train:
            warnings.append(f"Из train исключено {excluded_train} измерений с уровнями выше заданных.")
        if self._cancelled(cancel_event):
            return dict(cancelled=True, epochs_completed=0, total_epochs=self.trained_epochs, last={}, warnings=warnings)

        if not fine_tune:
            # Кандидат: всё считается в локальных переменных, затем атомарная замена.
            torch.manual_seed(config["seed"])
            filters = self._calibrate(data, train_idx, config, batch_size, cancel_event)
            total = square = None
            for b in self._batches(train_idx, batch_size):
                self._raise_if_cancelled(cancel_event)
                f = self._features(data["iq"][b], config, filters).astype(np.float64)
                total = f.sum(0) if total is None else total + f.sum(0)
                square = (f * f).sum(0) if square is None else square + (f * f).sum(0)
            mean = total / len(train_idx)
            scale = np.sqrt(np.maximum(square / len(train_idx) - mean ** 2, 0.0))
            scale[scale < 1e-8] = 1.0
            model = self._build_model(config, len(mean))
            # --- замена текущего состояния (только после успешной калибровки) ---
            self.config, self.filters, self.model = config, filters, model
            self.normalizer = dict(mean=mean.astype(np.float32), scale=scale.astype(np.float32))
            self.feature_count = len(mean)
            self.input_channels, self.input_samples = data["iq"].shape[1], data["iq"].shape[2]
            self.optimizer_state, self.optimizer_name = None, None
            self.trained_epochs = 0
            self._fitted = False
        warnings.extend(self.filters.get("warnings", []))

        params = dict(lr=learning_rate, weight_decay=weight_decay)
        if optimizer in ("SGD", "RMSprop"):
            params["momentum"] = momentum
        opt = getattr(torch.optim, optimizer)(self.model.parameters(), **params)
        if fine_tune and self.optimizer_state is not None and self.optimizer_name == optimizer:
            try:
                opt.load_state_dict(self.optimizer_state)
                for group in opt.param_groups:
                    group.update(params)
            except (ValueError, KeyError, RuntimeError):
                warnings.append("Состояние оптимизатора несовместимо — дообучение с новым оптимизатором.")
        self.optimizer_state = None   # состояние теперь живёт в opt; второй копии не держим
        self.optimizer_name = optimizer
        rng = np.random.default_rng(config["seed"] + self.trained_epochs)
        labels_all = data["labels"][:, config["atoms"]]
        cancelled, completed, last = False, 0, {}
        try:
            for _ in range(epochs):
                if self._cancelled(cancel_event):
                    cancelled = True
                    break
                self.model.train()
                loss_sum, right, seen = 0.0, 0, 0
                for b in self._batches(rng.permutation(train_idx), batch_size):
                    if self._cancelled(cancel_event):
                        cancelled = True
                        break
                    b = np.sort(b)  # упорядоченная выборка из массива быстрее
                    features = self._tensor(data["iq"][b])
                    target = torch.as_tensor(labels_all[b], dtype=torch.long, device=self.device)
                    opt.zero_grad(set_to_none=True)
                    logits = self._logits(features)
                    loss = self._loss(logits, target)
                    if not torch.isfinite(loss):
                        raise ValueError("Loss стал NaN/Inf — уменьшите learning rate и проверьте данные.")
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    opt.step()
                    self.update_steps += 1
                    self._fitted = True
                    with torch.no_grad():
                        pred = torch.stack([h.argmax(1) for h in self._split_logits(logits)], 1)
                        right += int(torch.all(pred == target, dim=1).sum())
                    seen += len(b)
                    loss_sum += loss.item() * len(b)
                if cancelled:
                    break
                completed += 1
                self.trained_epochs += 1
                last = dict(epoch=self.trained_epochs, loss=loss_sum / max(seen, 1),
                            accuracy=right / max(seen, 1), lr=learning_rate)
                if val_idx is not None and len(val_idx):
                    quick = self._evaluate(val_data, val_idx, batch_size)
                    m = self._metrics(quick)
                    last.update(val_loss=m["loss"], val_accuracy=m["accuracy"],
                                val_assignment_fidelity=m["assignment_fidelity"],
                                val_macro_f1=m["macro_f1"], val_xeb=m["xeb_labels"],
                                val_xeb_linear=m["xeb_linear"])
                if callback is not None:
                    callback(dict(last))
        finally:
            self.optimizer_state = opt.state_dict()
            self.model.eval()
        return dict(cancelled=cancelled, epochs_completed=completed, total_epochs=self.trained_epochs,
                    last=last, warnings=warnings, excluded=excluded_train)

    @staticmethod
    def _comparable(config):
        keys = ("levels", "hidden_layers", "atoms", "mode", "per_atom_networks", "activation", "filter_config")
        return {k: config.get(k) for k in keys}

    def _usable(self, data, indices, config):
        """Отбросить измерения, где метка выше числа уровней атома.

        Пример: атом размечен как |2⟩, а классификация ведётся как для
        кубита (2 уровня) — такая трасса не может быть корректно обучена.
        Возвращает (оставшиеся индексы, число исключённых).
        """
        if data["labels"] is None:
            return indices, 0
        lab = data["labels"][indices][:, config["atoms"]]
        ok = np.all(lab < np.asarray(config["levels"])[None, :], axis=1)
        return indices[ok], int((~ok).sum())

    @staticmethod
    def _cancelled(event) -> bool:
        return event is not None and event.is_set()

    @staticmethod
    def _raise_if_cancelled(event):
        if event is not None and event.is_set():
            raise InterruptedError("Операция остановлена пользователем.")

    @staticmethod
    def _indices(data, indices) -> np.ndarray:
        n = len(data["iq"])
        if indices is None:
            return np.arange(n)
        indices = np.asarray(indices, dtype=np.int64).ravel()
        if indices.size and (indices.min() < 0 or indices.max() >= n):
            raise ValueError("Номера измерений выходят за пределы набора данных.")
        return indices

    def _require_fitted(self):
        if not self.is_fitted:
            raise ValueError("Сначала обучите сеть или загрузите веса.")

    def _check_compatible(self, data) -> dict:
        """Проверить, что данные подходят к текущей обученной модели."""
        data = self._check_data(data)
        if self.input_samples is not None and data["iq"].shape[2] != self.input_samples:
            raise ValueError(f"Модель обучена на трассах из {self.input_samples} отсчётов, "
                             f"а в данных {data['iq'].shape[2]}.")
        if max(self.config["atoms"]) >= data["iq"].shape[1]:
            raise ValueError(f"Модель использует атом {max(self.config['atoms'])}, в данных {data['iq'].shape[1]} атомов.")
        grid = self.filters.get("time_ns")
        if grid is not None and data["time_ns"] is not None and not np.allclose(grid, data["time_ns"], rtol=1e-6):
            raise ValueError("Временная сетка данных отличается от сетки обучения — фильтры неприменимы.")
        return data

    # ------------------------------------------------------------------
    # Классификация и метрики
    # ------------------------------------------------------------------
    def _evaluate(self, data, indices, batch_size, keep=False) -> dict:
        """Прогнать сеть по данным: confusion-матрицы, loss и (опционально) выходы."""
        levels = self.config["levels"]
        atoms = self.config["atoms"]
        conf = [np.zeros((k, k), np.int64) for k in levels]
        loss_sum, right, n = 0.0, 0, 0
        xeb_sum, xeb_n = 0.0, 0
        p_ideal = data.get("ideal_probs")
        d = math.prod(levels)
        pred_all = np.empty((len(indices), len(levels)), np.int64) if keep else None
        prob_all = [np.empty((len(indices), k), np.float32) for k in levels] if keep else None
        pos = 0
        self.model.eval()
        with torch.inference_mode():
            for b in self._batches(indices, batch_size):
                logits = self._logits(self._tensor(data["iq"][b]))
                heads = self._split_logits(logits)
                pred = torch.stack([h.argmax(1) for h in heads], 1).cpu().numpy()
                if keep:
                    pred_all[pos:pos + len(b)] = pred
                    for a, h in enumerate(heads):
                        prob_all[a][pos:pos + len(b)] = h.softmax(1).cpu().numpy()
                if data["labels"] is not None:
                    y = data["labels"][b][:, atoms]
                    y_t = torch.as_tensor(np.minimum(y, np.asarray(levels) - 1), device=self.device)
                    loss_sum += float(self._loss(logits, y_t)) * len(b)
                    right += int(np.all(pred == y, axis=1).sum())
                    for a, k in enumerate(levels):
                        valid = y[:, a] < k
                        np.add.at(conf[a], (y[valid, a], pred[valid, a]), 1)
                if p_ideal is not None and d <= 2 ** 62 and p_ideal.shape[-1] == d:
                    flat = np.ravel_multi_index(pred.T, levels)
                    sel = p_ideal[flat] if p_ideal.ndim == 1 else p_ideal[b, flat]
                    xeb_sum += float(sel.sum())
                    xeb_n += len(b)
                n += len(b)
                pos += len(b)
        return dict(confusion=conf, loss_sum=loss_sum, right=right, n=n,
                    xeb_sum=xeb_sum, xeb_n=xeb_n, predictions=pred_all, probabilities=prob_all,
                    has_labels=data["labels"] is not None)

    def _metrics(self, ev) -> dict:
        """Вычислить метрики по результату :meth:`_evaluate`.

        * accuracy (= score) — доля измерений, где ВСЕ атомы распознаны верно;
        * assignment_fidelity атома — среднее по уровням P(верно | подготовлен
          уровень); регистра — геометрическое среднее F_nQ (как в статье);
        * macro_f1 — среднее F1 по уровням и атомам;
        * xeb_labels — нормированный XEB по меткам подготовки (дельта-
          распределения): (D·acc − 1)/(D − 1); 0 — случайное угадывание,
          1 — идеальное считывание. Для базисных состояний это аффинное
          преобразование accuracy, а не независимая физическая величина;
        * xeb_linear — linear XEB D·⟨p_ideal(ŝ)⟩ − 1, только если в данных
          есть ideal_probs (идеальные распределения схем).
        """
        levels = self.config["levels"]
        d = math.prod(levels)
        out = dict(loss=None, accuracy=None, assignment_fidelity=None, macro_f1=None,
                   xeb_labels=None, xeb_linear=None, per_atom=[])
        # 1/D через логарифмы: D = Πk может быть больше максимального float.
        inv_d = math.exp(-sum(math.log(k) for k in levels))
        if ev["xeb_n"]:
            out["xeb_linear"] = d * ev["xeb_sum"] / ev["xeb_n"] - 1.0
        if not ev["has_labels"] or not ev["n"]:
            return out
        out["loss"] = ev["loss_sum"] / ev["n"]
        out["accuracy"] = ev["right"] / ev["n"]
        out["xeb_labels"] = (out["accuracy"] - inv_d) / (1 - inv_d) if d > 1 else None
        fids, f1s = [], []
        for a, cm in enumerate(ev["confusion"]):
            support = cm.sum(1)
            recall = np.divide(np.diag(cm), support, out=np.zeros(len(cm)), where=support > 0)
            col = cm.sum(0)
            precision = np.divide(np.diag(cm), col, out=np.zeros(len(cm)), where=col > 0)
            f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(cm)),
                           where=(precision + recall) > 0)
            fid = float(recall[support > 0].mean()) if np.all(support > 0) else None
            down = [float(cm[i, :i].sum() / support[i]) if support[i] else None for i in range(len(cm))]
            up = [float(cm[i, i + 1:].sum() / support[i]) if support[i] else None for i in range(len(cm))]
            out["per_atom"].append(dict(atom=self.config["atoms"][a], confusion=cm, support=support,
                                        accuracy=float(np.trace(cm) / max(cm.sum(), 1)),
                                        assignment_fidelity=fid, macro_f1=float(f1[support > 0].mean()) if np.any(support > 0) else None,
                                        decay_error=down, excitation_error=up))
            fids.append(fid)
            if np.any(support > 0):
                f1s.append(float(f1[support > 0].mean()))
        if all(f is not None for f in fids):
            out["assignment_fidelity"] = 0.0 if min(fids) == 0 else float(np.exp(np.mean(np.log(fids))))
        out["macro_f1"] = float(np.mean(f1s)) if f1s else None
        return out

    def infer(self, data, indices=None, *, batch_size=4096) -> dict:
        """Классифицировать новые данные (метки не нужны), веса не меняются.

        :returns: ``{"predictions" [M, A_used], "probabilities" (список [M, k_a]),
            "spots" [M, A_used, 2] (MTV I, Q), "register_state" [M] (номер
            состояния регистра, атом 0 — старший разряд), "shot_ids" [M],
            "indices" [M], "atoms", "xeb_linear"}``.
        """
        self._require_fitted()
        data = self._check_compatible(data)
        indices = self._indices(data, indices)
        ev = self._evaluate(dict(data, labels=None), indices, batch_size, keep=True)
        spots = self._spots(data, indices, batch_size)
        m = self._metrics(ev)
        return dict(predictions=ev["predictions"], probabilities=ev["probabilities"], spots=spots,
                    register_state=self.register_state(ev["predictions"]),
                    shot_ids=data["shot_ids"][indices], indices=indices, atoms=list(self.config["atoms"]),
                    xeb_linear=m["xeb_linear"], warnings=self._data_warnings(data))

    def register_state(self, predictions: np.ndarray) -> np.ndarray:
        """Номер состояния регистра (атом 0 — старший разряд, основания = levels).

        Для больших регистров (Πk > 2⁶²) номер не помещается в int64 и
        возвращается массивом Python-целых (dtype=object).
        """
        levels = self.config["levels"]
        if math.prod(levels) <= 2 ** 62:
            return np.ravel_multi_index(predictions.T, levels)
        out = np.empty(len(predictions), dtype=object)
        for i, row in enumerate(predictions):
            value = 0
            for digit, base in zip(row, levels):
                value = value * base + int(digit)
            out[i] = value
        return out

    def _spots(self, data, indices, batch_size) -> np.ndarray:
        """MTV исходных трасс выбранных атомов (после окна, без свёртки)."""
        window = self.config["filter_config"]["window"]
        out = np.empty((len(indices), len(self.config["atoms"]), 2), np.float32)
        pos = 0
        for b in self._batches(indices, batch_size):
            x = data["iq"][b][:, self.config["atoms"]]
            if window is not None:
                x = x[:, :, window[0]:window[1]]
            m = self.mean_trace_value(x)
            out[pos:pos + len(b), :, 0], out[pos:pos + len(b), :, 1] = m.real, m.imag
            pos += len(b)
        return out

    def validate(self, data, indices=None, *, batch_size=4096, f01_ghz=None) -> dict:
        """Валидировать модель на размеченных данных и построить бенчмарки.

        Возвращает всё, что нужно для графиков режима «Валидация»:

        * ``metrics`` — accuracy/score, assignment fidelity F_nQ, macro-F1,
          loss, XEB (по меткам и linear по ideal_probs), а также accuracy
          бенчмарков ``centroid``, ``lda``, ``qda`` (на MTV);
        * ``per_atom`` — для каждого атома: матрица ошибок, fidelity, ошибки
          распада ``decay_error[i] = P(предсказан уровень < i | подготовлен i)``
          и возбуждения ``excitation_error[i]``, бенчмарки, разделение пятен
          (SNR и оценка ошибки перекрытия для каждой пары уровней), оценки
          переходов и термальной населённости (см. :meth:`_physics`);
        * ``predictions``, ``probabilities``, ``spots`` — для графиков.

        :param f01_ghz: частоты перехода 0→1 атомов (ГГц), список или число;
            если заданы, по термальной населённости оценивается эффективная
            температура T_eff = h·f01 / (k_B·ln((1−p)/p)). Без частоты
            температура не вычисляется (из IQ-трасс её не определить).
        """
        self._require_fitted()
        data = self._check_compatible(data)
        if data["labels"] is None:
            raise ValueError("Для валидации нужны истинные метки. Для данных без меток используйте классификацию.")
        indices, excluded = self._usable(data, self._indices(data, indices), self.config)
        if not len(indices):
            raise ValueError("Нет измерений с метками в пределах заданного числа уровней.")
        ev = self._evaluate(data, indices, batch_size, keep=True)
        result = self._metrics(ev)
        spots = self._spots(data, indices, batch_size)
        # Бенчмарки и SNR считаются на MTV ПОСЛЕ того же окна и свёртки, что
        # и при калибровке центроидов/LDA/QDA (spots без свёртки — для рисунка).
        prepared_mtv, early_mtv = self._prepared_mtv(data, indices, batch_size)
        y = data["labels"][indices][:, self.config["atoms"]]
        # Физика (переходы, термальная доля) — по подготовленным уровням, а не
        # по псевдометкам утечки, если окно их применило.
        prepared = data.get("prepared_labels")
        y_prepared = y if prepared is None else prepared[indices][:, self.config["atoms"]]
        n_atoms = len(self.config["levels"])
        freqs = [None] * n_atoms
        if f01_ghz is not None:
            if isinstance(f01_ghz, (list, tuple, np.ndarray)):
                freqs = [float(v) for v in f01_ghz]
                if not all(math.isfinite(v) and v > 0 for v in freqs):
                    raise ValueError("Частоты f01 должны быть конечными положительными числами (ГГц).")
                if len(freqs) == 1:
                    freqs = freqs * n_atoms
                if len(freqs) != n_atoms:
                    raise ValueError(f"Укажите одну частоту f01 для всех атомов или ровно {n_atoms} частот.")
            else:
                if not (math.isfinite(float(f01_ghz)) and float(f01_ghz) > 0):
                    raise ValueError("Частота f01 должна быть конечным положительным числом (ГГц).")
                freqs = [float(f01_ghz)] * n_atoms
        bench_right = {"centroid": np.ones(len(indices), bool), "lda": np.ones(len(indices), bool),
                       "qda": np.ones(len(indices), bool)}
        for a, entry in enumerate(result["per_atom"]):
            pts = prepared_mtv[:, a]
            bench = self._benchmarks(a, pts)
            entry["benchmarks"] = {}
            for name, pred in bench.items():
                ok = pred == y[:, a]
                bench_right[name] &= ok
                entry["benchmarks"][name] = float(ok.mean())
            entry["separation"] = self._separation(pts, y[:, a], self.config["levels"][a])
            early_sep = self._separation(early_mtv[:, a], y[:, a], self.config["levels"][a])
            entry.update(self._physics(data, indices, a, y_prepared[:, a], freqs[a], early_sep, batch_size))
        result["benchmarks"] = {name: float(ok.mean()) for name, ok in bench_right.items()}
        result.update(predictions=ev["predictions"], probabilities=ev["probabilities"], spots=spots,
                      labels=y, indices=indices, shot_ids=data["shot_ids"][indices],
                      atoms=list(self.config["atoms"]), excluded=excluded,
                      warnings=self._data_warnings(data))
        return result

    def _prepared_mtv(self, data, indices, batch_size):
        """MTV после окна и свёртки: (по всей трассе, по ранней трети) [M, A_used]."""
        n = len(self.config["levels"])
        full = np.empty((len(indices), n), np.complex64)
        early = np.empty((len(indices), n), np.complex64)
        pos = 0
        for b in self._batches(indices, batch_size):
            x = self._prepare(data["iq"][b])
            third = max(1, x.shape[-1] // 3)
            full[pos:pos + len(b)] = x.mean(-1)
            early[pos:pos + len(b)] = x[:, :, :third].mean(-1)
            pos += len(b)
        return full, early

    def _data_warnings(self, data) -> list[str]:
        """Предупреждения калибровки модели и несовпадения временной сетки."""
        warnings = list(self.filters.get("warnings", []))
        if self.filters.get("time_ns") is not None and data.get("time_ns") is None:
            warnings.append("Модель обучена на данных с time_ns, а в этих данных временной сетки нет: "
                            "совпадение шага дискретизации не проверено.")
        return warnings

    def _benchmarks(self, a, pts) -> dict:
        """Предсказания классических классификаторов на MTV: центроид, LDA, QDA."""
        p = np.stack([pts.real, pts.imag], axis=1).astype(float)
        cent = self.filters["centroids"][a]
        out = {"centroid": np.abs(pts[:, None] - cent[None]).argmin(1)}
        lda, qda = self.filters["lda"][a], self.filters["qda"][a]
        inv = np.linalg.pinv(lda["cov"])
        score = []
        for s, mu in enumerate(lda["means"]):
            score.append(p @ inv @ mu - 0.5 * mu @ inv @ mu + np.log(max(lda["priors"][s], 1e-12)))
        out["lda"] = np.argmax(np.stack(score, 1), 1)
        score = []
        for s, (mu, cov) in enumerate(zip(qda["means"], qda["covs"])):
            inv_s = np.linalg.pinv(cov)
            diff = p - mu
            _, logdet = np.linalg.slogdet(cov)
            score.append(-0.5 * np.einsum("ni,ij,nj->n", diff, inv_s, diff) - 0.5 * logdet
                         + np.log(max(qda["priors"][s], 1e-12)))
        out["qda"] = np.argmax(np.stack(score, 1), 1)
        return out

    @staticmethod
    def _separation(pts, y, k) -> list[dict]:
        """Разделение «пятен» каждой пары уровней на плоскости IQ.

        MTV проецируются на прямую через центры двух облаков. SNR =
        |m_j − m_i| / ((σ_i + σ_j)/2). Ошибка перекрытия двух гауссовых
        облаков при пороге посередине и равных априорных вероятностях:
        ½[Φ(−Δ/(2σ_i)) + Φ(−Δ/(2σ_j))] (при σ_i = σ_j это Φ(−SNR/2)).
        """
        out = []
        for i, j in itertools.combinations(range(k), 2):
            a, b = pts[y == i], pts[y == j]
            if len(a) < 2 or len(b) < 2:
                out.append(dict(pair=(i, j), snr=None, overlap_error=None))
                continue
            axis = b.mean() - a.mean()
            if abs(axis) == 0:
                out.append(dict(pair=(i, j), snr=0.0, overlap_error=0.5))
                continue
            u = axis / abs(axis)
            pa, pb = (a * np.conj(u)).real, (b * np.conj(u)).real
            gap = abs(pb.mean() - pa.mean())
            sa, sb = max(pa.std(), 1e-30), max(pb.std(), 1e-30)
            snr = gap / ((sa + sb) / 2)
            overlap = 0.5 * (ndtr(-gap / (2 * sa)) + ndtr(-gap / (2 * sb)))
            out.append(dict(pair=(i, j), snr=float(snr), overlap_error=float(overlap)))
        return out

    def _physics(self, data, indices, a, y, f01_ghz, separation, batch_size) -> dict:
        """Оценки ошибок распада и «температуры» по трассам атома a.

        * ``transition_down[i]`` / ``transition_up[i]`` — доля трасс,
          подготовленных в i, у которых ранняя треть трассы у центра i, а
          поздняя — у центра ниже/выше i: переход во время считывания
          (распад T1 / возбуждение измерением). Эвристика, а не измерение T1.
        * ``thermal_population`` — НАБЛЮДАЕМАЯ доля трасс, подготовленных в
          |0⟩, у которых уже ранняя треть ближе к |1⟩ или выше (диагностика).
          Наблюдаемая доля = p₁(1−ε₁₀) + (1−p₁)ε₀₁, где ε — ошибки назначения
          в раннем окне; без калибровки ε она не равна термальной населённости
          p₁ и в общем случае не является ни верхней, ни нижней её оценкой.
        * ``thermal_corrected`` — та же доля минус оценка перекрытия облаков
          Φ(−SNR₀₁/2), посчитанная по MTV ТОЙ ЖЕ ранней трети (не меньше 0).
        * ``effective_temperature_mK`` — T_eff = h·f01/(k_B·ln(p0/p1)) по
          наблюдаемым долям уровней 0 и 1 в начале |0⟩-трасс (p1 минус
          перекрытие облаков) при заданной f01, иначе None. Это оценка без
          коррекции ошибок подготовки и считывания.
        Все величины включают SPAM; это диагностика для сравнения, а не замена
        отдельной калибровки T1 и температуры.
        """
        k = self.config["levels"][a]
        means = self.filters["means"][a]
        third = max(1, means.shape[-1] // 3)
        early_ref, late_ref = means[:, :third].mean(-1), means[:, -third:].mean(-1)
        early = np.empty(len(indices), np.int64)
        late = np.empty(len(indices), np.int64)
        pos = 0
        for b in self._batches(indices, batch_size):
            tr = self._prepare(data["iq"][b])[:, a]
            early[pos:pos + len(b)] = np.abs(tr[:, :third].mean(-1)[:, None] - early_ref).argmin(1)
            late[pos:pos + len(b)] = np.abs(tr[:, -third:].mean(-1)[:, None] - late_ref).argmin(1)
            pos += len(b)
        down, up = [], []
        for i in range(k):
            ok = (y == i) & (early == i)
            down.append(float(np.mean(late[ok] < i)) if ok.any() else None)
            up.append(float(np.mean(late[ok] > i)) if ok.any() else None)
        zero = y == 0
        p_th = float(np.mean(early[zero] > 0)) if zero.any() else None
        overlap = separation[0]["overlap_error"] if separation and separation[0]["pair"] == (0, 1) else None
        corrected = None if p_th is None or overlap is None else max(p_th - overlap, 0.0)
        t_eff = None
        if f01_ghz is not None and zero.any() and overlap is not None:
            # Больцман для пары уровней 0 и 1: p1/p0 = exp(−h·f01/(k_B·T)). Берётся
            # именно отношение p1/p0 (а не сумма всех верхних уровней — для кудита
            # это другая величина); p1 поправлено на перекрытие облаков 0|1.
            p1 = max(float(np.mean(early[zero] == 1)) - overlap, 0.0)
            p0 = float(np.mean(early[zero] == 0)) + overlap
            if 0 < p1 < p0:
                h, k_b = 6.62607015e-34, 1.380649e-23
                t_eff = h * float(f01_ghz) * 1e9 / (k_b * math.log(p0 / p1)) * 1e3
        return dict(transition_down=down, transition_up=up, thermal_population=p_th,
                    thermal_corrected=corrected, effective_temperature_mK=t_eff)

    def classify_and_plot(self, data, atoms=None, *, max_points=5000, color_by="prediction"):
        """Классифицировать набор и построить «пятна» IQ (MTV) выбранных атомов.

        Если в данных есть метки, выполняется :meth:`validate`, иначе
        :meth:`infer`. Рисунок (matplotlib Figure) возвращается вызывающему
        коду и в объекте не хранится. Одновременно — не более пяти атомов;
        на рисунок попадает воспроизводимая подвыборка ≤ max_points точек,
        метрики считаются по всем данным.

        :param atoms: номера каналов (из использованных моделью); None — первые 5.
        :param color_by: ``prediction`` — цвет по предсказанию, ``label`` — по метке.
        :returns: (результат validate/infer, Figure).
        """
        from matplotlib.figure import Figure
        data = self._check_data(data)
        result = self.validate(data) if data["labels"] is not None else self.infer(data)
        used = self.config["atoms"]
        atoms = used[:5] if atoms is None else list(atoms)
        if not atoms or len(atoms) > 5 or any(a not in used for a in atoms):
            raise ValueError("Выберите от 1 до 5 атомов, используемых моделью.")
        n = len(result["predictions"])
        show = np.arange(n) if n <= max_points else np.sort(np.random.default_rng(0).choice(n, max_points, replace=False))
        fig = Figure(figsize=(4.2 * len(atoms), 4), constrained_layout=True)
        for i, channel in enumerate(atoms, 1):
            a = used.index(channel)
            ax = fig.add_subplot(1, len(atoms), i)
            colors = result["labels"][show, a] if color_by == "label" and "labels" in result else result["predictions"][show, a]
            for s in np.unique(colors):
                sel = show[colors == s]
                ax.scatter(result["spots"][sel, a, 0], result["spots"][sel, a, 1], s=5, alpha=0.5, label=f"|{s}⟩")
            ax.set(xlabel="I (MTV)", ylabel="Q (MTV)", title=f"Атом {channel}")
            ax.legend(markerscale=3, fontsize=8)
        return result, fig

    # ------------------------------------------------------------------
    # Экспорт / загрузка весов
    # ------------------------------------------------------------------
    @staticmethod
    def _pack(value):
        """Привести numpy-объекты к тензорам и простым типам (без pickle-классов)."""
        if isinstance(value, np.ndarray):
            if np.iscomplexobj(value):
                return {"__complex__": True, "re": torch.from_numpy(np.ascontiguousarray(value.real)),
                        "im": torch.from_numpy(np.ascontiguousarray(value.imag))}
            return torch.from_numpy(np.ascontiguousarray(value))
        if isinstance(value, dict):
            return {str(k): NeuralService._pack(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [NeuralService._pack(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _unpack(value):
        if isinstance(value, torch.Tensor):
            return value.numpy()
        if isinstance(value, dict):
            if value.get("__complex__") is True:
                return value["re"].numpy() + 1j * value["im"].numpy()
            return {k: NeuralService._unpack(v) for k, v in value.items()}
        if isinstance(value, list):
            return [NeuralService._unpack(v) for v in value]
        return value

    def export_weights(self, path, *, include_optimizer=True, metadata=None) -> str:
        """Записать ТЕКУЩЕЕ состояние сети в файл ``path`` (.pt).

        Файл содержит не только веса, но и всё, без чего они бессмысленны:
        конфигурацию, откалиброванные фильтры, нормировку признаков, форму
        входа и (по желанию) состояние оптимизатора для дообучения. Каталог
        и имя выбирает MainWindow; сам сервис ничего автоматически не пишет.

        Запись идёт во временный файл рядом с целевым и затем атомарно
        переименовывается — недописанный файл весов не появится.

        :param metadata: произвольный словарь простых значений (дата, метрики,
            источник данных) — сохраняется как есть.
        :returns: путь к записанному файлу.
        """
        self._require_fitted()
        path = Path(path)
        if not path.parent.is_dir():
            raise ValueError(f"Каталог не существует: {path.parent}")
        bundle = dict(format=self.BUNDLE_FORMAT, version=self.BUNDLE_VERSION,
                      config=copy.deepcopy(self.config), input_channels=self.input_channels,
                      input_samples=self.input_samples, feature_count=self.feature_count,
                      trained_epochs=self.trained_epochs,
                      normalizer=self._pack(self.normalizer), filters=self._pack(self.filters),
                      model_state={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                      optimizer_name=self.optimizer_name,
                      optimizer_state=self.optimizer_state if include_optimizer else None,
                      metadata=self._pack(metadata or {}))
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(bundle, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return str(path)

    def load_weights(self, path) -> dict:
        """Загрузить файл весов и сделать его текущей моделью.

        Используется безопасная загрузка ``torch.load(weights_only=True)``:
        файл может содержать только тензоры и простые контейнеры, поэтому
        чужой .pt не выполнит произвольный код. Сначала строится и
        проверяется кандидат; текущая модель заменяется только при успехе.
        После загрузки можно классифицировать данные или дообучать сеть
        (:meth:`train` с ``fine_tune=True``).

        :returns: ``{"config": ..., "metadata": ..., "trained_epochs": ...}``.
        :raises ValueError: если файл повреждён, не того формата или веса не
            соответствуют архитектуре.
        """
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Файл весов не найден: {path}")
        try:
            bundle = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 — любой сбой чтения = неверный файл
            raise ValueError(f"Не удалось прочитать файл весов: {exc}") from exc
        if not isinstance(bundle, dict) or bundle.get("format") != self.BUNDLE_FORMAT:
            raise ValueError("Это не файл весов Q-Net Lab.")
        if bundle.get("version") != self.BUNDLE_VERSION:
            raise ValueError(f"Версия файла весов {bundle.get('version')} не поддерживается (нужна {self.BUNDLE_VERSION}).")
        try:
            cfg = bundle["config"]
            probe = NeuralService.__new__(NeuralService)
            probe.config, probe.model, probe._fitted = {}, None, False
            config = NeuralService.configure(probe, cfg["levels"], cfg["hidden_layers"], atoms=cfg["atoms"],
                                             mode=cfg["mode"], per_atom_networks=cfg["per_atom_networks"],
                                             activation=cfg["activation"], dropout=cfg["dropout"],
                                             filter_config=cfg["filter_config"], seed=cfg["seed"])
            feature_count = self._int(bundle["feature_count"], "feature_count", 1)
            normalizer = self._unpack(bundle["normalizer"])
            filters = self._unpack(bundle["filters"])
            for key in ("mean", "scale"):
                if np.asarray(normalizer[key]).shape != (feature_count,) or not np.isfinite(normalizer[key]).all():
                    raise ValueError("Повреждены параметры нормировки.")
            if np.any(np.asarray(normalizer["scale"]) <= 0):
                raise ValueError("Масштаб нормировки должен быть положительным.")
            samples = self._int(bundle["input_samples"], "input_samples", 1)
            k_atoms = len(config["levels"])
            for key in ("means", "centroids", "lda", "qda"):
                if len(filters.get(key, [])) != k_atoms:
                    raise ValueError(f"В файле нет калибровки '{key}' для всех атомов.")
            if config["mode"] == "matched":
                if len(filters["kernels"]) != feature_count or len(filters["descriptions"]) != feature_count:
                    raise ValueError("Число фильтров не совпадает с размером входа сети.")
                if len(filters.get("kernel_atoms", [])) != feature_count:
                    raise ValueError("Число номеров атомов фильтров не совпадает с числом фильтров.")
                for atom, d, kernel in zip(filters["kernel_atoms"], filters["descriptions"], filters["kernels"]):
                    if not (isinstance(d, dict) and {"kind", "source", "target", "available"} <= set(d)
                            and isinstance(atom, int) and 0 <= atom < k_atoms and d.get("atom") == atom):
                        raise ValueError("Повреждено описание фильтра.")
                    if not np.isfinite(np.asarray(kernel)).all():
                        raise ValueError("Ядро фильтра содержит NaN/Inf.")
            for a, k in enumerate(config["levels"]):
                lda, qda = filters["lda"][a], filters["qda"][a]
                arrays = [filters["centroids"][a], filters["means"][a], lda["means"], lda["cov"], lda["priors"],
                          qda["means"], qda["covs"], qda["priors"]]
                if not all(np.isfinite(np.asarray(v)).all() for v in arrays):
                    raise ValueError("Калибровка центроидов/LDA/QDA содержит NaN/Inf.")
                if (np.asarray(filters["centroids"][a]).shape != (k,) or np.asarray(lda["means"]).shape != (k, 2)
                        or np.asarray(lda["cov"]).shape != (2, 2) or np.asarray(qda["covs"]).shape != (k, 2, 2)
                        or np.asarray(qda["means"]).shape != (k, 2) or np.asarray(lda["priors"]).shape != (k,)
                        or np.asarray(qda["priors"]).shape != (k,)):
                    raise ValueError("Форма калибровки центроидов/LDA/QDA не соответствует уровням.")
            window = config["filter_config"]["window"]
            length = (window[1] - window[0]) if window else samples
            if window and window[1] > samples:
                raise ValueError("Окно фильтров выходит за пределы трасс модели.")
            for a, k in enumerate(config["levels"]):
                if np.asarray(filters["means"][a]).shape != (k, length):
                    raise ValueError("Средние трассы в файле не соответствуют окну и уровням.")
            if config["mode"] == "matched":
                if any(np.asarray(kr).shape != (length,) for kr in filters["kernels"]):
                    raise ValueError("Длина ядер фильтров не соответствует окну.")
            elif feature_count != 2 * len(config["atoms"]) * length:
                raise ValueError("Размер входа FNN не соответствует форме трасс.")
            model = self._build_model(config, feature_count)
            model.load_state_dict(bundle["model_state"], strict=True)
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Веса содержат NaN/Inf.")
            trained_epochs = self._int(bundle.get("trained_epochs", 0), "trained_epochs", 0)
            channels = bundle.get("input_channels")
            if channels is not None:
                channels = self._int(channels, "input_channels", 1)
            metadata = self._unpack(bundle.get("metadata", {}))
            if not isinstance(metadata, dict):
                raise ValueError("Метаданные файла весов повреждены.")
            optimizer_name = bundle.get("optimizer_name")
            if optimizer_name is not None and optimizer_name not in self.OPTIMIZERS:
                raise ValueError("Неизвестный оптимизатор в файле весов.")
        except (KeyError, TypeError, RuntimeError, IndexError, AttributeError) as exc:
            raise ValueError(f"Файл весов не соответствует архитектуре или повреждён: {exc}") from exc
        # --- замена текущего состояния: только после ВСЕХ проверок ---
        self.config, self.model, self.filters = config, model, filters
        self.normalizer = {k: np.asarray(v, np.float32) for k, v in normalizer.items()}
        self.feature_count, self.input_samples, self.input_channels = feature_count, samples, channels
        self.trained_epochs = trained_epochs
        self.optimizer_name = optimizer_name
        self.optimizer_state = bundle.get("optimizer_state")
        self._fitted = True
        self.model.eval()
        return dict(config=copy.deepcopy(config), metadata=metadata, trained_epochs=trained_epochs)

    def describe(self) -> dict:
        """Краткое описание текущей модели для интерфейса и отчётов."""
        info = dict(fitted=self.is_fitted, config=copy.deepcopy(self.config), epochs=self.trained_epochs,
                    feature_count=self.feature_count, parameters=None, hidden=None)
        if self.model is not None:
            info["parameters"] = int(sum(p.numel() for p in self.model.parameters()))
            info["hidden"] = [m.out_features for m in self.model[0] if isinstance(m, nn.Linear)][:-1]
        return info
