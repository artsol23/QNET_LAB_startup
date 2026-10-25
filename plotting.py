"""Построение графиков по готовым данным.

Модуль не зависит от Qt, MainWindow и NeuralService.
Вызовы рисования выполняются в главном потоке приложения.
"""

from __future__ import annotations

import math

import matplotlib
import numpy as np


LEVEL_COLORS = [
    "#2563b8",
    "#e08a1e",
    "#c43c3c",
    "#2e9e5b",
    "#7a3e9d",
    "#7f7f7f",
]


def grid_shape(count: int) -> tuple[int, int]:
    count = max(count, 1)
    columns = 1 if count == 1 else 2 if count <= 4 else 3
    return math.ceil(count / columns), columns


def draw_message(figure, text: str) -> None:
    figure.clear()
    ax = figure.add_subplot(111)
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        wrap=True,
        color="#5b6b82",
        fontsize=12,
    )
    ax.set_axis_off()


def finish_figure(figure) -> None:
    """Общая легенда над сеткой графиков."""
    handles = []
    labels = []

    for ax in figure.axes:
        axis_handles, axis_labels = ax.get_legend_handles_labels()

        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels and not label.startswith("_"):
                handles.append(handle)
                labels.append(label)

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    top = 1.0

    if handles:
        columns = min(len(labels), 8)
        rows = math.ceil(len(labels) / columns)

        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=columns,
            fontsize=8,
            frameon=False,
            markerscale=2,
        )
        top = 1.0 - 0.06 * rows

    figure.tight_layout(rect=(0, 0, 1, top))


def draw_heatmap(ax, entry: dict, compact: bool = False) -> None:
    """Матрица ошибок: доли по строкам и число измерений."""
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    confusion = np.asarray(entry["confusion"], dtype=float)
    normalized = confusion / np.maximum(
        confusion.sum(axis=1, keepdims=True),
        1,
    )

    cell_format = "{:.2f}" if compact else "{:.2f}\n({})"

    annotations = np.array([
        [
            cell_format.format(
                normalized[i, j],
                int(confusion[i, j]),
            )
            for j in range(len(confusion))
        ]
        for i in range(len(confusion))
    ])

    font_size = 7 if compact else 8

    if sns is not None:
        sns.heatmap(
            normalized,
            ax=ax,
            annot=annotations,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            annot_kws={"fontsize": font_size},
        )
    else:
        ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)

        for i in range(len(confusion)):
            for j in range(len(confusion)):
                ax.text(
                    j,
                    i,
                    annotations[i, j],
                    ha="center",
                    va="center",
                    fontsize=font_size,
                )

    ax.set_xlabel("предсказание", fontsize=8 if compact else 10)
    ax.set_ylabel("подготовлено", fontsize=8 if compact else 10)
    ax.set_title(
        f"Атом {entry['atom']}"
        + ("" if compact else ": матрица ошибок"),
        fontsize=10,
    )


def draw_confusions(ax, result, selected_atoms) -> None:
    """Матрица ошибок первого выбранного атома."""
    if not result or not result.get("per_atom"):
        ax.text(
            0.5,
            0.5,
            "матрица появится после обучения",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    entries = [
        entry
        for entry in result["per_atom"]
        if entry["atom"] in selected_atoms
    ] or result["per_atom"]

    draw_heatmap(ax, entries[0])


def draw_training(
    figure,
    *,
    history,
    evaluation,
    chart_keys,
    chart_specs,
    selected_atoms,
) -> None:
    """Графики обучения по истории метрик."""
    if not chart_keys:
        draw_message(
            figure,
            "Отметьте хотя бы один график сверху.",
        )
        return

    if not history and evaluation is None:
        draw_message(
            figure,
            "Графики появятся после первой эпохи обучения\n"
            "или после загрузки журнала обучения.",
        )
        return

    figure.clear()
    rows, columns = grid_shape(len(chart_keys))
    epochs = [row["epoch"] for row in history]

    for position, key in enumerate(chart_keys, 1):
        ax = figure.add_subplot(rows, columns, position)

        title, train_key, validation_key = chart_specs[key]
        ax.set_title(title, fontsize=10)

        if key == "confusion":
            draw_confusions(ax, evaluation, selected_atoms)
            continue

        if key == "xeb":
            if any(
                row.get("val_xeb_linear") is not None
                for row in history
            ):
                validation_key = "val_xeb_linear"
                ax.set_title(
                    "Linear XEB (по ideal_probs)",
                    fontsize=10,
                )
            else:
                ax.set_title(
                    "XEB по меткам: (D·acc−1)/(D−1)\n"
                    "(accuracy с поправкой на угадывание)",
                    fontsize=9,
                )

        drawn = False

        series = (
            (train_key, "train", "#2563b8"),
            (validation_key, "validation", "#e08a1e"),
        )

        for field, label, color in series:
            if field is None:
                continue

            values = [
                np.nan if row.get(field) is None else row[field]
                for row in history
            ]

            if not np.all(np.isnan(values)):
                ax.plot(
                    epochs,
                    values,
                    label=label,
                    color=color,
                    lw=1.6,
                    marker="o" if len(epochs) < 30 else None,
                    ms=3,
                )
                drawn = True

        if drawn:
            ax.set_xlabel("эпоха")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, frameon=False)
        else:
            ax.text(
                0.5,
                0.5,
                "нет данных (нужна validation)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    figure.tight_layout()


def draw_spots(
    ax,
    result,
    atom_index: int,
    channel: int,
    *,
    color_by: str = "prediction",
) -> None:
    """Пятна IQ; ошибочные предсказания отмечаются крестиком."""
    count = len(result["predictions"])

    if count <= 5000:
        shown = np.arange(count)
    else:
        shown = np.sort(
            np.random.default_rng(0).choice(
                count,
                5000,
                replace=False,
            )
        )

    by_label = color_by == "label" and "labels" in result

    labels = (
        result["labels"] if by_label else result["predictions"]
    )[shown, atom_index]

    for level in np.unique(labels):
        selected = shown[labels == level]

        ax.scatter(
            result["spots"][selected, atom_index, 0],
            result["spots"][selected, atom_index, 1],
            s=6,
            alpha=0.45,
            color=LEVEL_COLORS[int(level) % len(LEVEL_COLORS)],
            label=f"|{level}⟩",
            rasterized=True,
        )

    if "labels" in result:
        wrong = shown[
            result["labels"][shown, atom_index]
            != result["predictions"][shown, atom_index]
        ]

        ax.scatter(
            result["spots"][wrong, atom_index, 0],
            result["spots"][wrong, atom_index, 1],
            s=14,
            marker="x",
            color="black",
            lw=0.7,
            label="ошибка",
        )

    ax.set(xlabel="I (MTV)", ylabel="Q (MTV)")
    ax.set_title(f"Атом {channel}: пятна IQ", fontsize=10)
    ax.grid(alpha=0.15)


def draw_benchmarks(ax, entry: dict) -> None:
    names = ["NN", "центроид", "LDA", "QDA"]
    values = [entry["accuracy"]] + [
        entry["benchmarks"].get(key)
        for key in ("centroid", "lda", "qda")
    ]

    bars = ax.bar(
        names,
        values,
        color=["#1f5bb4", "#9aa9bd", "#7fa3c9", "#5f87b8"],
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{100 * value:.1f}%",
            ha="center",
            fontsize=8,
        )

    ax.set_ylim(max(0.0, min(values) - 0.1), 1.02)

    snr = ", ".join(
        f"{item['pair'][0]}|{item['pair'][1]} {item['snr']:.1f}"
        for item in entry["separation"]
        if item["snr"] is not None
    )

    ax.set_title(
        f"Атом {entry['atom']}: точность\nSNR пятен: {snr}",
        fontsize=8,
    )
    ax.grid(alpha=0.2, axis="y")


def draw_physics(ax, entry: dict) -> None:
    """Отобразить уже рассчитанные физические оценки."""
    count = len(entry["decay_error"])
    x = np.arange(count)
    width = 0.2

    series = (
        ("распад (матрица)", entry["decay_error"], "#c43c3c"),
        ("распад в окне", entry["transition_down"], "#ef8a8a"),
        (
            "возбуждение (матрица)",
            entry["excitation_error"],
            "#1f5bb4",
        ),
        ("возбуждение в окне", entry["transition_up"], "#8fb3e8"),
    )

    for index, (label, values, color) in enumerate(series):
        ax.bar(
            x + (index - 1.5) * width,
            [0 if value is None else value for value in values],
            width,
            label=label,
            color=color,
        )

    ax.set_xticks(x, [f"|{index}⟩" for index in range(count)])
    ax.set_xlabel("подготовленный уровень")
    ax.set_ylabel("вероятность")
    ax.set_title(f"Атом {entry['atom']}", fontsize=10)

    population = entry["thermal_population"]
    population_text = (
        "н/д" if population is None else f"{100 * population:.2f}%"
    )
    note = f"|0⟩ рано выше: {population_text}"

    temperature = entry.get("effective_temperature_mK")
    if temperature is not None:
        note += f"\nT_eff ≈ {temperature:.0f} мК*"

    ax.text(
        0.02,
        0.97,
        note,
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        bbox=dict(
            boxstyle="round",
            fc="white",
            ec="#c3cfdf",
            alpha=0.9,
        ),
    )
    ax.grid(alpha=0.2, axis="y")

def draw_kernels(
    figure,
    *,
    atoms,
    kernels,
    filter_names,
) -> None:
    """Построить подготовленные ядра фильтров.

    kernels — список словарей:
        channel: номер канала данных;
        name: название фильтра;
        values: массив коэффициентов.
    """
    if not atoms:
        draw_message(
            figure,
            "Выбранные атомы не используются моделью.",
        )
        return

    figure.clear()
    rows, columns = grid_shape(len(atoms))

    cmap = matplotlib.colormaps["tab10"]

    # Один фильтр получает одинаковый цвет на всех подграфиках.
    names = list(dict.fromkeys(filter_names))
    for entry in kernels:
        if entry["name"] not in names:
            names.append(entry["name"])

    colors = {
        name: cmap(index % 10)
        for index, name in enumerate(names)
    }

    for position, channel in enumerate(atoms, 1):
        ax = figure.add_subplot(rows, columns, position)
        drawn = False

        for entry in kernels:
            if entry["channel"] != channel:
                continue

            ax.plot(
                np.real(entry["values"]),
                lw=1.1,
                label=entry["name"],
                color=colors[entry["name"]],
            )
            drawn = True

        if not drawn:
            ax.text(
                0.5,
                0.5,
                "Нет откалиброванных ядер",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title(
            f"Атом {channel}: ядра (Re = вес I)",
            fontsize=10,
        )
        ax.set_xlabel("отсчёт окна")
        ax.grid(alpha=0.2)

    finish_figure(figure)


def draw_histograms(
    figure,
    *,
    atoms,
    diagnostics,
    filter_name,
) -> None:
    """Распределения готовых выходов фильтров."""
    if diagnostics is None:
        draw_message(
            figure,
            "Нажмите «Применить фильтры к данным» "
            "на вкладке «Фильтры».",
        )
        return

    atoms = [atom for atom in atoms if atom in diagnostics]

    if not atoms:
        draw_message(
            figure,
            "Для выбранных атомов нет результатов "
            "применения фильтров.",
        )
        return

    if all(
        diagnostics[atom]["values"] is None
        for atom in atoms
    ):
        draw_message(
            figure,
            "У модели «FNN по I/Q» нет банка фильтров — "
            "гистограммы недоступны.",
        )
        return

    figure.clear()
    rows, columns = grid_shape(len(atoms))

    for position, channel in enumerate(atoms, 1):
        ax = figure.add_subplot(rows, columns, position)
        entry = diagnostics[channel]
        values_all = entry["values"]
        original_names = entry["names"]

        if (
            values_all is None
            or not len(original_names)
            or values_all.shape[1] == 0
        ):
            ax.text(
                0.5,
                0.5,
                "Нет выходов фильтров",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Атом {channel}", fontsize=10)
            ax.set_axis_off()
            continue

        names = [
            name.replace(" (н/д)", "")
            for name in original_names
        ]

        index = names.index(filter_name) if filter_name in names else 0
        values = values_all[:, index]
        labels = entry["labels"]

        if labels is None:
            ax.hist(
                values,
                bins=60,
                color="#1f5bb4",
                alpha=0.8,
            )
        else:
            for level in np.unique(labels):
                ax.hist(
                    values[labels == level],
                    bins=60,
                    alpha=0.55,
                    label=f"|{level}⟩",
                    color=LEVEL_COLORS[
                        int(level) % len(LEVEL_COLORS)
                    ],
                )

        ax.set_title(
            f"Атом {channel}: выход {original_names[index]}",
            fontsize=10,
        )
        ax.set_xlabel("значение фильтра")
        ax.grid(alpha=0.2)

    finish_figure(figure)


def draw_traces(
    figure,
    *,
    atoms,
    data,
) -> None:
    """Средние трассы I/Q по меткам или отдельные трассы без меток."""
    if data is None:
        draw_message(figure, "Загрузите данные.")
        return

    iq = data["iq"]
    labels = data["labels"]

    atoms = [
        atom
        for atom in atoms
        if 0 <= atom < iq.shape[1]
    ]

    if not atoms:
        draw_message(
            figure,
            "В загруженных данных нет выбранных каналов.",
        )
        return

    if len(iq) == 0:
        draw_message(figure, "В данных нет измерений.")
        return

    time_ns = data.get("time_ns")
    time = (
        time_ns
        if time_ns is not None
        else np.arange(iq.shape[2])
    )

    figure.clear()
    rows, columns = grid_shape(len(atoms))

    for position, channel in enumerate(atoms, 1):
        ax = figure.add_subplot(rows, columns, position)
        traces = iq[:, channel]

        if labels is not None:
            for level in np.unique(labels[:, channel]):
                # Как и раньше: не более 2000 трасс на уровень.
                selected = np.flatnonzero(
                    labels[:, channel] == level
                )[:2000]

                mean_trace = traces[selected].mean(axis=0)
                color = LEVEL_COLORS[
                    int(level) % len(LEVEL_COLORS)
                ]

                ax.plot(
                    time,
                    mean_trace.real,
                    lw=1.6,
                    color=color,
                    label=f"I, |{level}⟩",
                )
                ax.plot(
                    time,
                    mean_trace.imag,
                    lw=1.2,
                    ls="--",
                    color=color,
                    label=f"Q, |{level}⟩",
                )

            title = f"Атом {channel}: средние трассы I (—) и Q (--)"
        else:
            for index in range(min(8, len(traces))):
                ax.plot(
                    time,
                    traces[index].real,
                    color="#2563b8",
                    alpha=0.3,
                    lw=0.7,
                    label="I" if index == 0 else "_nolegend_",
                )
                ax.plot(
                    time,
                    traces[index].imag,
                    color="#e08a1e",
                    alpha=0.3,
                    lw=0.7,
                    label="Q" if index == 0 else "_nolegend_",
                )

            title = f"Атом {channel}: отдельные трассы I и Q"

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(
            "время, нс" if time_ns is not None else "отсчёт"
        )
        ax.grid(alpha=0.2)

    finish_figure(figure)


def draw_leakage(
    figure,
    *,
    result,
    atom,
) -> None:
    """Отобразить готовый результат поиска утечки."""
    figure.clear()
    ax = figure.add_subplot(111)

    mtv = result["mtv"]
    labels = result["labels"]

    for level in np.unique(labels):
        selected = labels == level

        ax.scatter(
            mtv[selected].real,
            mtv[selected].imag,
            s=6,
            alpha=0.5,
            label=f"предложено |{level}⟩ ({selected.sum()})",
        )

    ax.set(
        xlabel="I (MTV)",
        ylabel="Q (MTV)",
        title=(
            f"Атом {atom}: "
            "спектральная кластеризация MTV (§V-A)"
        ),
    )

    if len(labels):
        ax.legend(fontsize=8)

    ax.grid(alpha=0.2)
    figure.tight_layout()
