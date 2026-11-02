"""Проверка вычислительного устройства и перенос модели."""

from __future__ import annotations

import copy
import os

import torch

#: Предел потоков CPU для режима raw: выше него выигрыша нет.
MAX_CPU_THREADS = 8


def cpu_threads(mode: str) -> int:
    """Сколько потоков CPU отдать расчёту в данном режиме.

    Замер на 32-ядерной машине (Q13, батч 256, 5 эпох). В режиме matched
    вход сети — единицы признаков (P = 6), работа на один батч меньше
    затрат на синхронизацию потоков, и обучение на 16 потоках шло в 60 раз
    медленнее, чем на одном. В режиме raw вход 2·A·T (сотни–тысячи
    значений), потоки уже полезны: сеть [256, 128] на 8 потоках втрое
    быстрее, чем на одном, а на 16 потоках снова чуть медленнее.
    """
    if mode != "raw":
        return 1
    return max(1, min(MAX_CPU_THREADS, os.cpu_count() or 1))


def apply_cpu_threads(mode: str) -> int:
    """Применить :func:`cpu_threads` к PyTorch; вернуть число потоков."""
    threads = cpu_threads(mode)
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    return threads


def resolve_device(name: str) -> torch.device:
    """Проверить явно выбранное устройство: CPU или CUDA."""
    device = torch.device(name)

    if device.type == "cpu":
        return torch.device("cpu")

    if device.type != "cuda":
        raise ValueError("Поддерживаются CPU и NVIDIA CUDA.")

    if torch.version.cuda is None:
        raise RuntimeError(
            "Установлена CPU-сборка PyTorch. "
            "Для GPU нужна NVIDIA-версия приложения."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA недоступна. Проверьте наличие совместимой "
            "видеокарты NVIDIA и её драйвера."
        )

    index = 0 if device.index is None else device.index

    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f"Видеокарта cuda:{index} не найдена.")

    target = torch.device(f"cuda:{index}")

    # Проверяем операции, необходимые для обучения.
    try:
        with torch.enable_grad():
            x = torch.ones(
                (16, 16),
                dtype=torch.float32,
                device=target,
                requires_grad=True,
            )
            loss = (x @ x).square().mean()
            loss.backward()
            torch.cuda.synchronize(target)

        del loss, x

    except RuntimeError as exc:
        raise RuntimeError(
            f"Проверка вычислений на {target} не пройдена: {exc}"
        ) from exc

    return target


def tensors_to_cpu(value):
    """Скопировать вложенное состояние оптимизатора на CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)

    if isinstance(value, dict):
        return {
            key: tensors_to_cpu(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [tensors_to_cpu(item) for item in value]

    if isinstance(value, tuple):
        return tuple(tensors_to_cpu(item) for item in value)

    return copy.deepcopy(value)


def prepare_model_transfer(model, optimizer_state, target):
    """Подготовить кандидата, не изменяя исходную модель."""
    state = tensors_to_cpu(optimizer_state)

    if model is None:
        return None, state

    candidate = copy.deepcopy(model).to(target)

    if target.type == "cuda":
        torch.cuda.synchronize(target)

    return candidate, state