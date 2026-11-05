"""Проверки элементов интерфейса.

* колесо мыши не должно менять значения полей и ползунков;
* галочка «скрытые слои автоматически» должна выставлять в полях те
  размеры, которые сеть получит при обучении: ⌊P/2⌋ и ⌊P/4⌋.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from types import SimpleNamespace

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication

# Сохраняем предусмотренный приложением порядок загрузки Qt и PyTorch.
from main import preload_system_icu

preload_system_icu()

from mainwindow import (  # noqa: E402
    MainWindow,
    NoWheelDoubleSpinBox,
    NoWheelSlider,
    NoWheelSpinBox,
)
from neural_service import NeuralService  # noqa: E402

# Виджеты Qt требуют экземпляра QApplication.
app = QApplication.instance() or QApplication([])


def send_wheel(widget, delta=120):
    event = QWheelEvent(
        QPointF(4, 4), QPointF(4, 4), QPoint(0, 0), QPoint(0, delta),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    QApplication.sendEvent(widget, event)


class WheelTests(unittest.TestCase):
    def test_spin_box_keeps_value(self):
        box = MainWindow._spin(0, 100, 40)
        box.setFocus()

        send_wheel(box)

        self.assertEqual(box.value(), 40)

    def test_double_spin_box_keeps_value(self):
        box = MainWindow._dspin(0, 1, 0.5, 0.1, 3)
        box.setFocus()

        send_wheel(box)

        self.assertEqual(box.value(), 0.5)

    def test_slider_keeps_value(self):
        slider = MainWindow._slider(0, 100, 40)
        slider.setFocus()

        send_wheel(slider)

        self.assertEqual(slider.value(), 40)

    def test_widget_classes(self):
        self.assertIsInstance(MainWindow._spin(0, 1, 0), NoWheelSpinBox)
        self.assertIsInstance(MainWindow._dspin(0, 1, 0, 0.1, 2), NoWheelDoubleSpinBox)
        self.assertIsInstance(MainWindow._slider(0, 1, 0), NoWheelSlider)


class FakeValue:
    """Поле или ползунок: значение, диапазон и признак доступности."""

    def __init__(self, value, minimum=0, maximum=100):
        self._value = value
        self._minimum = minimum
        self._maximum = maximum
        self.enabled = True

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = int(value) if isinstance(value, (int, float)) else value

    def setText(self, text):
        self._value = text

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def setEnabled(self, enabled):
        self.enabled = enabled


class FakeLabel:
    def __init__(self, text=""):
        self._text = text

    def text(self):
        return self._text

    def setText(self, text):
        self._text = text


class FakeWindow:
    """Минимальное состояние окна для _set_auto_layers."""

    _set_auto_layers = MainWindow._set_auto_layers
    _auto_layers_toggled = MainWindow._auto_layers_toggled

    def __init__(self, sizes):
        self.layers_slider = FakeValue(2, 0, 8)
        self.width_slider = FakeValue(64, 2, 1024)
        self.hidden_edit = FakeValue("64, 32")
        self.layers_label = FakeLabel("2")
        self.width_label = FakeLabel("64")
        self.auto_layers = SimpleNamespace(isChecked=lambda: True)
        self._sizes = list(sizes)

    def _auto_layer_sizes(self):
        return list(self._sizes)


class AutoLayersTests(unittest.TestCase):
    def test_checkbox_restores_auto_sizes(self):
        window = FakeWindow([8, 4])
        # Пользователь снял галочку и задал размеры вручную.
        window.hidden_edit.setValue("200, 100")
        window.width_slider.setValue(200)

        MainWindow._set_auto_layers(window, True)

        self.assertEqual(window.hidden_edit.value(), "8, 4")
        self.assertEqual(window.width_slider.value(), 8)
        self.assertEqual(window.layers_slider.value(), 2)
        self.assertEqual(window.width_label.text(), "8")
        self.assertEqual(window.layers_label.text(), "2")
        # В автоматическом режиме поля недоступны для правки.
        self.assertFalse(window.hidden_edit.enabled)
        self.assertFalse(window.width_slider.enabled)

    def test_unchecking_enables_manual_fields(self):
        window = FakeWindow([8, 4])

        MainWindow._set_auto_layers(window, False)

        self.assertTrue(window.hidden_edit.enabled)
        self.assertTrue(window.width_slider.enabled)
        self.assertEqual(window.hidden_edit.value(), "64, 32")


class AutoHiddenRuleTests(unittest.TestCase):
    def test_rule_for_separate_networks(self):
        config = {"per_atom_networks": True, "levels": [2, 3]}
        # P = 12: ⌊P/2⌋ = 6, но не уже 2k+2 = 8; ⌊P/4⌋ = 3, но не уже k+1 = 4.
        self.assertEqual(NeuralService.auto_hidden_layers(config, 12), [8, 4])

    def test_rule_for_shared_network(self):
        config = {"per_atom_networks": False, "levels": [2, 3]}
        self.assertEqual(NeuralService.auto_hidden_layers(config, 12), [20, 10])

    def test_rule_without_limits(self):
        config = {"per_atom_networks": True, "levels": [2]}
        self.assertEqual(NeuralService.auto_hidden_layers(config, 64), [32, 16])


if __name__ == "__main__":
    unittest.main()
