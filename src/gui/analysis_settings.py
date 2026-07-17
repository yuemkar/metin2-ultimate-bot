from __future__ import annotations

from pathlib import Path

import yaml
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class AnalysisSettings(QWidget):
    """Offline analysis settings panel; it never sends input to any window."""

    settings_saved = pyqtSignal(dict)

    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        super().__init__()
        self.config_path = Path(config_path)

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["red", "blue", "green"])

        self.wait_spin = QDoubleSpinBox()
        self.wait_spin.setRange(0.0, 10.0)
        self.wait_spin.setSingleStep(0.1)
        self.wait_spin.setValue(0.5)

        self.status_label = QLabel("Hazir")
        self.save_button = QPushButton("Kaydet")
        self.save_button.clicked.connect(self.save_settings)
        self.load_settings()

        group = QGroupBox("Offline Analiz Ayarlari")
        form = QFormLayout(group)
        form.addRow("Renk profili", self.profile_combo)
        form.addRow("Bekleme suresi", self.wait_spin)
        form.addRow(self.save_button)

        layout = QVBoxLayout(self)
        layout.addWidget(group)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def save_settings(self) -> None:
        config = self._read_config()
        offline = config.setdefault("offline_analysis", {})
        offline["color_profile"] = self.profile_combo.currentText()
        offline["wait_seconds"] = float(self.wait_spin.value())
        offline["live_input_enabled"] = False

        self.config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        self.status_label.setText("Ayarlar kaydedildi")
        self.settings_saved.emit(offline)

    def load_settings(self) -> None:
        config = self._read_config()
        offline = config.get("offline_analysis", {})
        profile = str(offline.get("color_profile", "red"))
        index = self.profile_combo.findText(profile)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)
        self.wait_spin.setValue(float(offline.get("wait_seconds", 0.5)))

    def _read_config(self) -> dict:
        if not self.config_path.exists():
            return {}
        with self.config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
