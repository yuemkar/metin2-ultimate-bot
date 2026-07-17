from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QMainWindow, QTabWidget

from src.gui.analysis_settings import AnalysisSettings
from src.gui.training_tab import TrainingTab


class MainWindow(QMainWindow):
    def __init__(self, project_root: str | Path = ".") -> None:
        super().__init__()
        self.setWindowTitle("Metin2 Ultimate Automation")
        self.resize(900, 640)

        tabs = QTabWidget()
        tabs.addTab(TrainingTab(project_root), "Training")
        tabs.addTab(AnalysisSettings(Path(project_root) / "config.yaml"), "Offline Analysis")
        self.setCentralWidget(tabs)
