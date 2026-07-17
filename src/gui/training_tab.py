from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.recorder import DataRecorder
from src.trainer import ModelTrainer


class RecorderWorker(QObject):
    finished = pyqtSignal()
    failed = pyqtSignal(str)
    saved = pyqtSignal(str)

    def __init__(self, label: str, project_root: Path) -> None:
        super().__init__()
        self.label = label
        self.project_root = project_root
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            recorder = DataRecorder(self.project_root / "training_data")
            result = recorder.capture_and_save(self.label)
            self.saved.emit(result.image_path)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class TrainerWorker(QObject):
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, project_root: Path, epochs: int, imgsz: int) -> None:
        super().__init__()
        self.project_root = project_root
        self.epochs = epochs
        self.imgsz = imgsz

    def run(self) -> None:
        try:
            trainer = ModelTrainer(project_root=self.project_root)
            result = trainer.train(epochs=self.epochs, imgsz=self.imgsz)
            self.finished.emit(result.output_model)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class TrainingTab(QWidget):
    def __init__(self, project_root: str | Path = ".") -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.recorder_thread: QThread | None = None
        self.recorder_worker: RecorderWorker | None = None
        self.trainer_thread: QThread | None = None
        self.trainer_worker: TrainerWorker | None = None

        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("metin")
        self.record_button = QPushButton("Kaydet")
        self.stop_record_button = QPushButton("Durdur")
        self.stop_record_button.setEnabled(False)
        self.status_label = QLabel("Hazir")

        self.epoch_input = QSpinBox()
        self.epoch_input.setRange(1, 500)
        self.epoch_input.setValue(25)
        self.imgsz_input = QSpinBox()
        self.imgsz_input.setRange(128, 2048)
        self.imgsz_input.setSingleStep(32)
        self.imgsz_input.setValue(640)
        self.train_button = QPushButton("Egit")

        self.record_button.clicked.connect(self.start_recording)
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.train_button.clicked.connect(self.start_training)

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_recording_group())
        layout.addWidget(self._build_training_group())
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def _build_recording_group(self) -> QGroupBox:
        group = QGroupBox("Veri Toplama")
        form = QFormLayout(group)
        buttons = QHBoxLayout()
        buttons.addWidget(self.record_button)
        buttons.addWidget(self.stop_record_button)
        form.addRow("Metin adi", self.label_input)
        form.addRow(buttons)
        return group

    def _build_training_group(self) -> QGroupBox:
        group = QGroupBox("Model Egitimi")
        form = QFormLayout(group)
        form.addRow("Epoch", self.epoch_input)
        form.addRow("Image size", self.imgsz_input)
        form.addRow(self.train_button)
        return group

    def start_recording(self) -> None:
        label = self.label_input.text().strip() or "metin"
        self.recorder_thread = QThread(self)
        self.recorder_worker = RecorderWorker(label, self.project_root)
        self.recorder_worker.moveToThread(self.recorder_thread)
        self.recorder_thread.started.connect(self.recorder_worker.run)
        self.recorder_worker.saved.connect(lambda path: self.status_label.setText(f"Kaydedildi: {path}"))
        self.recorder_worker.failed.connect(lambda error: self.status_label.setText(f"Hata: {error}"))
        self.recorder_worker.finished.connect(self.recorder_thread.quit)
        self.recorder_worker.finished.connect(self.recorder_worker.deleteLater)
        self.recorder_thread.finished.connect(self.recorder_thread.deleteLater)
        self.recorder_thread.finished.connect(self._recording_finished)
        self.record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        self.status_label.setText("Kayit aliniyor...")
        self.recorder_thread.start()

    def stop_recording(self) -> None:
        if self.recorder_worker:
            self.recorder_worker.stop()
        self.status_label.setText("Durdurma istendi")

    def start_training(self) -> None:
        self.trainer_thread = QThread(self)
        self.trainer_worker = TrainerWorker(self.project_root, self.epoch_input.value(), self.imgsz_input.value())
        self.trainer_worker.moveToThread(self.trainer_thread)
        self.trainer_thread.started.connect(self.trainer_worker.run)
        self.trainer_worker.finished.connect(lambda path: self.status_label.setText(f"Model hazir: {path}"))
        self.trainer_worker.failed.connect(lambda error: self.status_label.setText(f"Hata: {error}"))
        self.trainer_worker.finished.connect(self.trainer_thread.quit)
        self.trainer_worker.failed.connect(self.trainer_thread.quit)
        self.trainer_worker.finished.connect(self.trainer_worker.deleteLater)
        self.trainer_thread.finished.connect(self.trainer_thread.deleteLater)
        self.trainer_thread.finished.connect(lambda: self.train_button.setEnabled(True))
        self.train_button.setEnabled(False)
        self.status_label.setText("Egitim basladi...")
        self.trainer_thread.start()

    def _recording_finished(self) -> None:
        self.record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
