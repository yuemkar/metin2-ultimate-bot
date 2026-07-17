from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    dataset_yaml: str
    output_model: str
    run_dir: str


class ModelTrainer:
    """Prepare a local YOLO dataset file and run Ultralytics training."""

    def __init__(
        self,
        project_root: str | Path = ".",
        data_dir: str | Path = "training_data",
        output_model: str | Path = "models/custom_metin.pt",
    ) -> None:
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / data_dir
        self.output_model = self.project_root / output_model
        self.dataset_yaml = self.data_dir / "dataset.yaml"

    def create_dataset_yaml(self, class_names: list[str] | None = None) -> Path:
        names = class_names or self._read_class_names() or ["metin"]
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "labels").mkdir(parents=True, exist_ok=True)

        dataset = {
            "path": str(self.data_dir.resolve()),
            "train": "images",
            "val": "images",
            "names": {index: name for index, name in enumerate(names)},
        }
        self.dataset_yaml.write_text(
            yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return self.dataset_yaml

    def train(
        self,
        base_model: str | Path = "yolov8n.pt",
        epochs: int = 25,
        imgsz: int = 640,
        class_names: list[str] | None = None,
    ) -> TrainingResult:
        dataset_yaml = self.create_dataset_yaml(class_names)
        self._validate_dataset()

        from ultralytics import YOLO

        model = YOLO(str(base_model))
        results = model.train(
            data=str(dataset_yaml),
            epochs=int(epochs),
            imgsz=int(imgsz),
            project=str((self.project_root / "runs").resolve()),
            name="custom_metin",
            exist_ok=True,
        )

        run_dir = Path(getattr(results, "save_dir", self.project_root / "runs" / "custom_metin"))
        best_model = run_dir / "weights" / "best.pt"
        if not best_model.exists():
            raise FileNotFoundError(f"Egitim tamamlandi ama best.pt bulunamadi: {best_model}")

        self.output_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model, self.output_model)
        logger.info("Custom model saved: %s", self.output_model)
        return TrainingResult(
            dataset_yaml=str(dataset_yaml),
            output_model=str(self.output_model),
            run_dir=str(run_dir),
        )

    def _read_class_names(self) -> list[str]:
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            return []

        try:
            metadata: list[dict[str, Any]] = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

        names = sorted({str(item.get("label", "")).strip() for item in metadata if item.get("label")})
        return [name for name in names if name]

    def _validate_dataset(self) -> None:
        image_files = list((self.data_dir / "images").glob("*.png"))
        if not image_files:
            raise RuntimeError("training_data/images altinda egitim goruntusu yok")

        label_files = list((self.data_dir / "labels").glob("*.txt"))
        if not label_files:
            raise RuntimeError(
                "YOLO egitimi icin training_data/labels altinda en az bir .txt etiket dosyasi gerekli"
            )
