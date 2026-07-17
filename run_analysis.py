from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.recorder import DataRecorder
from src.recorder import VideoRecorder
from src.trainer import ModelTrainer
from src.vision.debug_overlay import DebugOverlay
from src.vision.offline_analyzer import OfflineAnalyzer
from src.vision.performance_monitor import PerformanceMonitor
from src.core.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parent


def detection_to_dict(detection: object) -> dict[str, object]:
    return {
        "label": getattr(detection, "label", "offline_text_region"),
        "confidence": float(getattr(detection, "confidence", 0.0)),
        "bbox": tuple(getattr(detection, "bbox", (0, 0, 0, 0))),
        "center": tuple(getattr(detection, "center", (0, 0))),
    }


def analyze_image(args: argparse.Namespace) -> int:
    image_path = Path(args.image_path)
    analyzer = OfflineAnalyzer(color_profile=args.color_profile)
    monitor = PerformanceMonitor(PROJECT_ROOT / "logs" / "performance.json")

    monitor.start_measurement()
    detections = analyzer.analyze_image(image_path)
    monitor.end_measurement("analyze_image")
    monitor.log_performance()

    payload = {
        "image_path": str(image_path),
        "detections": [detection_to_dict(detection) for detection in detections],
    }

    if args.debug:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Goruntu okunamadi: {image_path}")
        debug_dir = PROJECT_ROOT / "training_data" / "debug"
        debug_path = debug_dir / f"{image_path.stem}_debug.png"
        overlay = DebugOverlay()
        overlay.save_debug_image(overlay.draw_boxes(image, detections), debug_path)
        payload["debug_image"] = str(debug_path)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def batch_analyze(args: argparse.Namespace) -> int:
    analyzer = OfflineAnalyzer(color_profile=args.color_profile)
    monitor = PerformanceMonitor(PROJECT_ROOT / "logs" / "performance.json")

    monitor.start_measurement()
    report = analyzer.batch_analyze(args.folder_path)
    monitor.end_measurement("batch_analyze")
    monitor.log_performance()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def auto_label(args: argparse.Namespace) -> int:
    analyzer = OfflineAnalyzer(color_profile=args.color_profile)
    images_dir = PROJECT_ROOT / "training_data" / "images"
    report = analyzer.auto_label_folder(images_dir, class_id=0)
    report["label"] = args.label
    report["labels_dir"] = str(PROJECT_ROOT / "training_data" / "labels")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def record(args: argparse.Namespace) -> int:
    recorder = DataRecorder(PROJECT_ROOT / "training_data")
    result = recorder.capture_and_save(args.label)
    print(json.dumps({"image_path": result.image_path, "label": result.label}, ensure_ascii=False, indent=2))
    return 0


def record_video(args: argparse.Namespace) -> int:
    config = load_config(PROJECT_ROOT / "config.yaml")
    fps = args.fps if args.fps is not None else float(config.get("recorder.video_fps", 5))
    duration = args.duration if args.duration is not None else float(config.get("recorder.default_duration", 60))

    recorder = VideoRecorder(PROJECT_ROOT / "training_data")
    print(f"Video frame kaydi basladi: label={args.label}, duration={duration}, fps={fps}")
    saved_paths = recorder.start_recording(args.label, duration=duration, fps=fps)
    print(
        json.dumps(
            {
                "label": args.label,
                "frame_count": len(saved_paths),
                "output_dir": str(PROJECT_ROOT / "training_data" / "images"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def train(args: argparse.Namespace) -> int:
    trainer = ModelTrainer(project_root=PROJECT_ROOT)
    result = trainer.train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        class_names=["surgun", "kizil", "golge"],
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


def gui(_: argparse.Namespace) -> int:
    from PyQt5.QtWidgets import QApplication

    from src.gui.main_window import MainWindow

    app = QApplication([])
    window = MainWindow(PROJECT_ROOT)
    window.show()
    return app.exec_()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline analysis tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("analyze-image", help="Tek goruntuyu offline analiz et")
    image_parser.add_argument("image_path")
    image_parser.add_argument("--debug", action="store_true", help="Debug kutulu goruntu kaydet")
    image_parser.add_argument("--color-profile", default="red", choices=["red", "blue", "green"])
    image_parser.set_defaults(func=analyze_image)

    batch_parser = subparsers.add_parser("batch-analyze", help="Klasordeki goruntuleri analiz et")
    batch_parser.add_argument("folder_path")
    batch_parser.add_argument("--color-profile", default="red", choices=["red", "blue", "green"])
    batch_parser.set_defaults(func=batch_analyze)

    auto_label_parser = subparsers.add_parser("auto-label", help="Goruntuler icin YOLO .txt etiketleri uret")
    auto_label_parser.add_argument("--label", default="surgun")
    auto_label_parser.add_argument("--color-profile", default="red", choices=["red", "blue", "green"])
    auto_label_parser.set_defaults(func=auto_label)

    record_parser = subparsers.add_parser("record", help="Ekran goruntusu kaydet")
    record_parser.add_argument("--label", required=True)
    record_parser.set_defaults(func=record)

    video_parser = subparsers.add_parser("record-video", help="Belirli FPS ile frame kaydi yap")
    video_parser.add_argument("--label", required=True)
    video_parser.add_argument("--duration", type=float, default=None)
    video_parser.add_argument("--fps", type=float, default=None)
    video_parser.set_defaults(func=record_video)

    train_parser = subparsers.add_parser("train", help="YOLO model egitimi baslat")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.set_defaults(func=train)

    gui_parser = subparsers.add_parser("gui", help="Offline analiz GUI'sini ac")
    gui_parser.set_defaults(func=gui)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
