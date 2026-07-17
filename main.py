from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.bot import Bot
from src.core.config import load_config
from src.utils.logger import setup_logging
from src.vision.offline_analyzer import OfflineAnalyzer
from src.vision.performance_monitor import PerformanceMonitor


def run_offline_analysis(target: str) -> int:
    analyzer = OfflineAnalyzer()
    monitor = PerformanceMonitor()
    target_path = Path(target)

    if target_path.is_dir():
        monitor.start_measurement()
        report = analyzer.batch_analyze(target_path)
        monitor.end_measurement("batch_analyze")
        monitor.log_performance()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    monitor.start_measurement()
    detections = analyzer.analyze_image(target_path)
    monitor.end_measurement("analyze_image")
    monitor.log_performance()
    payload = [
        {
            "label": detection.label,
            "confidence": detection.confidence,
            "bbox": detection.bbox,
            "center": detection.center,
        }
        for detection in detections
    ]
    print(json.dumps({"image_path": str(target_path), "detections": payload}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Metin2 Ultimate Automation")
    parser.add_argument("--analyze", help="Kayitli bir goruntu veya klasor uzerinde offline analiz calistir")
    args = parser.parse_args()

    setup_logging()
    if args.analyze:
        return run_offline_analysis(args.analyze)

    config = load_config(Path(__file__).with_name("config.yaml"))
    bot = Bot(config)
    if config.get("window.auto_detect", True):
        hwnd = bot.find_game_window()
        if not hwnd:
            print("Oyun penceresi bulunamadı!")
            return 1
        bot.window_hwnd = hwnd
        bot.mouse.set_window_context(hwnd)
        if config.get("window.auto_focus", True):
            bot.bring_to_front(hwnd)

    try:
        bot.run()
    except KeyboardInterrupt:
        bot.stop()
        print("\nDurduruldu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
