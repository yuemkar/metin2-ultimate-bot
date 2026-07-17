# -*- coding: utf-8 -*-
import os
import time
import json
from datetime import datetime
import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api


class RecordResult:
    def __init__(self, image_path: str, label: str, timestamp: str):
        self.image_path = image_path
        self.label = label
        self.timestamp = timestamp


class DataRecorder:
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        self.metadata_path = os.path.join(output_dir, "metadata.json")
        self.metadata = []
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)

    def capture_screen(self, region=None):
        capture_region = self._normalize_region(region)
        hwnd_dc = win32gui.GetWindowDC(0)
        dc = win32ui.CreateDCFromHandle(hwnd_dc)
        create_dc = dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(dc, capture_region["width"], capture_region["height"])
        create_dc.SelectObject(bitmap)
        create_dc.BitBlt((0, 0), (capture_region["width"], capture_region["height"]), dc, 
                         (capture_region["left"], capture_region["top"]), win32con.SRCCOPY)
        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((capture_region["height"], capture_region["width"], 4))
        dc.DeleteDC()
        create_dc.DeleteDC()
        win32gui.ReleaseDC(0, hwnd_dc)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _normalize_region(self, region):
        if region is not None:
            x, y, w, h = region
            return {"left": x, "top": y, "width": w, "height": h}
        return {
            "left": win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN),
            "top": win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN),
            "width": win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN),
            "height": win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN),
        }

    def capture_and_save(self, label: str, region=None):
        image_bgr = self.capture_screen(region)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{label}_{timestamp}.png"
        filepath = os.path.join(self.images_dir, filename)
        cv2.imwrite(filepath, image_bgr)
        metadata_entry = {
            "filename": filename,
            "label": label,
            "timestamp": timestamp,
            "region": region,
            "image_path": filepath
        }
        self.metadata.append(metadata_entry)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        return RecordResult(
            image_path=filepath,
            label=label,
            timestamp=timestamp
        )

    def start_recording(self, label: str, region=None):
        print(f"🔴 Kayıt başladı: {label}")
        print("Durdurmak için Ctrl+C...")
        try:
            while True:
                self.capture_and_save(label, region)
                time.sleep(0.2)
        except KeyboardInterrupt:
            print(f"\n⏹️ Kayıt durduruldu. {len(self.metadata)} görüntü kaydedildi.")
