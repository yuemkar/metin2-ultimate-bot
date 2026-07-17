# Offline Analysis

Bu moduller yalnizca kayitli goruntuler uzerinde bilgisayarli goru, debug ve veri toplama islemleri yapar. Canli oyuna tiklama, hareket veya input gondermez.

## OfflineAnalyzer

`OfflineAnalyzer`, kayitli goruntude HSV renk profili ve kontur analizi ile metin benzeri bolgeleri bulur. `analyze_image()` sonucu mevcut `Detection` nesneleri listesi olarak doner.

```python
from src.vision.offline_analyzer import OfflineAnalyzer

analyzer = OfflineAnalyzer(color_profile="red")
regions = analyzer.analyze_image("training_data/images/sample.png")
for detection in regions:
    print(detection.label, detection.confidence, detection.bbox)
```

Toplu analiz:

```python
report = analyzer.batch_analyze("training_data/images")
print(report["total_detections"])
```

Toplu analiz, klasore `offline_analysis_report.json` yazar.

## HSV Renk Profili

Varsayilan profiller:

- `red`: kirmizi tonlari
- `blue`: mavi tonlari
- `green`: yesil tonlari

`config.yaml` icinde offline varsayilanlar `analysis_settings` altinda tutulur:

```yaml
analysis_settings:
  offline_mode: true
  color_profile: "red"
  wait_seconds: 0.5
```

Dar veya genis tespit icin `min_area`, `max_area`, `min_aspect` ve `max_aspect` degerleri `OfflineAnalyzer` olusturulurken ayarlanabilir.

```python
analyzer = OfflineAnalyzer(
    color_profile="blue",
    min_area=100,
    max_area=12000,
    min_aspect=0.4,
    max_aspect=4.0,
)
```

## DebugOverlay

Tespit edilen bolgeleri yesil dikdortgenlerle gorsellestirir.

```python
import cv2
from src.vision.debug_overlay import DebugOverlay

image = cv2.imread("training_data/images/sample.png")
overlay = DebugOverlay()
debug = overlay.draw_boxes(image, regions)
overlay.save_debug_image(debug, "training_data/debug/sample_debug.png")
```

## PerformanceMonitor

Detector veya offline analyzer gibi fonksiyonlarin latency ve FPS degerlerini pasif olarak olcer.

```python
from src.vision.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor("training_data/performance_log.json")
regions = monitor.measure("offline_analyze", analyzer.analyze_image, "training_data/images/sample.png")
monitor.log_performance()
```

JSON ciktisinda `average_latency_ms`, `average_fps` ve tek tek sample kayitlari bulunur. Varsayilan cikti `logs/performance.json` dosyasidir.

```python
monitor.start_measurement()
regions = analyzer.analyze_image("training_data/images/sample.png")
monitor.end_measurement("analyze_image")
monitor.log_performance()
```

## CLI Kullanim

Offline analiz icin onerilen giris noktasi proje kokundeki `run_analysis.py` dosyasidir.

Tek goruntu:

```powershell
python run_analysis.py analyze-image training_data/images/sample.png
```

Debug gorseli ile tek goruntu:

```powershell
python run_analysis.py analyze-image training_data/images/sample.png --debug
```

Klasor:

```powershell
python run_analysis.py batch-analyze training_data/images
```

Kayit:

```powershell
python run_analysis.py record --label surgun
```

Egitim:

```powershell
python run_analysis.py train --epochs 50 --imgsz 640
```

GUI:

```powershell
python run_analysis.py gui
```

`main.py --analyze` geriye uyumluluk icin korunur, ancak yeni offline is akisinda `run_analysis.py` kullanilmalidir.

## Ornek Cikti

Tek goruntu analizi JSON formatinda Detection bilgilerini yazar:

```json
{
  "image_path": "training_data/images/sample.png",
  "detections": [
    {
      "label": "offline_text_region",
      "confidence": 0.82,
      "bbox": [120, 88, 64, 42],
      "center": [152, 109]
    }
  ],
  "debug_image": "training_data/debug/sample_debug.png"
}
```

Toplu analiz, analiz edilen klasore `offline_analysis_report.json` yazar:

```json
{
  "image_count": 12,
  "total_detections": 31,
  "total_regions": 31,
  "results": []
}
```

## Metadata Ile Kaydetme

`DataRecorder.save_with_metadata()` goruntuyu ve pasif analiz bolgelerini `metadata.json` dosyasina ekler.

```python
import cv2
from src.recorder import DataRecorder

image = cv2.imread("training_data/images/sample.png")
recorder = DataRecorder("training_data")
recorder.save_with_metadata(image, "offline_sample", regions)
```

Bu akis egitim verisi etiketleme ve kalite kontrol icindir.
