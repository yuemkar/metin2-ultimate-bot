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

Tek goruntu:

```powershell
python main.py --analyze training_data/images/sample.png
```

Klasor:

```powershell
python main.py --analyze training_data/images
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
