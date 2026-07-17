# Pasif Analiz Modulleri

Bu dokuman HP template matching, hedef aday kuyrugu ve veri toplama tarafindaki HP bolgesi metadata destegini anlatir. Bu moduller oyun ici hareket, tiklama, loot, kill akisi veya otomatik savas davranisi tetiklemez.

## HP Takibi

`src/vision/hp_tracker.py` icindeki `HPTracker`, verilen bir frame ile HP bar template goruntusunu karsilastirir ve `0.0` ile `1.0` arasinda skor dondurur.

```python
import cv2
from src.vision.hp_tracker import HPTracker

frame = cv2.imread("training_data/images/sample.png")
tracker = HPTracker(threshold=0.75)
score = tracker.get_hp_score(frame, "templates/hp_template.png")
visible = tracker.is_visible(frame, "templates/hp_template.png")
print(score, visible)
```

Skor sadece goruntude template'in ne kadar benzedigini ifade eder. Herhangi bir input veya oyun aksiyonu uretmez.

## Hedef Kuyrugu

`src/vision/target_queue.py` icindeki `TargetQueue`, hedef adaylarini confidence degerine gore siralar. Offline debug, model ciktisi inceleme veya egitim verisi kalite kontrolu icin kullanilir.

```python
from src.vision.target_queue import TargetQueue

queue = TargetQueue(max_size=10)
for detection in detections:
    queue.add_candidate(detection)

ranked_candidates = queue.get_queue()
queue.clear()
```

Bu sinif sadece bellek ici veri yapisidir; tiklama, hareket veya pencere kontrolu yapmaz.

## HP Bolgesi Veri Toplama

`DataRecorder.capture_hp_region()` ekrani kaydeder ve secilen HP bar bolgesini `metadata.json` icine `hp_region` olarak yazar.

```python
from src.recorder import DataRecorder

recorder = DataRecorder("training_data")
recorder.capture_hp_region(
    label="metin_hp_ornegi",
    hp_region=(420, 80, 180, 24),
)
```

`hp_region` formati `(x, y, width, height)` seklindedir ve kaydedilen goruntu uzerindeki bolgeyi temsil eder.
