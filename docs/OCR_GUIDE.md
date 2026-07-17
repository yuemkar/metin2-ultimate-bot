# OCR Guide

OCR destegi pasif ve opsiyoneldir. Bu modul yalnizca offline analiz, veri toplama ve debug metadata uretimi icin kullanilir. Canli bot akisi, tiklama, hareket veya input islemlerine bagli degildir.

## Windows Tesseract Kurulumu

1. Tesseract Windows kurulum notlarini okuyun: https://tesseract-ocr.github.io/tessdoc/Installation.html
2. Windows installer icin UB Mannheim paketlerini kullanin: https://github.com/UB-Mannheim/tesseract/wiki
3. Kurulum klasorunu PATH'e ekleyin:

```powershell
C:\Program Files\Tesseract-OCR
```

4. Turkce OCR icin `tur.traineddata` dosyasinin su klasorde oldugunu kontrol edin:

```powershell
C:\Program Files\Tesseract-OCR\tessdata
```

5. Kurulumu dogrulayin:

```powershell
tesseract --version
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

## Python Paketleri

```powershell
pip install -r requirements.txt
```

`requirements.txt` icinde `pytesseract` ve `Pillow` bulunur.

## Config

OCR varsayilan olarak kapalidir:

```yaml
ocr:
  enabled: false
  lang: "tur"
  keywords: ["Sürgün", "Kayası", "Kaya", "Metin"]
  tesseract_path: "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

Tesseract kurulu degilse `OCRReader` hata firlatmaz; uyari loglar ve pasif kalir.

## Offline Analizde Kullanma

```python
from src.core.config import load_config
from src.vision.offline_analyzer import OfflineAnalyzer
from src.vision.ocr_reader import OCRReader

config = load_config("config.yaml")
ocr = OCRReader(config)
analyzer = OfflineAnalyzer(ocr_reader=ocr)
detections = analyzer.analyze_image("training_data/images/sample.png")

for item in analyzer.last_metadata:
    print(item["bbox"], item.get("ocr_text"), item.get("ocr_keyword_match"))
```

## Veri Toplama Metadata

```python
from src.recorder import DataRecorder

recorder = DataRecorder("training_data")
recorder.capture_with_metadata("offline_ocr_sample", analyzer=analyzer, ocr_reader=ocr)
```

Bu kayit `metadata.json` icine tespit kutularini ve OCR etkinse okunan metni yazar.
