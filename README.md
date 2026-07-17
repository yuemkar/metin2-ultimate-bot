# Metin2 Ultimate Automation

Windows icin terminal tabanli ekran yakalama, HSV renk tespiti, opsiyonel YOLO tespiti ve kontrollu fare tiklama iskeleti.

Bu proje egitim amaclidir. Kullandiginiz uygulama veya oyunun kullanim sartlarini ve otomasyon kurallarini kontrol edin. Anti-cheat atlatma, gizleme veya guvenlik mekanizmasi bypass kodu icermez.

## Kurulum

```powershell
cd C:\Users\yek\Documents\mt\metin2_ultimate
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Calistirma

```powershell
python main.py
```

Varsayilan olarak tiklama kapali gelir. `config.yaml` icindeki `input.click_enabled` degeri `true` yapilmadan sadece ekran yakalama, hedef tespiti, terminal istatistikleri ve `debug_frame.jpg` uretimi calisir.

Guvenlik icin `config.yaml` icinde `input.click_enabled: false` varsayilandir. Offline analiz araclari canli tiklama, hareket veya input gondermez.

## Offline Analiz

Offline analiz icin ana giris noktasi [run_analysis.py](run_analysis.py) dosyasidir.

Tek goruntu analizi:

```powershell
python run_analysis.py analyze-image training_data\images\sample.png
```

Debug kutulari cizerek analiz:

```powershell
python run_analysis.py analyze-image training_data\images\sample.png --debug
```

Klasor analizi:

```powershell
python run_analysis.py batch-analyze training_data\images
```

Ekran goruntusu kaydi:

```powershell
python run_analysis.py record --label surgun
```

Video/frame kaydi:

```powershell
python run_analysis.py record-video --label surgun --duration 60 --fps 5
```

Bu komut belirtilen sure boyunca ekran frame'lerini bellekte toplar ve kayit bittiginde `training_data\images` altina PNG olarak yazar. Varsayilan degerler `config.yaml` icindeki `recorder.video_fps` ve `recorder.default_duration` alanlarindan gelir.

YOLO egitimi:

```powershell
python run_analysis.py train --epochs 50 --imgsz 640
```

Offline GUI:

```powershell
python run_analysis.py gui
```

## Ayarlar

Ana ayarlar [config.yaml](config.yaml) icindedir:

- `capture.region`: Yakalanacak ekran bolgesi.
- `detection.mode`: `color`, `yolo` veya `hybrid`.
- `detection.hsv_ranges`: HSV alt/ust renk araliklari.
- `input.click_enabled`: Baslangicta tiklama aktif mi.
- `detection.click_cooldown_ms`: Tiklamalar arasi bekleme.
- `app.save_debug_frame`: Son debug goruntusunu dosyaya yazma.
- `analysis_settings.offline_mode`: Offline analiz varsayilan modu.
- `analysis_settings.color_profile`: `red`, `blue` veya `green`.

YOLO modunu kullanmak icin `detection.mode` degerini `yolo` veya `hybrid` yapin. `ultralytics` ilk calismada model dosyasini indirebilir; isterseniz modeli `models/yolov8n.pt` konumuna elle koyabilirsiniz.

## OCR Kurulumu

OCR destegi Tesseract motoru ve Python tarafinda `pytesseract`/`Pillow` paketleri ile calisir. Python paketleri `requirements.txt` icindedir:

```powershell
pip install -r requirements.txt
```

Windows icin Tesseract motorunu ayrica kurmaniz gerekir:

1. Tesseract dokumantasyonundaki Windows kurulum notlarini okuyun: https://tesseract-ocr.github.io/tessdoc/Installation.html
2. Windows installer icin UB Mannheim paketleri kullanilir: https://github.com/UB-Mannheim/tesseract/wiki
3. Kurulumdan sonra `C:\Program Files\Tesseract-OCR` klasorunu PATH degiskenine ekleyin.
4. Turkish OCR icin `tur.traineddata` dosyasinin `tessdata` klasorunde oldugunu kontrol edin.

Kontrol:

```powershell
tesseract --version
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

OCR ayarlari `config.yaml` icindeki `ocr` bolumundedir ve varsayilan olarak `enabled: false` gelir. `OCRReader`, `OfflineAnalyzer` ve `DataRecorder.capture_with_metadata()` ile birlikte kullanildiginda kayitli goruntu bolgelerinden metin okur ve metadata uretir. Tesseract kurulu degilse hata firlatmadan pasif kalir.

Ayrintili rehber: [docs/OCR_GUIDE.md](docs/OCR_GUIDE.md)

## EXE Uretimi

```powershell
.\build_exe.ps1
```

EXE ciktisi `dist\metin2_automation_tool.exe` olarak uretilir. Arac bilerek acik ve tanimlayici bir isimle paketlenir.
