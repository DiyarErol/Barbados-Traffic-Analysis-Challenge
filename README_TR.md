# Barbados Trafik Sıkışıklığı Analizi Çözümü

Bu çözüm, Norman Niles kavşağındaki trafik sıkışıklığı seviyesini video verilerinden tahmin eder.

## 🎯 Çözüm Özeti

### Yaklaşım
- **Video İşleme**: OpenCV ile otomatik araç tespiti (Background Subtraction)
- **Özellik Çıkarma**: Video verilerinden 30+ trafik özelliği
- **Zaman Serisi Modelleme**: Geçmiş verileri kullanan Gradient Boosting modeli
- **Gerçek Zamanlı Tahmin**: 15 dk input → 2 dk embargo → 5 dk tahmin

### Temel Özellikler

#### 1. Video Bazlı Özellikler
- **Araç Sayısı**: Ortalama, maksimum, minimum, standart sapma
- **Hareket Skorları**: Frame-to-frame değişim analizi
- **Yoğunluk Metrikleri**: Piksel bazlı trafik yoğunluğu

#### 2. Zaman Bazlı Özellikler
- **Zaman Kategorileri**: Saat, dakika, gün içi periyot
- **Rush Hour Tespiti**: Yoğun trafik saatleri (07:00-09:00, 16:00-18:00)
- **Döngüsel Özellikler**: Sin/Cos transformasyonları

#### 3. İstatistiksel Özellikler
- **Lagged Features**: 1, 2, 3, 5 dakika gecikmeli değerler
- **Rolling Statistics**: 3, 5, 10 dakikalık hareketli ortalama/std
- **Trend Analizi**: Zaman içindeki değişim tespiti

## 📋 Özellik Önemi (Top 20)

Model eğitiminden sonra `feature_importance_report.csv` dosyasında detaylı rapor oluşturulur.

### En Önemli Faktörler

1. **vehicle_count_mean**: Ortalama araç sayısı - Ana gösterge
2. **density_mean**: Yoğunluk skoru - Trafik akış kalitesi
3. **movement_mean**: Hareket skoru - Duran vs hareketli araçlar
4. **vehicle_count_rolling_mean_5**: Son 5 dk araç sayısı trendi
5. **is_rush_hour**: Yoğun saat göstergesi
6. **vehicle_count_lag_1**: 1 dakika önceki araç sayısı
7. **density_rolling_std_10**: 10 dk yoğunluk değişkenliği
8. **hour**: Gün içi saat bilgisi
9. **signaling_encoded**: Sinyal kullanım seviyesi
10. **movement_rolling_trend_5**: 5 dk hareket trendi

### Özellik Kategorileri ve Katkıları
| Kategori | Katkı (%) | Açıklama |
|----------|-----------|----------|
| Araç Sayısı Metrikleri | ~35% | En temel tıkanıklık göstergesi |
| Yoğunluk Analizi | ~25% | Kavşak doluluk oranı |
| Zaman Özellikleri | ~20% | Günlük ve saatlik paternler |
| İstatistiksel Trendler | ~5% | Kısa/orta vadeli değişimler |
## 🚀 Kullanım

### 1. Ortam Kurulumu

```bash
```

### 2. Video Verilerini Hazırlama

Video dosyalarının `videos/` klasöründe bulunması gerekir:
videos/
  normanniles1/
    normanniles1_2025-10-20-06-00-45.mp4
    normanniles1_2025-10-20-06-01-45.mp4
    ...
```
python traffic_analysis_solution.py
```

Bu script:
- Eğitim verilerini yükler
- Video özelliklerini çıkarır
- Zaman serisi özellikleri ekler
- Modeli eğitir ve kaydeder
- Özellik önem raporunu oluşturur

### 4. Test Tahmini

```python
python test_prediction.py
```

Bu script:
- Eğitilmiş modeli yükler
- Test verilerini işler
- Gerçek zamanlı kısıtlamalara uygun tahmin yapar
- `submission.csv` dosyasını oluşturur

## 🔬 Teknik Detaylar

### Video İşleme Pipeline

```python
# 1. Video yükleme
cap = cv2.VideoCapture(video_path)

# 2. Background subtraction
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
fg_mask = bg_subtractor.apply(frame)

# 3. Morfolojik işlemler
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

# 4. Araç tespiti (kontur analizi)
contours = cv2.findContours(cleaned, cv2.RETR_EXTERNAL)
vehicles = [c for c in contours if cv2.contourArea(c) > min_area]
```

### Gerçek Zamanlı Kısıtlamalar

```python
# 15 dakika input penceresi
input_window = test_data[time_segment: time_segment+15]

# 2 dakika embargo (operasyonel gecikme)
embargo_start = time_segment + 15
embargo_end = embargo_start + 2

# 5 dakika tahmin penceresi
prediction_start = embargo_end
prediction_end = prediction_start + 5

# ÖNEMLİ: Her tahminde SADECE geçmiş verileri kullan
for t in range(prediction_start, prediction_end):
    available_data = test_data[:t]  # Gelecek verisi YOK
    prediction = model.predict(available_data)
```

### Model Yapısı

```python
# Gradient Boosting Classifier (Geri yayılım YOK)
model = GradientBoostingClassifier(
    n_estimators=200,      # 200 ağaç
    learning_rate=0.1,     # Öğrenme hızı
    max_depth=5,           # Maksimum derinlik
    subsample=0.8,         # Veri örnekleme
    random_state=42        # Tekrarlanabilirlik
)

# Enter ve Exit için ayrı modeller
model_enter.fit(X_train, y_enter)
model_exit.fit(X_train, y_exit)
```

## 📊 Performans Optimizasyonu

### Video İşleme Hızlandırma

1. **Frame Sampling**: Her frame yerine saniyede 2 frame işle
   ```python
   sample_rate = max(1, int(fps / 2))
   if frame_count % sample_rate != 0:
       continue
   ```

2. **Çoklu İşlem**: Paralel video işleme
   ```python
   from multiprocessing import Pool
   with Pool(processes=4) as pool:
       results = pool.map(process_video, video_list)
   ```

3. **GPU Kullanımı**: CUDA destekli OpenCV (opsiyonel)

### Model Optimizasyonu

- **Feature Selection**: En önemli 50 özelliği seç
- **Early Stopping**: Validation kaybı artınca dur
- **Hyperparameter Tuning**: Grid search ile en iyi parametreler

## 🎓 Veri Augmentation (İsteğe Bağlı)

Eğitim verisini artırmak için:

```python
# 1. Video rotasyonu/flipping (dikkatli kullan)
# 2. Brightness/contrast ayarları
# 3. Zaman penceresi kayması
# 4. Sentetik örnekler (interpolasyon)
```

**NOT**: Tüm augmentation süreçleri tekrarlanabilir ve kodda bulunmalıdır.

## 📝 Özellik Belgesi (Top 20 İçin)

### Feature Importance Report Format

| Feature Name | Category | Importance (Enter) | Importance (Exit) | Description |
|--------------|----------|-------------------|-------------------|-------------|
| vehicle_count_mean | Video | 0.145 | 0.132 | Ortalama araç sayısı |
| density_mean | Video | 0.128 | 0.118 | Ortalama yoğunluk skoru |
| movement_mean | Video | 0.095 | 0.089 | Ortalama hareket skoru |
| ... | ... | ... | ... | ... |

**Notlar**:
- Importance değerleri 0-1 arası normalize edilmiştir
- Toplam importance = 1.0
- Kategori: Video, Temporal, Statistical, Lagged, Rolling

## ⚠️ Önemli Notlar

### Geri Yayılım Yasağı

- ✅ **İZİN VERİLEN**: Eğitim sırasında model ağırlık güncellemesi
- ❌ **YASAK**: Test/inference sırasında model güncelleme
- ❌ **YASAK**: Online learning/adaptive modeller

### Gerçek Zamanlı Gereksinimler

- Her dakika sıralı tahmin
- Gelecek verilerini kullanmama
- 2 dakika operasyonel gecikme
- Manuel etiketleme yasak

### Veri Kullanımı

```python
# ✅ DOĞRU: Geçmiş verileri kullan
prediction_t = model.predict(data[:t])

# ❌ YANLIŞ: Gelecek verileri kullanma
prediction_t = model.predict(data[:t+5])  # t+5 gelecek!
```

## 🔧 Geliştirme Önerileri

### Kısa Vadeli İyileştirmeler

1. **YOLO Entegrasyonu**: Daha doğru araç tespiti
2. **Araç Takibi**: ByteTrack/DeepSORT ile araç sayımı
3. **Optik Akış**: Lucas-Kanade ile hız tahmini
4. **Ensemble Modeller**: RF + GB + XGBoost kombinasyonu

### Uzun Vadeli İyileştirmeler

1. **Deep Learning**: LSTM/Transformer modelleri (dikkatli: geri yayılım!)
2. **Grafik Modelleme**: Kavşak yapısını grafta modelle
3. **Anomaly Detection**: Olağandışı trafik paterni tespiti
4. **Multi-Camera Fusion**: 4 kamerayı birlikte değerlendir

## 📚 Kaynaklar

- OpenCV Documentation: https://docs.opencv.org/
- Scikit-learn: https://scikit-learn.org/
- Traffic Flow Theory: Highway Capacity Manual
- Computer Vision for Traffic Analysis: Recent surveys

## 📞 İletişim

Bu çözüm Barbados Traffic Analysis Challenge için geliştirilmiştir.

**Yarışma Detayları**: https://zindi.africa/

---

**Lisans**: MIT  
**Geliştirme**: 2025  
**Versiyon**: 1.0
