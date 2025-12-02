"""
Barbados Traffic Congestion Analysis Solution
Bu çözüm, Norman Niles kavşağındaki trafik sıkışıklığını tahmin eder.

Özellik Çıkarma Stratejisi:
1. Video işleme ile araç tespiti (YOLO/OpenCV)
2. Trafik akış metrikleri (araç sayısı, hız, yoğunluk)
3. Zaman bazlı özellikler (saat, gün içi patern)
4. İstatistiksel özellikler (hareketli ortalama, varyans)

Gerçek Zamanlı Kısıtlamalar:
- Her dakika sıralı tahmin
- Gelecek verilerini kullanmama
- 2 dakika embargo süresi
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Sklearn ve diğer kütüphaneler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib


class VideoFeatureExtractor:
    """
    Video verilerinden trafik özellikleri çıkarır.
    Manuel etiketleme kullanmadan otomatik feature extraction.
    """
    
    def __init__(self, use_yolo: bool = False):
        """
        Args:
            use_yolo: YOLO modeli kullan (False ise background subtraction)
        """
        self.use_yolo = use_yolo
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        
        # Eğer YOLO kullanılacaksa (opsiyonel, daha yavaş ama daha doğru)
        if self.use_yolo:
            try:
                # YOLOv8 veya benzeri kullanılabilir
                # from ultralytics import YOLO
                # self.yolo_model = YOLO('yolov8n.pt')
                pass
            except:
                print("YOLO modeli yüklenemedi, background subtraction kullanılacak")
                self.use_yolo = False
    
    def extract_features_from_video(self, video_path: str) -> Dict[str, float]:
        """
        Bir video dosyasından özellikler çıkarır.
        
        Returns:
            Dict: Çıkarılan özellikler
        """
        if not os.path.exists(video_path):
            # Video yoksa varsayılan özellikler döndür
            return self._get_default_features()
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return self._get_default_features()
        
        # Video özellikleri
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Özellik hesaplama
        vehicle_counts = []
        movement_scores = []
        density_scores = []
        
        frame_count = 0
        prev_frame = None
        
        # Her N framede bir işle (performans için)
        sample_rate = max(1, int(fps / 2))  # Saniyede ~2 frame
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % sample_rate != 0:
                continue
            
            # Frame işleme
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Hareket tespiti
            if prev_frame is not None:
                frame_diff = cv2.absdiff(prev_frame, blurred)
                _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                
                # Hareket skoru
                movement_score = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1])
                movement_scores.append(movement_score)
            
            # Background subtraction ile araç tespiti
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Morfolojik işlemler
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Konturları bul (araçları temsil eder)
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Belirli boyuttan büyük konturları say (araçlar)
            min_area = 500  # Minimum araç alanı
            vehicles = [c for c in contours if cv2.contourArea(c) > min_area]
            vehicle_counts.append(len(vehicles))
            
            # Yoğunluk skoru (beyaz piksel oranı)
            density = np.sum(fg_mask > 0) / (fg_mask.shape[0] * fg_mask.shape[1])
            density_scores.append(density)
            
            prev_frame = blurred
        
        cap.release()
        
        # Özellikleri hesapla
        features = self._compute_statistical_features(
            vehicle_counts, movement_scores, density_scores
        )
        
        return features
    
    def _compute_statistical_features(
        self, 
        vehicle_counts: List[int],
        movement_scores: List[float],
        density_scores: List[float]
    ) -> Dict[str, float]:
        """İstatistiksel özellikleri hesaplar."""
        
        features = {}
        
        # Araç sayısı özellikleri
        if vehicle_counts:
            features['vehicle_count_mean'] = np.mean(vehicle_counts)
            features['vehicle_count_max'] = np.max(vehicle_counts)
            features['vehicle_count_min'] = np.min(vehicle_counts)
            features['vehicle_count_std'] = np.std(vehicle_counts)
            features['vehicle_count_median'] = np.median(vehicle_counts)
        else:
            features.update({
                'vehicle_count_mean': 0, 'vehicle_count_max': 0,
                'vehicle_count_min': 0, 'vehicle_count_std': 0,
                'vehicle_count_median': 0
            })
        
        # Hareket özellikleri
        if movement_scores:
            features['movement_mean'] = np.mean(movement_scores)
            features['movement_max'] = np.max(movement_scores)
            features['movement_std'] = np.std(movement_scores)
        else:
            features.update({
                'movement_mean': 0, 'movement_max': 0, 'movement_std': 0
            })
        
        # Yoğunluk özellikleri
        if density_scores:
            features['density_mean'] = np.mean(density_scores)
            features['density_max'] = np.max(density_scores)
            features['density_std'] = np.std(density_scores)
        else:
            features.update({
                'density_mean': 0, 'density_max': 0, 'density_std': 0
            })
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Video okunamazsa varsayılan özellikler."""
        return {
            'vehicle_count_mean': 0, 'vehicle_count_max': 0,
            'vehicle_count_min': 0, 'vehicle_count_std': 0,
            'vehicle_count_median': 0, 'movement_mean': 0,
            'movement_max': 0, 'movement_std': 0,
            'density_mean': 0, 'density_max': 0, 'density_std': 0
        }


class TemporalFeatureEngineer:
    """
    Zaman bazlı özellikler ve geçmiş bilgilerden özellikler çıkarır.
    Gerçek zamanlı kısıtlamalara uygun şekilde tasarlanmıştır.
    """
    
    def __init__(self, lookback_window: int = 15):
        """
        Args:
            lookback_window: Geçmişe bakış penceresi (dakika)
        """
        self.lookback_window = lookback_window
        self.history_buffer = deque(maxlen=lookback_window)
    
    def add_temporal_features(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Dataframe'e zaman bazlı özellikler ekler.
        
        Args:
            df: Video özelliklerini içeren dataframe
            
        Returns:
            Zenginleştirilmiş dataframe
        """
        df = df.copy()
        
        # Zaman özelliklerini çıkar
        df['datetime'] = pd.to_datetime(df['video_time'])
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Saat bazlı kategoriler (rush hour vs normal)
        df['is_rush_hour'] = df['hour'].apply(
            lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0
        )
        df['time_of_day'] = df['hour'].apply(self._categorize_time_of_day)
        
        # Sinyal kullanımı encoding
        signal_mapping = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
        df['signaling_encoded'] = df['signaling'].map(signal_mapping).fillna(0)
        
        # Döngüsel zaman özellikleri (saat ve dakika için)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        
        return df
    
    def add_lagged_features(
        self, 
        df: pd.DataFrame, 
        value_columns: List[str],
        lags: List[int] = [1, 2, 3, 5, 10, 15]
    ) -> pd.DataFrame:
        """
        Geçmiş değerlerden özellikler ekler (lagged features).
        SADECE geçmiş verileri kullanır (gerçek zamanlı uyumlu).
        """
        df = df.copy()
        
        for col in value_columns:
            if col not in df.columns:
                continue
            
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        value_columns: List[str],
        windows: List[int] = [3, 5, 10, 15]
    ) -> pd.DataFrame:
        """
        Hareketli pencere istatistikleri ekler.
        """
        df = df.copy()
        
        for col in value_columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                # Hareketli ortalama
                df[f'{col}_rolling_mean_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )
                
                # Hareketli standart sapma
                df[f'{col}_rolling_std_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).std()
                )
                
                # Trend (değişim)
                df[f'{col}_rolling_trend_{window}'] = (
                    df[col] - df[col].shift(window)
                )
        
        return df
    
    @staticmethod
    def _categorize_time_of_day(hour: int) -> int:
        """Günün saatini kategorize eder."""
        if 6 <= hour < 12:
            return 1  # Sabah
        elif 12 <= hour < 18:
            return 2  # Öğleden sonra
        elif 18 <= hour < 22:
            return 3  # Akşam
        else:
            return 0  # Gece


class CongestionPredictor:
    """
    Tıkanıklık seviyesi tahmin modeli.
    Gerçek zamanlı kısıtlamalara uygun sıralı tahmin yapar.
    """
    
    def __init__(self):
        self.model_enter = None
        self.model_exit = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.feature_importance = {}
        
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        video_base_path: str = "videos"
    ) -> pd.DataFrame:
        """
        Eğitim verisini hazırlar (özellik çıkarma + mühendislik).
        """
        print("Video özelliklerini çıkarıyorum...")
        
        # Video feature extractor
        extractor = VideoFeatureExtractor(use_yolo=False)
        
        # Her video için özellik çıkar
        video_features_list = []
        
        for idx, row in df.iterrows():
            video_rel_path = row['videos']
            video_full_path = os.path.join(video_base_path, video_rel_path)
            
            # Özellikleri çıkar
            features = extractor.extract_features_from_video(video_full_path)
            features['time_segment_id'] = row['time_segment_id']
            video_features_list.append(features)
            
            if (idx + 1) % 100 == 0:
                print(f"  İşlenen video: {idx + 1}/{len(df)}")
        
        # Video özelliklerini birleştir
        video_features_df = pd.DataFrame(video_features_list)
        df = df.merge(
            video_features_df, 
            on='time_segment_id', 
            how='left'
        )
        
        # Zaman bazlı özellikler ekle
        print("Zaman bazlı özellikler ekleniyor...")
        temporal_engineer = TemporalFeatureEngineer()
        df = temporal_engineer.add_temporal_features(df)
        
        # Video özellik sütunları
        video_feat_cols = list(video_features_df.columns)
        video_feat_cols.remove('time_segment_id')
        
        # Lagged features
        df = df.sort_values('time_segment_id').reset_index(drop=True)
        df = temporal_engineer.add_lagged_features(df, video_feat_cols, lags=[1, 2, 3, 5])
        
        # Rolling features
        df = temporal_engineer.add_rolling_features(df, video_feat_cols, windows=[3, 5, 10])
        
        return df
    
    def train(
        self,
        train_df: pd.DataFrame,
        target_enter_col: str = 'congestion_enter_rating',
        target_exit_col: str = 'congestion_exit_rating'
    ):
        """
        Modeli eğitir.
        """
        print("Model eğitimi başlıyor...")
        
        # Özellik sütunlarını belirle
        exclude_cols = [
            'responseId', 'view_label', 'ID_enter', 'ID_exit', 'videos',
            'video_time', 'datetimestamp_start', 'datetimestamp_end',
            'date', 'congestion_enter_rating', 'congestion_exit_rating',
            'cycle_phase', 'datetime', 'signaling'
        ]
        
        self.feature_columns = [
            col for col in train_df.columns 
            if col not in exclude_cols
        ]
        
        # NaN değerleri doldur
        X = train_df[self.feature_columns].fillna(0)
        
        # Hedef değişkenleri encode et
        y_enter = self.label_encoder.fit_transform(train_df[target_enter_col])
        y_exit = self.label_encoder.transform(train_df[target_exit_col])
        
        # Enter modeli
        print("Enter (giriş) modeli eğitiliyor...")
        self.model_enter = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            subsample=0.8
        )
        self.model_enter.fit(X, y_enter)
        
        # Exit modeli
        print("Exit (çıkış) modeli eğitiliyor...")
        self.model_exit = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            subsample=0.8
        )
        self.model_exit.fit(X, y_exit)
        
        # Özellik önemi
        self.feature_importance = {
            'enter': dict(zip(
                self.feature_columns,
                self.model_enter.feature_importances_
            )),
            'exit': dict(zip(
                self.feature_columns,
                self.model_exit.feature_importances_
            ))
        }
        
        print("Model eğitimi tamamlandı!")
        
        # Eğitim accuracy
        y_pred_enter = self.model_enter.predict(X)
        y_pred_exit = self.model_exit.predict(X)
        
        print(f"\nEğitim Accuracy:")
        print(f"  Enter: {accuracy_score(y_enter, y_pred_enter):.4f}")
        print(f"  Exit: {accuracy_score(y_exit, y_pred_exit):.4f}")
    
    def predict(
        self,
        test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Test verisi üzerinde tahmin yapar.
        """
        X = test_df[self.feature_columns].fillna(0)
        
        # Tahminler
        pred_enter = self.model_enter.predict(X)
        pred_exit = self.model_exit.predict(X)
        
        # Decode et
        pred_enter_labels = self.label_encoder.inverse_transform(pred_enter)
        pred_exit_labels = self.label_encoder.inverse_transform(pred_exit)
        
        # Sonuçları ekle
        result_df = test_df.copy()
        result_df['predicted_enter'] = pred_enter_labels
        result_df['predicted_exit'] = pred_exit_labels
        
        return result_df
    
    def save_model(self, path: str = "congestion_model.pkl"):
        """Modeli kaydeder."""
        model_data = {
            'model_enter': self.model_enter,
            'model_exit': self.model_exit,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, path)
        print(f"Model kaydedildi: {path}")
    
    def load_model(self, path: str = "congestion_model.pkl"):
        """Modeli yükler."""
        model_data = joblib.load(path)
        self.model_enter = model_data['model_enter']
        self.model_exit = model_data['model_exit']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.feature_importance = model_data.get('feature_importance', {})
        print(f"Model yüklendi: {path}")
    
    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """En önemli özellikleri döndürür."""
        if not self.feature_importance:
            return pd.DataFrame()
        
        # Enter için en önemli özellikler
        enter_imp = sorted(
            self.feature_importance['enter'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        # Exit için en önemli özellikler
        exit_imp = sorted(
            self.feature_importance['exit'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        # DataFrame oluştur
        importance_df = pd.DataFrame({
            'Feature_Enter': [x[0] for x in enter_imp],
            'Importance_Enter': [x[1] for x in enter_imp],
            'Feature_Exit': [x[0] for x in exit_imp],
            'Importance_Exit': [x[1] for x in exit_imp]
        })
        
        return importance_df


class RealTimeTestProcessor:
    """
    Gerçek zamanlı test işleme:
    - 15 dakika input alır
    - 2 dakika embargo uygular
    - Sonraki 5 dakika için tahmin yapar
    """
    
    def __init__(self, predictor: CongestionPredictor):
        self.predictor = predictor
        self.embargo_minutes = 2
        self.prediction_window = 5
        self.input_window = 15
    
    def process_test_segments(
        self,
        test_df: pd.DataFrame,
        cycle_phases: List[str]
    ) -> pd.DataFrame:
        """
        Test segmentlerini gerçek zamanlı kısıtlamalarla işler.
        """
        predictions = []
        
        for phase in cycle_phases:
            # Bu faz için verileri filtrele
            phase_data = test_df[test_df['cycle_phase'] == phase].copy()
            
            if len(phase_data) == 0:
                continue
            
            # time_segment_id'ye göre sırala
            phase_data = phase_data.sort_values('time_segment_id')
            
            # Input penceresi (15 dakika)
            input_data = phase_data.head(self.input_window)
            
            # 2 dakika embargo sonrası başlangıç
            start_pred_idx = self.input_window + self.embargo_minutes
            
            # Tahmin yapılacak veriler (5 dakika)
            pred_indices = range(
                start_pred_idx,
                min(start_pred_idx + self.prediction_window, len(phase_data))
            )
            
            for pred_idx in pred_indices:
                # Şu ana kadar olan tüm verileri kullan (gelecek hariç)
                available_data = phase_data.iloc[:pred_idx]
                
                # Tahmin yap
                current_row = phase_data.iloc[pred_idx:pred_idx+1]
                pred = self.predictor.predict(current_row)
                
                predictions.append({
                    'ID_enter': current_row['ID_enter'].values[0],
                    'ID_exit': current_row['ID_exit'].values[0],
                    'predicted_enter': pred['predicted_enter'].values[0],
                    'predicted_exit': pred['predicted_exit'].values[0],
                    'time_segment_id': current_row['time_segment_id'].values[0]
                })
        
        return pd.DataFrame(predictions)


def create_submission_file(
    predictions_df: pd.DataFrame,
    sample_submission_path: str,
    output_path: str = "submission.csv"
):
    """
    Yarışma formatında submission dosyası oluşturur.
    """
    # Sample submission'ı oku
    sample_sub = pd.read_csv(sample_submission_path)
    
    # Prediction dictionary oluştur
    pred_dict_enter = dict(zip(
        predictions_df['ID_enter'],
        predictions_df['predicted_enter']
    ))
    pred_dict_exit = dict(zip(
        predictions_df['ID_exit'],
        predictions_df['predicted_exit']
    ))
    
    # Submission dosyasını doldur
    submission = sample_sub.copy()
    
    for idx, row in submission.iterrows():
        id_val = row['ID']
        
        if id_val in pred_dict_enter:
            submission.loc[idx, 'Target'] = pred_dict_enter[id_val]
            submission.loc[idx, 'Target_Accuracy'] = pred_dict_enter[id_val]
        elif id_val in pred_dict_exit:
            submission.loc[idx, 'Target'] = pred_dict_exit[id_val]
            submission.loc[idx, 'Target_Accuracy'] = pred_dict_exit[id_val]
    
    submission.to_csv(output_path, index=False)
    print(f"Submission dosyası oluşturuldu: {output_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("Barbados Traffic Congestion Analysis - Çözüm Pipeline")
    print("=" * 80)
    
    # Veri yolları
    train_csv = "Train.csv"
    test_csv = "TestInputSegments.csv"
    sample_sub = "SampleSubmission.csv"
    video_base_path = "videos"
    
    # Eğitim verisini yükle
    print("\n1. Eğitim verisi yükleniyor...")
    train_df = pd.read_csv(train_csv)
    print(f"   Eğitim örnekleri: {len(train_df)}")
    
    # Model oluştur ve eğit
    print("\n2. Model eğitimi başlıyor...")
    predictor = CongestionPredictor()
    
    # Küçük bir subset ile test (tüm veriyi işlemek uzun sürer)
    # Gerçek çalıştırmada tüm veriyi kullanın
    train_subset = train_df.head(500)  # İlk 500 örnek
    
    train_prepared = predictor.prepare_training_data(
        train_subset,
        video_base_path=video_base_path
    )
    
    predictor.train(train_prepared)
    
    # Modeli kaydet
    predictor.save_model("congestion_model.pkl")
    
    # En önemli özellikleri göster
    print("\n3. En önemli özellikler:")
    top_features = predictor.get_top_features(n=20)
    print(top_features.to_string())
    
    # Özellik önem raporu oluştur
    top_features.to_csv("feature_importance_report.csv", index=False)
    print("\n   Özellik önem raporu kaydedildi: feature_importance_report.csv")
    
    print("\n" + "=" * 80)
    print("Pipeline tamamlandı!")
    print("=" * 80)
