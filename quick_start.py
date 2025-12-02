"""
Hızlı başlangıç scripti - Örnek kullanım
NOT: Gerçek video dosyaları gereklidir
"""

import os
import pandas as pd
from traffic_analysis_solution import CongestionPredictor
import warnings
warnings.filterwarnings('ignore')


def quick_start_demo():
    """
    Hızlı demo - veri var mı kontrol et ve temel iş akışını göster
    """
    print("=" * 80)
    print("BARBADOS TRAFFIC ANALYSIS - HIZLI BAŞLANGIÇ")
    print("=" * 80)
    
    # Veri dosyalarını kontrol et
    print("\n1. Veri dosyaları kontrol ediliyor...")
    
    required_files = {
        'Train.csv': 'Eğitim verisi',
        'TestInputSegments.csv': 'Test input verisi',
        'SampleSubmission.csv': 'Örnek submission formatı'
    }
    
    all_exists = True
    for file, desc in required_files.items():
        if os.path.exists(file):
            print(f"   ✓ {file} - {desc}")
        else:
            print(f"   ✗ {file} - {desc} BULUNAMADI!")
            all_exists = False
    
    # Video klasörünü kontrol et
    video_dir = "videos"
    if os.path.exists(video_dir):
        video_count = sum([len(files) for _, _, files in os.walk(video_dir)])
        print(f"   ✓ {video_dir}/ klasörü - {video_count} dosya")
    else:
        print(f"   ✗ {video_dir}/ klasörü BULUNAMADI!")
        print(f"      Not: Video dosyalarını {video_dir}/ klasörüne koyun")
        all_exists = False
    
    if not all_exists:
        print("\n⚠️  Eksik dosyalar var! Lütfen tüm gerekli dosyaları hazırlayın.")
        print("\nGerekli yapı:")
        print("  ├── Train.csv")
        print("  ├── TestInputSegments.csv")
        print("  ├── SampleSubmission.csv")
        print("  └── videos/")
        print("      └── normanniles1/")
        print("          ├── normanniles1_2025-10-20-06-00-45.mp4")
        print("          └── ...")
        return
    
    # Veriyi incele
    print("\n2. Veri istatistikleri...")
    train_df = pd.read_csv('Train.csv')
    test_df = pd.read_csv('TestInputSegments.csv')
    
    print(f"   Eğitim örnekleri: {len(train_df)}")
    print(f"   Test örnekleri: {len(test_df)}")
    
    # Sınıf dağılımı
    print("\n   Eğitim - Enter sınıf dağılımı:")
    for label, count in train_df['congestion_enter_rating'].value_counts().items():
        pct = 100 * count / len(train_df)
        print(f"     {label:20s}: {count:5d} ({pct:5.1f}%)")
    
    print("\n   Eğitim - Exit sınıf dağılımı:")
    for label, count in train_df['congestion_exit_rating'].value_counts().items():
        pct = 100 * count / len(train_df)
        print(f"     {label:20s}: {count:5d} ({pct:5.1f}%)")
    
    # Örnek eğitim
    print("\n3. Demo model eğitimi (ilk 500 örnek, tüm sınıflar dahil)...")
    print("   NOT: Tam eğitim için tüm veriyi kullanın!")
    
    choice = input("\n   Demo eğitimi başlatmak ister misiniz? (e/h): ").lower()
    
    if choice == 'e':
        print("\n   Başlatılıyor...")
        
        # Her sınıftan örnek içeren dengeli subset
        # Bu, modelin tüm sınıfları görmesini sağlar
        train_subset = []
        for label in train_df['congestion_enter_rating'].unique():
            label_samples = train_df[train_df['congestion_enter_rating'] == label].head(125)
            train_subset.append(label_samples)
        train_subset = pd.concat(train_subset).head(500)
        
        predictor = CongestionPredictor()
        
        print("   Video özellikleri çıkarılıyor...")
        train_prepared = predictor.prepare_training_data(
            train_subset,
            video_base_path="videos"
        )
        
        print("   Model eğitiliyor...")
        predictor.train(train_prepared)
        
        print("   Model kaydediliyor...")
        predictor.save_model("demo_model.pkl")
        
        print("\n   ✓ Demo eğitimi tamamlandı!")
        print("     Model: demo_model.pkl")
        
        # Özellik önemi
        top_feats = predictor.get_top_features(n=10)
        print("\n   En önemli 10 özellik:")
        print(top_feats[['Feature_Enter', 'Importance_Enter']].to_string(index=False))
    
    print("\n4. Sonraki adımlar:")
    print("   a) Tam eğitim için:")
    print("      python traffic_analysis_solution.py")
    print("   ")
    print("   b) Test tahmini için:")
    print("      python test_prediction.py")
    print("   ")
    print("   c) Detaylı bilgi için:")
    print("      README_TR.md dosyasını okuyun")
    
    print("\n" + "=" * 80)
    print("HIZLI BAŞLANGIÇ TAMAMLANDI")
    print("=" * 80)


if __name__ == "__main__":
    quick_start_demo()
