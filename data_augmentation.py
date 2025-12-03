"""
Data Augmentation System for Traffic Analysis
Includes video augmentation and SMOTE for class balancing
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')


class VideoAugmentor:
    """
    Video augmentation for increasing training data diversity
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
    
    def brightness_adjust(self, frame: np.ndarray, factor: float = None) -> np.ndarray:
        """
        Adjust brightness
        
        Args:
            frame: Input frame (BGR)
            factor: Brightness factor (0.5-1.5), random if None
            
        Returns:
            Augmented frame
        """
        if factor is None:
            factor = np.random.uniform(0.7, 1.3)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def contrast_adjust(self, frame: np.ndarray, alpha: float = None, 
                       beta: int = None) -> np.ndarray:
        """
        Adjust contrast and brightness
        
        Args:
            frame: Input frame (BGR)
            alpha: Contrast (1.0-3.0), random if None
            beta: Brightness (-50 to 50), random if None
            
        Returns:
            Augmented frame
        """
        if alpha is None:
            alpha = np.random.uniform(0.8, 1.4)
        if beta is None:
            beta = np.random.randint(-30, 30)
        
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    def add_noise(self, frame: np.ndarray, noise_type: str = 'gaussian',
                 intensity: float = None) -> np.ndarray:
        """
        Add noise to frame
        
        Args:
            frame: Input frame (BGR)
            noise_type: 'gaussian' or 'salt_pepper'
            intensity: Noise intensity (0.0-1.0), random if None
            
        Returns:
            Noisy frame
        """
        if intensity is None:
            intensity = np.random.uniform(0.01, 0.05)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, intensity * 255, frame.shape)
            noisy = frame.astype(np.float32) + noise
            return np.clip(noisy, 0, 255).astype(np.uint8)
        
        elif noise_type == 'salt_pepper':
            noisy = frame.copy()
            # Salt
            num_salt = int(intensity * frame.size * 0.5)
            coords = [np.random.randint(0, i, num_salt) for i in frame.shape[:2]]
            noisy[coords[0], coords[1], :] = 255
            
            # Pepper
            num_pepper = int(intensity * frame.size * 0.5)
            coords = [np.random.randint(0, i, num_pepper) for i in frame.shape[:2]]
            noisy[coords[0], coords[1], :] = 0
            
            return noisy
        
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
    
    def blur(self, frame: np.ndarray, blur_type: str = 'gaussian',
            kernel_size: int = None) -> np.ndarray:
        """
        Apply blur
        
        Args:
            frame: Input frame (BGR)
            blur_type: 'gaussian', 'median', or 'motion'
            kernel_size: Kernel size (odd number), random if None
            
        Returns:
            Blurred frame
        """
        if kernel_size is None:
            kernel_size = np.random.choice([3, 5, 7])
        
        if blur_type == 'gaussian':
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        elif blur_type == 'median':
            return cv2.medianBlur(frame, kernel_size)
        elif blur_type == 'motion':
            # Motion blur kernel
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            return cv2.filter2D(frame, -1, kernel)
        else:
            raise ValueError(f"Unknown blur_type: {blur_type}")
    
    def flip(self, frame: np.ndarray, mode: int = None) -> np.ndarray:
        """
        Flip frame
        
        Args:
            frame: Input frame (BGR)
            mode: 0=vertical, 1=horizontal, -1=both, random if None
            
        Returns:
            Flipped frame
        """
        if mode is None:
            mode = np.random.choice([0, 1, -1])
        
        return cv2.flip(frame, mode)
    
    def augment_frame(self, frame: np.ndarray, 
                     augmentations: List[str] = None) -> np.ndarray:
        """
        Apply multiple augmentations to a frame
        
        Args:
            frame: Input frame (BGR)
            augmentations: List of augmentation names, random if None
            
        Returns:
            Augmented frame
        """
        if augmentations is None:
            # Randomly select 1-3 augmentations
            all_augs = ['brightness', 'contrast', 'noise', 'blur']
            n_augs = np.random.randint(1, 4)
            augmentations = np.random.choice(all_augs, n_augs, replace=False)
        
        result = frame.copy()
        
        for aug in augmentations:
            if aug == 'brightness':
                result = self.brightness_adjust(result)
            elif aug == 'contrast':
                result = self.contrast_adjust(result)
            elif aug == 'noise':
                result = self.add_noise(result, 
                                      noise_type=np.random.choice(['gaussian', 'salt_pepper']))
            elif aug == 'blur':
                result = self.blur(result, 
                                  blur_type=np.random.choice(['gaussian', 'median']))
            elif aug == 'flip':
                result = self.flip(result)
        
        return result
    
    def augment_video(self, video_path: str, output_path: str,
                     n_augmentations: int = 3):
        """
        Create augmented versions of a video
        
        Args:
            video_path: Input video path
            output_path: Output video path (will add suffix)
            n_augmentations: Number of augmented versions to create
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Read all frames first
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        print(f"📹 Video Augmentation: {Path(video_path).name}")
        print(f"   Original Frames: {len(frames)}")
        print(f"   Creating {n_augmentations} augmented versions...")
        
        # Create augmented versions
        output_base = Path(output_path).stem
        output_dir = Path(output_path).parent
        output_ext = Path(output_path).suffix
        
        for i in range(n_augmentations):
            out_path = output_dir / f"{output_base}_aug{i+1}{output_ext}"
            out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            
            for frame in frames:
                aug_frame = self.augment_frame(frame)
                out.write(aug_frame)
            
            out.release()
            print(f"   ✓ Created: {out_path.name}")


class FeatureAugmentor:
    """
    Feature-level augmentation using SMOTE and variants
    """
    
    def __init__(self, method: str = 'smote', random_state: int = 42):
        """
        Args:
            method: 'smote', 'adasyn', or 'smote_tomek'
            random_state: Random seed
        """
        self.method = method
        self.random_state = random_state
        self.sampler = self._get_sampler()
    
    def _get_sampler(self):
        """Get the appropriate sampler"""
        if self.method == 'smote':
            return SMOTE(random_state=self.random_state, k_neighbors=5)
        elif self.method == 'adasyn':
            return ADASYN(random_state=self.random_state, n_neighbors=5)
        elif self.method == 'smote_tomek':
            return SMOTETomek(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def balance_classes(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance classes using oversampling
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            (X_resampled, y_resampled)
        """
        print(f"\n🔧 Class Balancing: {self.method.upper()}")
        print(f"   Original Distribution:")
        for label, count in y.value_counts().sort_index().items():
            print(f"      {label}: {count} ({count/len(y)*100:.1f}%)")
        
        # Resample
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        
        print(f"\n   Resampled Distribution:")
        for label, count in pd.Series(y_resampled).value_counts().sort_index().items():
            print(f"      {label}: {count} ({count/len(y_resampled)*100:.1f}%)")
        
        print(f"\n✓ Samples: {len(X)} → {len(X_resampled)} (+{len(X_resampled)-len(X)})")
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    def augment_features(self, X_train: pd.DataFrame, 
                        y_enter: pd.Series, y_exit: pd.Series,
                        balance_both: bool = True) -> Tuple:
        """
        Augment features for both targets
        
        Args:
            X_train: Training features
            y_enter: Enter labels
            y_exit: Exit labels
            balance_both: Whether to balance both targets
            
        Returns:
            (X_aug, y_enter_aug, y_exit_aug)
        """
        print("\n" + "="*60)
        print("🎯 FEATURE AUGMENTATION")
        print("="*60)
        
        # Balance enter labels
        print("\n📊 Enter Congestion:")
        X_aug, y_enter_aug = self.balance_classes(X_train, y_enter)
        
        if balance_both:
            # For exit, we need to align with enter resampling
            # Use enter's indices to get corresponding exit labels
            # This is a simplification; in practice, you might want to:
            # 1. Balance enter first
            # 2. Use those samples to train exit model
            # OR balance them independently
            
            print("\n📊 Exit Congestion:")
            # Get corresponding exit labels for augmented samples
            # Note: This assumes we're resampling based on enter labels
            y_exit_subset = y_exit.loc[X_train.index]
            
            # Resample exit independently
            X_aug_exit, y_exit_aug = self.balance_classes(X_train, y_exit_subset)
            
            # For simplicity, use enter-balanced data with original exit mapping
            # In production, you'd want more sophisticated alignment
            y_exit_aug = y_exit.loc[X_train.index].reindex(
                pd.Series(y_enter_aug).index, 
                method='nearest'
            )
        else:
            y_exit_aug = y_exit.loc[X_train.index]
        
        print("\n" + "="*60)
        print("✅ AUGMENTATION COMPLETE")
        print("="*60)
        
        return X_aug, pd.Series(y_enter_aug), pd.Series(y_exit_aug)


def demo_augmentation():
    """Demo function for data augmentation"""
    print("\n" + "="*60)
    print("🎯 DATA AUGMENTATION DEMO")
    print("="*60)
    
    # Load training data
    print("\n📁 Veri Yükleniyor...")
    train_df = pd.read_csv('Train.csv')
    
    print(f"✓ Toplam Örnek: {len(train_df)}")
    print(f"✓ Sınıf Dağılımı (Enter):")
    print(train_df['congestion_enter_rating'].value_counts().sort_index())
    
    # Generate synthetic features
    print("\n🔧 Sentetik Özellikler Oluşturuluyor...")
    np.random.seed(42)
    
    features = pd.DataFrame()
    for idx, row in train_df.head(5000).iterrows():  # Use subset for demo
        congestion_level = row['congestion_enter_rating']
        
        if congestion_level == 0:
            vehicle_count = np.random.randint(5, 15)
            speed = np.random.uniform(45, 60)
            density = np.random.uniform(0.1, 0.3)
        elif congestion_level == 1:
            vehicle_count = np.random.randint(12, 25)
            speed = np.random.uniform(30, 45)
            density = np.random.uniform(0.25, 0.5)
        elif congestion_level == 2:
            vehicle_count = np.random.randint(22, 38)
            speed = np.random.uniform(15, 30)
            density = np.random.uniform(0.45, 0.7)
        else:
            vehicle_count = np.random.randint(35, 55)
            speed = np.random.uniform(5, 18)
            density = np.random.uniform(0.65, 0.95)
        
        features.loc[idx, 'vehicle_count'] = vehicle_count
        features.loc[idx, 'avg_speed'] = speed
        features.loc[idx, 'traffic_density'] = density
        features.loc[idx, 'vehicle_variance'] = np.random.uniform(0, vehicle_count * 0.2)
    
    y_enter = train_df.loc[features.index, 'congestion_enter_rating']
    y_exit = train_df.loc[features.index, 'congestion_exit_rating']
    
    # Test different augmentation methods
    for method in ['smote', 'adasyn', 'smote_tomek']:
        print(f"\n{'='*60}")
        print(f"Testing {method.upper()}")
        print('='*60)
        
        augmentor = FeatureAugmentor(method=method)
        X_aug, y_enter_aug, y_exit_aug = augmentor.augment_features(
            features, y_enter, y_exit, balance_both=False
        )
        
        print(f"\nAugmented Feature Shape: {X_aug.shape}")
    
    # Video augmentation demo (if videos exist)
    video_dir = Path('videos/normanniles1')
    if video_dir.exists() and list(video_dir.glob('*.mp4')):
        print("\n" + "="*60)
        print("📹 VIDEO AUGMENTATION DEMO")
        print("="*60)
        
        video_files = list(video_dir.glob('*.mp4'))
        if video_files:
            print(f"\n✓ Found {len(video_files)} video files")
            print("Video augmentation özelliği hazır (demo'da çalıştırılmıyor)")
            print("\nKullanım:")
            print("  from data_augmentation import VideoAugmentor")
            print("  augmentor = VideoAugmentor()")
            print("  augmentor.augment_video('input.mp4', 'output.mp4', n_augmentations=3)")
    
    print("\n" + "="*60)
    print("✅ AUGMENTATION DEMO TAMAMLANDI!")
    print("="*60)


if __name__ == "__main__":
    # Check if imbalanced-learn is installed
    try:
        import imblearn
    except ImportError:
        print("⚠️  imbalanced-learn paketi yüklü değil!")
        print("Yüklemek için: pip install imbalanced-learn")
        exit(1)
    
    demo_augmentation()
