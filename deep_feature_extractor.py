"""
Deep Feature Extraction using Pre-trained CNNs
Extracts deep features from video frames for traffic analysis
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch yüklü değil. Pip ile yükleyin: pip install torch torchvision")


class DeepFeatureExtractor:
    """
    Extract deep features from video frames using pre-trained CNNs
    """
    
    def __init__(self, model_name: str = 'resnet18', device: str = 'cpu'):
        """
        Args:
            model_name: 'resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v2'
            device: 'cpu' or 'cuda'
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch gerekli. pip install torch torchvision")
        
        self.model_name = model_name
        self.device = device
        self.model = None
        self.transform = None
        self.feature_dim = None
        
        self._load_model()
        self._setup_transform()
    
    def _load_model(self):
        """Load pre-trained model"""
        print(f"🔧 {self.model_name.upper()} modeli yükleniyor...")
        
        if self.model_name == 'resnet18':
            self.model = models.resnet18(weights='DEFAULT')
            self.feature_dim = 512
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(weights='DEFAULT')
            self.feature_dim = 2048
        elif self.model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights='DEFAULT')
            self.feature_dim = 1280
        elif self.model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(weights='DEFAULT')
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Remove classification layer
        if 'resnet' in self.model_name:
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        elif 'efficientnet' in self.model_name:
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        elif 'mobilenet' in self.model_name:
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model yüklendi: {self.model_name} (Feature dim: {self.feature_dim})")
    
    def _setup_transform(self):
        """Setup image preprocessing"""
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features from a single frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Feature vector (1D numpy array)
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        
        # Preprocess
        input_tensor = self.transform(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_batch)
        
        # Flatten
        features = features.squeeze().cpu().numpy()
        
        return features
    
    def process_video(self, video_path: str, max_frames: Optional[int] = None,
                     sample_rate: int = 30, aggregation: str = 'mean') -> Dict:
        """
        Process entire video and extract aggregated features
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            sample_rate: Process every Nth frame (default 30 = 1 per second)
            aggregation: 'mean', 'max', 'std', or 'all'
            
        Returns:
            Dictionary with aggregated deep features
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"📹 Deep Feature Extraction: {Path(video_path).name}")
        print(f"   FPS: {fps:.1f}, Frames: {total_frames}, Sample Rate: 1/{sample_rate}")
        
        # Feature collector
        features_list = []
        
        frame_idx = 0
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % sample_rate == 0:
                features = self.extract_frame_features(frame)
                features_list.append(features)
            
            frame_idx += 1
        
        cap.release()
        
        if not features_list:
            return self._empty_features()
        
        # Convert to numpy array
        features_array = np.array(features_list)  # (n_frames, feature_dim)
        
        # Aggregate features
        result = {}
        
        if aggregation in ['mean', 'all']:
            mean_features = np.mean(features_array, axis=0)
            for i, val in enumerate(mean_features):
                result[f'deep_feat_mean_{i}'] = float(val)
        
        if aggregation in ['std', 'all']:
            std_features = np.std(features_array, axis=0)
            for i, val in enumerate(std_features):
                result[f'deep_feat_std_{i}'] = float(val)
        
        if aggregation in ['max', 'all']:
            max_features = np.max(features_array, axis=0)
            for i, val in enumerate(max_features):
                result[f'deep_feat_max_{i}'] = float(val)
        
        # Add summary statistics
        result['deep_feat_frames_processed'] = len(features_list)
        result['deep_feat_temporal_variance'] = float(np.var(features_array))
        
        print(f"✓ İşlendi: {len(features_list)} frame")
        print(f"   Feature Dimension: {self.feature_dim}")
        print(f"   Total Features: {len(result)}")
        
        return result
    
    def _empty_features(self) -> Dict:
        """Return empty features when extraction fails"""
        result = {}
        for i in range(self.feature_dim):
            result[f'deep_feat_mean_{i}'] = 0.0
        result['deep_feat_frames_processed'] = 0
        result['deep_feat_temporal_variance'] = 0.0
        return result
    
    def extract_video_embeddings(self, video_path: str, 
                                sample_rate: int = 30) -> np.ndarray:
        """
        Extract frame-by-frame embeddings for temporal analysis
        
        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame
            
        Returns:
            Embeddings array (n_frames, feature_dim)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")
        
        embeddings = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                features = self.extract_frame_features(frame)
                embeddings.append(features)
            
            frame_idx += 1
        
        cap.release()
        
        return np.array(embeddings)


def demo_deep_features():
    """Demo function for deep feature extraction"""
    print("\n" + "="*60)
    print("🎯 DEEP FEATURE EXTRACTION DEMO")
    print("="*60)
    
    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch paketi bulunamadı!")
        print("Yüklemek için: pip install torch torchvision")
        return
    
    # Check for video files
    video_dir = Path('videos/normanniles1')
    
    if not video_dir.exists() or not list(video_dir.glob('*.mp4')):
        print(f"\n⚠️  Video dosyası bulunamadı: {video_dir}")
        print("Demo sentetik örnek ile devam ediyor...\n")
        
        print("📋 Desteklenen Modeller:")
        print("   - ResNet18 (512-dim features)")
        print("   - ResNet50 (2048-dim features)")
        print("   - EfficientNet-B0 (1280-dim features)")
        print("   - MobileNet-V2 (1280-dim features)")
        
        print("\n🎯 Özellik Agregasyonu:")
        print("   • Mean: Ortalama features")
        print("   • Std: Standart sapma")
        print("   • Max: Maksimum features")
        print("   • All: Hepsi birlikte")
        
        print("\n💡 Kullanım:")
        print("   from deep_feature_extractor import DeepFeatureExtractor")
        print("   extractor = DeepFeatureExtractor(model_name='resnet18')")
        print("   features = extractor.process_video('video.mp4')")
        
        print("\n🚀 Avantajlar:")
        print("   • Transfer learning ile güçlü features")
        print("   • Pre-trained ImageNet weights")
        print("   • Yüksek seviye semantik bilgi")
        print("   • 512-2048 boyutlu zengin temsil")
        
        return
    
    # Find video files
    video_files = list(video_dir.glob('*.mp4'))
    video_path = str(video_files[0])
    
    print(f"\n📹 Test Video: {Path(video_path).name}")
    
    # Test with lightweight model
    try:
        extractor = DeepFeatureExtractor(model_name='mobilenet_v2', device='cpu')
    except Exception as e:
        print(f"\n❌ Model yükleme hatası: {e}")
        return
    
    # Process video (sample every 60 frames for speed)
    print("\n🔍 Video işleniyor (sample_rate=60)...")
    features = extractor.process_video(video_path, max_frames=300, 
                                      sample_rate=60, aggregation='mean')
    
    # Display sample features
    print("\n📊 Deep Features (ilk 10):")
    print("="*60)
    for i, (key, value) in enumerate(list(features.items())[:10]):
        print(f"   {key}: {value:.6f}")
    print(f"   ... ({len(features)-10} more features)")
    
    print("\n" + "="*60)
    print("✅ DEEP FEATURE EXTRACTION TAMAMLANDI!")
    print("="*60)


if __name__ == "__main__":
    demo_deep_features()
