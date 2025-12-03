"""
YOLOv8 Vehicle Detection System for Traffic Analysis
Provides advanced vehicle detection, classification, and tracking
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics paketi yüklü değil. Pip ile yükleyin: pip install ultralytics")


class YOLOVehicleDetector:
    """
    YOLOv8-based vehicle detector with classification and tracking
    """
    
    # COCO dataset vehicle class IDs
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, model_size: str = 'n', confidence: float = 0.25,
                 iou_threshold: float = 0.45, device: str = 'cpu'):
        """
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
            confidence: Minimum confidence threshold (0-1)
            iou_threshold: IoU threshold for NMS
            device: 'cpu' or 'cuda'
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics paketi gerekli. pip install ultralytics")
        
        self.model_size = model_size
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model"""
        model_name = f'yolov8{self.model_size}.pt'
        print(f"🔧 YOLOv8{self.model_size.upper()} modeli yükleniyor...")
        
        try:
            self.model = YOLO(model_name)
            self.model.to(self.device)
            print(f"✓ Model yüklendi: {model_name} (Device: {self.device})")
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            raise
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in a single frame
        
        Args:
            frame: Input video frame (BGR)
            
        Returns:
            List of detections with bbox, class, confidence
        """
        if self.model is None:
            raise ValueError("Model yüklenmedi!")
        
        # Run inference
        results = self.model(frame, conf=self.confidence, iou=self.iou_threshold, 
                           verbose=False)[0]
        
        detections = []
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            
            # Filter only vehicles
            if class_id in self.VEHICLE_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'class_id': class_id,
                    'class_name': self.VEHICLE_CLASSES[class_id],
                    'confidence': conf,
                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    'area': (x2 - x1) * (y2 - y1)
                })
        
        return detections
    
    def process_video(self, video_path: str, max_frames: Optional[int] = None,
                     sample_rate: int = 1) -> Dict:
        """
        Process entire video and extract vehicle statistics
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all)
            sample_rate: Process every Nth frame
            
        Returns:
            Dictionary with aggregated statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"📹 Video işleniyor: {Path(video_path).name}")
        print(f"   FPS: {fps:.1f}, Toplam Frame: {total_frames}, Örnekleme: 1/{sample_rate}")
        
        # Statistics collectors
        vehicle_counts = []
        vehicle_types = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        confidence_scores = []
        vehicle_areas = []
        
        frame_idx = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break
            
            # Sample frames
            if frame_idx % sample_rate == 0:
                detections = self.detect_vehicles(frame)
                
                # Collect statistics
                vehicle_counts.append(len(detections))
                
                for det in detections:
                    vehicle_types[det['class_name']] += 1
                    confidence_scores.append(det['confidence'])
                    vehicle_areas.append(det['area'])
                
                processed_frames += 1
            
            frame_idx += 1
        
        cap.release()
        
        # Calculate aggregated features
        features = {
            # Count features
            'vehicle_count_mean': np.mean(vehicle_counts) if vehicle_counts else 0,
            'vehicle_count_std': np.std(vehicle_counts) if vehicle_counts else 0,
            'vehicle_count_max': np.max(vehicle_counts) if vehicle_counts else 0,
            'vehicle_count_min': np.min(vehicle_counts) if vehicle_counts else 0,
            
            # Type distribution
            'car_ratio': vehicle_types['car'] / max(sum(vehicle_types.values()), 1),
            'truck_bus_ratio': (vehicle_types['truck'] + vehicle_types['bus']) / max(sum(vehicle_types.values()), 1),
            'motorcycle_ratio': vehicle_types['motorcycle'] / max(sum(vehicle_types.values()), 1),
            
            # Confidence features
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'min_confidence': np.min(confidence_scores) if confidence_scores else 0,
            
            # Density features
            'avg_vehicle_area': np.mean(vehicle_areas) if vehicle_areas else 0,
            'total_vehicle_area': np.sum(vehicle_areas) if vehicle_areas else 0,
            
            # Processing info
            'frames_processed': processed_frames,
            'total_detections': sum(vehicle_types.values())
        }
        
        print(f"✓ İşlendi: {processed_frames} frame, {features['total_detections']} tespit")
        
        return features
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict],
                           show_labels: bool = True) -> np.ndarray:
        """
        Draw bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections from detect_vehicles()
            show_labels: Whether to show class labels
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            conf = det['confidence']
            
            # Color by vehicle type
            colors = {
                'car': (0, 255, 0),      # Green
                'motorcycle': (255, 0, 0),  # Blue
                'bus': (0, 165, 255),    # Orange
                'truck': (0, 0, 255)     # Red
            }
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_labels:
                label = f"{class_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw count
        count_text = f"Vehicles: {len(detections)}"
        cv2.putText(annotated, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated
    
    def save_annotated_video(self, video_path: str, output_path: str,
                           max_frames: Optional[int] = None):
        """
        Process video and save with annotations
        
        Args:
            video_path: Input video path
            output_path: Output video path
            max_frames: Maximum frames to process
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"💾 Annotated video kaydediliyor: {output_path}")
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break
            
            # Detect and annotate
            detections = self.detect_vehicles(frame)
            annotated = self.visualize_detections(frame, detections)
            
            out.write(annotated)
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                print(f"   İşlendi: {frame_idx} frame")
        
        cap.release()
        out.release()
        
        print(f"✓ Video kaydedildi: {frame_idx} frame")


def demo_yolo():
    """Demo function for YOLO detector"""
    print("\n" + "="*60)
    print("🎯 YOLO VEHICLE DETECTOR DEMO")
    print("="*60)
    
    if not YOLO_AVAILABLE:
        print("\n❌ ultralytics paketi bulunamadı!")
        print("Yüklemek için: pip install ultralytics")
        return
    
    # Check for video files
    video_dir = Path('videos/normanniles1')
    
    if not video_dir.exists():
        print(f"\n⚠️  Video klasörü bulunamadı: {video_dir}")
        print("Demo sentetik örnek ile devam ediyor...\n")
        
        # Demonstrate YOLO setup
        print("📋 YOLO Model Seçenekleri:")
        print("   - yolov8n.pt: Nano (en hızlı, 3.2M parametreler)")
        print("   - yolov8s.pt: Small (hızlı, 11.2M parametreler)")
        print("   - yolov8m.pt: Medium (dengeli, 25.9M parametreler)")
        print("   - yolov8l.pt: Large (yüksek doğruluk, 43.7M parametreler)")
        print("   - yolov8x.pt: XLarge (en yüksek doğruluk, 68.2M parametreler)")
        
        print("\n🎯 Önerilen Konfigürasyon:")
        print("   Model: yolov8n (nano) - Real-time için en uygun")
        print("   Confidence: 0.25")
        print("   IoU Threshold: 0.45")
        print("   Sample Rate: 2 (her 2 frame'de bir)")
        
        print("\n💡 Kullanım:")
        print("   from traffic_yolo_detector import YOLOVehicleDetector")
        print("   detector = YOLOVehicleDetector(model_size='n')")
        print("   features = detector.process_video('video.mp4')")
        
        return
    
    # Find video files
    video_files = list(video_dir.glob('*.mp4'))
    
    if not video_files:
        print(f"\n⚠️  Video dosyası bulunamadı: {video_dir}")
        return
    
    # Use first video for demo
    video_path = str(video_files[0])
    
    print(f"\n📹 Test Video: {Path(video_path).name}")
    
    # Initialize detector
    try:
        detector = YOLOVehicleDetector(model_size='n', confidence=0.25)
    except Exception as e:
        print(f"\n❌ YOLO yükleme hatası: {e}")
        return
    
    # Process video (sample rate=5 for speed)
    print("\n🔍 Video işleniyor (ilk 300 frame)...")
    features = detector.process_video(video_path, max_frames=300, sample_rate=5)
    
    # Display results
    print("\n📊 YOLO Özellikler:")
    print("="*60)
    for key, value in features.items():
        if isinstance(value, float):
            print(f"   {key:25s}: {value:.3f}")
        else:
            print(f"   {key:25s}: {value}")
    
    print("\n" + "="*60)
    print("✅ YOLO DEMO TAMAMLANDI!")
    print("="*60)


if __name__ == "__main__":
    demo_yolo()
