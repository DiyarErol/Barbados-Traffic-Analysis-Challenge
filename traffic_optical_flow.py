"""
Optical Flow Analysis for Traffic Speed and Movement Detection
Uses Farneback dense optical flow for real-time traffic analysis
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class OpticalFlowAnalyzer:
    """
    Dense optical flow analyzer for traffic speed estimation
    and movement pattern detection
    """
    
    def __init__(self, pyr_scale: float = 0.5, levels: int = 3,
                 winsize: int = 15, iterations: int = 3,
                 poly_n: int = 5, poly_sigma: float = 1.2):
        """
        Args:
            pyr_scale: Pyramid scale (< 1)
            levels: Number of pyramid layers
            winsize: Averaging window size
            iterations: Number of iterations at each pyramid level
            poly_n: Size of pixel neighborhood
            poly_sigma: Standard deviation of Gaussian for polynomial expansion
        """
        self.params = {
            'pyr_scale': pyr_scale,
            'levels': levels,
            'winsize': winsize,
            'iterations': iterations,
            'poly_n': poly_n,
            'poly_sigma': poly_sigma,
            'flags': 0
        }
        
    def calculate_flow(self, prev_gray: np.ndarray, 
                       curr_gray: np.ndarray) -> np.ndarray:
        """
        Calculate dense optical flow between two frames
        
        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)
            
        Returns:
            Flow field (height, width, 2) with dx, dy components
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            **self.params
        )
        
        return flow
    
    def flow_to_magnitude_angle(self, flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert flow field to magnitude and angle
        
        Args:
            flow: Flow field from calculate_flow()
            
        Returns:
            (magnitude, angle) in degrees
        """
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        return mag, ang
    
    def estimate_speed(self, flow: np.ndarray, fps: float = 30.0,
                      pixel_to_meter: float = 0.05) -> Dict:
        """
        Estimate traffic speed from optical flow
        
        Args:
            flow: Flow field
            fps: Video frame rate
            pixel_to_meter: Conversion factor (default: 0.05m per pixel)
            
        Returns:
            Dictionary with speed statistics
        """
        mag, ang = self.flow_to_magnitude_angle(flow)
        
        # Filter out low motion (noise)
        motion_threshold = 0.5
        valid_motion = mag > motion_threshold
        
        if not np.any(valid_motion):
            return {
                'avg_speed_mps': 0.0,
                'max_speed_mps': 0.0,
                'motion_percentage': 0.0,
                'avg_speed_kmh': 0.0,
                'max_speed_kmh': 0.0
            }
        
        # Calculate speeds in meters per second
        pixel_speed = mag[valid_motion]  # pixels per frame
        speed_mps = pixel_speed * fps * pixel_to_meter  # m/s
        
        avg_speed_mps = np.mean(speed_mps)
        max_speed_mps = np.percentile(speed_mps, 95)  # 95th percentile
        motion_percentage = np.sum(valid_motion) / valid_motion.size * 100
        
        return {
            'avg_speed_mps': float(avg_speed_mps),
            'max_speed_mps': float(max_speed_mps),
            'motion_percentage': float(motion_percentage),
            'avg_speed_kmh': float(avg_speed_mps * 3.6),
            'max_speed_kmh': float(max_speed_mps * 3.6),
            'motion_pixels': int(np.sum(valid_motion))
        }
    
    def detect_traffic_direction(self, flow: np.ndarray) -> Dict:
        """
        Detect dominant traffic flow directions
        
        Args:
            flow: Flow field
            
        Returns:
            Dictionary with directional statistics
        """
        mag, ang = self.flow_to_magnitude_angle(flow)
        
        # Filter significant motion
        motion_mask = mag > 0.5
        
        if not np.any(motion_mask):
            return {
                'dominant_direction': 'none',
                'horizontal_flow': 0.0,
                'vertical_flow': 0.0,
                'direction_variance': 0.0
            }
        
        angles = ang[motion_mask]
        
        # Calculate direction statistics
        # 0°/360° = Right, 90° = Down, 180° = Left, 270° = Up
        horizontal_flow = np.mean(np.cos(np.radians(angles)))  # -1 to 1
        vertical_flow = np.mean(np.sin(np.radians(angles)))    # -1 to 1
        
        # Determine dominant direction
        if abs(horizontal_flow) > abs(vertical_flow):
            dominant = 'right' if horizontal_flow > 0 else 'left'
        else:
            dominant = 'down' if vertical_flow > 0 else 'up'
        
        return {
            'dominant_direction': dominant,
            'horizontal_flow': float(horizontal_flow),
            'vertical_flow': float(vertical_flow),
            'direction_variance': float(np.std(angles))
        }
    
    def process_video(self, video_path: str, max_frames: Optional[int] = None,
                     sample_rate: int = 1, fps_override: Optional[float] = None) -> Dict:
        """
        Process entire video and extract flow-based features
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            sample_rate: Process every Nth frame
            fps_override: Override video FPS
            
        Returns:
            Dictionary with aggregated flow statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")
        
        fps = fps_override if fps_override else cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"📹 Optical Flow Analizi: {Path(video_path).name}")
        print(f"   FPS: {fps:.1f}, Toplam Frame: {total_frames}, Örnekleme: 1/{sample_rate}")
        
        # Statistics collectors
        speed_stats = []
        direction_stats = []
        flow_magnitudes = []
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("İlk frame okunamadı")
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        frame_idx = 1
        processed_pairs = 0
        
        while frame_idx < total_frames:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % sample_rate == 0:
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow
                flow = self.calculate_flow(prev_gray, curr_gray)
                
                # Estimate speed
                speed_info = self.estimate_speed(flow, fps)
                speed_stats.append(speed_info)
                
                # Detect direction
                dir_info = self.detect_traffic_direction(flow)
                direction_stats.append(dir_info)
                
                # Store magnitude statistics
                mag, _ = self.flow_to_magnitude_angle(flow)
                flow_magnitudes.append(np.mean(mag))
                
                prev_gray = curr_gray
                processed_pairs += 1
            
            frame_idx += 1
        
        cap.release()
        
        # Aggregate features
        if not speed_stats:
            return self._empty_features()
        
        features = {
            # Speed features
            'avg_speed_kmh_mean': np.mean([s['avg_speed_kmh'] for s in speed_stats]),
            'avg_speed_kmh_std': np.std([s['avg_speed_kmh'] for s in speed_stats]),
            'max_speed_kmh_mean': np.mean([s['max_speed_kmh'] for s in speed_stats]),
            'max_speed_kmh_max': np.max([s['max_speed_kmh'] for s in speed_stats]),
            
            # Motion features
            'motion_percentage_mean': np.mean([s['motion_percentage'] for s in speed_stats]),
            'motion_percentage_std': np.std([s['motion_percentage'] for s in speed_stats]),
            
            # Flow magnitude features
            'flow_magnitude_mean': np.mean(flow_magnitudes),
            'flow_magnitude_std': np.std(flow_magnitudes),
            'flow_magnitude_max': np.max(flow_magnitudes),
            
            # Direction features
            'horizontal_flow_mean': np.mean([d['horizontal_flow'] for d in direction_stats]),
            'vertical_flow_mean': np.mean([d['vertical_flow'] for d in direction_stats]),
            'direction_variance_mean': np.mean([d['direction_variance'] for d in direction_stats]),
            
            # Processing info
            'frames_processed': processed_pairs
        }
        
        print(f"✓ İşlendi: {processed_pairs} frame çifti")
        print(f"   Ortalama Hız: {features['avg_speed_kmh_mean']:.1f} km/h")
        print(f"   Hareket Yüzdesi: {features['motion_percentage_mean']:.1f}%")
        
        return features
    
    def _empty_features(self) -> Dict:
        """Return empty features when no flow detected"""
        return {
            'avg_speed_kmh_mean': 0.0,
            'avg_speed_kmh_std': 0.0,
            'max_speed_kmh_mean': 0.0,
            'max_speed_kmh_max': 0.0,
            'motion_percentage_mean': 0.0,
            'motion_percentage_std': 0.0,
            'flow_magnitude_mean': 0.0,
            'flow_magnitude_std': 0.0,
            'flow_magnitude_max': 0.0,
            'horizontal_flow_mean': 0.0,
            'vertical_flow_mean': 0.0,
            'direction_variance_mean': 0.0,
            'frames_processed': 0
        }
    
    def visualize_flow(self, frame: np.ndarray, flow: np.ndarray,
                      step: int = 16) -> np.ndarray:
        """
        Visualize optical flow with arrows
        
        Args:
            frame: Original frame (BGR)
            flow: Flow field
            step: Arrow sampling step
            
        Returns:
            Annotated frame
        """
        vis = frame.copy()
        h, w = flow.shape[:2]
        
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        
        # Create line endpoints
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        
        # Draw arrows
        for (x1, y1), (x2, y2) in lines:
            mag = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if mag > 1:  # Only draw significant motion
                color = (0, 255, 0) if mag < 5 else (0, 165, 255)
                cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 1,
                              tipLength=0.3)
        
        return vis
    
    def visualize_flow_hsv(self, flow: np.ndarray) -> np.ndarray:
        """
        Visualize optical flow using HSV color space
        Hue = direction, Saturation = magnitude
        
        Args:
            flow: Flow field
            
        Returns:
            HSV visualization (BGR format)
        """
        mag, ang = self.flow_to_magnitude_angle(flow)
        
        # Create HSV image
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang / 2  # Hue: direction
        hsv[..., 1] = 255  # Full saturation
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude
        
        # Convert to BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return bgr


def demo_optical_flow():
    """Demo function for optical flow analyzer"""
    print("\n" + "="*60)
    print("🎯 OPTICAL FLOW ANALYZER DEMO")
    print("="*60)
    
    # Check for video files
    video_dir = Path('videos/normanniles1')
    
    if not video_dir.exists() or not list(video_dir.glob('*.mp4')):
        print(f"\n⚠️  Video dosyası bulunamadı: {video_dir}")
        print("Demo sentetik örnek ile devam ediyor...\n")
        
        print("📋 Optical Flow Parametreleri:")
        print("   - pyr_scale: 0.5 (Pyramid ölçeği)")
        print("   - levels: 3 (Pyramid katmanları)")
        print("   - winsize: 15 (Pencere boyutu)")
        print("   - iterations: 3 (İterasyon sayısı)")
        print("   - poly_n: 5 (Komşuluk boyutu)")
        print("   - poly_sigma: 1.2 (Gaussian std)")
        
        print("\n🎯 Çıkarılan Özellikler:")
        print("   • Ortalama/Max hız (km/h)")
        print("   • Hareket yüzdesi")
        print("   • Flow büyüklüğü istatistikleri")
        print("   • Trafik yönü (horizontal/vertical)")
        print("   • Yön varyansı")
        
        print("\n💡 Kullanım:")
        print("   from traffic_optical_flow import OpticalFlowAnalyzer")
        print("   analyzer = OpticalFlowAnalyzer()")
        print("   features = analyzer.process_video('video.mp4')")
        
        return
    
    # Find video files
    video_files = list(video_dir.glob('*.mp4'))
    video_path = str(video_files[0])
    
    print(f"\n📹 Test Video: {Path(video_path).name}")
    
    # Initialize analyzer
    analyzer = OpticalFlowAnalyzer()
    
    # Process video
    print("\n🔍 Video işleniyor (ilk 300 frame)...")
    features = analyzer.process_video(video_path, max_frames=300, sample_rate=2)
    
    # Display results
    print("\n📊 Optical Flow Özellikleri:")
    print("="*60)
    for key, value in features.items():
        if isinstance(value, float):
            print(f"   {key:30s}: {value:.3f}")
        else:
            print(f"   {key:30s}: {value}")
    
    print("\n" + "="*60)
    print("✅ OPTICAL FLOW DEMO TAMAMLANDI!")
    print("="*60)


if __name__ == "__main__":
    demo_optical_flow()
