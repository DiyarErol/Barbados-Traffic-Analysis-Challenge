"""
Video Feature Extraction
========================
Extract traffic features from video files.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from .base import BaseFeatureExtractor
from ..config.feature_config import VideoFeatureConfig


class VideoFeatureExtractor(BaseFeatureExtractor):
    """Extract features from video files."""
    
    def __init__(self, config: VideoFeatureConfig = None):
        """
        Initialize video feature extractor.
        
        Args:
            config: Video feature configuration
        """
        super().__init__(config or VideoFeatureConfig())
        self.bg_subtractor = None
        self.feature_names = [
            'vehicle_count',
            'density_score',
            'movement_score',
            'avg_contour_area',
            'motion_intensity',
            'frame_difference',
            'foreground_ratio',
            'active_regions'
        ]
    
    def extract(self, video_path: Path) -> pd.DataFrame:
        """
        Extract features from a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            DataFrame with video features
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        features = self._process_video(cap)
        cap.release()
        
        return pd.DataFrame([features], columns=self.feature_names)
    
    def _process_video(self, cap: cv2.VideoCapture) -> Dict[str, float]:
        """
        Process video and extract features.
        
        Args:
            cap: OpenCV video capture object
            
        Returns:
            Dictionary of features
        """
        # Initialize background subtractor
        if self.config.use_background_subtraction:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.config.bg_history,
                varThreshold=self.config.bg_var_threshold,
                detectShadows=self.config.bg_detect_shadows
            )
        
        frame_count = 0
        total_vehicle_count = 0
        total_density = 0
        total_movement = 0
        total_contour_area = 0
        total_motion = 0
        total_foreground = 0
        total_active_regions = 0
        prev_gray = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame for efficiency
            if frame_count % self.config.target_fps != 0:
                frame_count += 1
                continue
            
            # Resize frame
            frame = cv2.resize(frame, self.config.resize_dims)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Vehicle counting via background subtraction
            if self.bg_subtractor is not None:
                vehicle_count, contour_area = self._count_vehicles(frame)
                total_vehicle_count += vehicle_count
                total_contour_area += contour_area
            
            # Density calculation
            density = self._calculate_density(gray)
            total_density += density
            
            # Movement detection
            if prev_gray is not None:
                movement = self._detect_movement(gray, prev_gray)
                total_movement += movement
                
                motion = self._calculate_motion_intensity(gray, prev_gray)
                total_motion += motion
            
            # Foreground analysis
            if self.bg_subtractor is not None:
                fg_mask = self.bg_subtractor.apply(frame)
                foreground_ratio = np.sum(fg_mask > 0) / fg_mask.size
                total_foreground += foreground_ratio
                
                active_regions = self._count_active_regions(fg_mask)
                total_active_regions += active_regions
            
            prev_gray = gray
            frame_count += 1
        
        # Calculate averages
        if frame_count == 0:
            return {name: 0.0 for name in self.feature_names}
        
        processed_frames = frame_count // self.config.target_fps
        
        return {
            'vehicle_count': total_vehicle_count / max(processed_frames, 1),
            'density_score': total_density / max(processed_frames, 1),
            'movement_score': total_movement / max(processed_frames, 1),
            'avg_contour_area': total_contour_area / max(total_vehicle_count, 1),
            'motion_intensity': total_motion / max(processed_frames, 1),
            'frame_difference': total_movement / max(processed_frames, 1),
            'foreground_ratio': total_foreground / max(processed_frames, 1),
            'active_regions': total_active_regions / max(processed_frames, 1)
        }
    
    def _count_vehicles(self, frame: np.ndarray) -> Tuple[int, float]:
        """Count vehicles using background subtraction."""
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        vehicle_count = 0
        total_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config.min_contour_area < area < self.config.max_contour_area:
                vehicle_count += 1
                total_area += area
        
        return vehicle_count, total_area
    
    def _calculate_density(self, gray: np.ndarray) -> float:
        """Calculate traffic density using grid analysis."""
        h, w = gray.shape
        grid_h, grid_w = self.config.grid_size
        
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        dense_cells = 0
        
        for i in range(grid_h):
            for j in range(grid_w):
                cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                std = np.std(cell)
                
                if std > self.config.density_threshold * 255:
                    dense_cells += 1
        
        return dense_cells / (grid_h * grid_w)
    
    def _detect_movement(self, current: np.ndarray, previous: np.ndarray) -> float:
        """Detect movement between frames."""
        diff = cv2.absdiff(current, previous)
        _, thresh = cv2.threshold(diff, self.config.motion_threshold, 255, cv2.THRESH_BINARY)
        
        movement_ratio = np.sum(thresh > 0) / thresh.size
        return movement_ratio
    
    def _calculate_motion_intensity(self, current: np.ndarray, previous: np.ndarray) -> float:
        """Calculate overall motion intensity."""
        diff = cv2.absdiff(current, previous)
        motion = np.mean(diff) / 255.0
        return motion
    
    def _count_active_regions(self, fg_mask: np.ndarray) -> int:
        """Count number of active regions in foreground mask."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated = cv2.dilate(fg_mask, kernel)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return len([c for c in contours if cv2.contourArea(c) > self.config.min_contour_area])
    
    def extract_batch(self, video_paths: list, show_progress: bool = True) -> pd.DataFrame:
        """
        Extract features from multiple videos.
        
        Args:
            video_paths: List of video file paths
            show_progress: Show progress bar
            
        Returns:
            DataFrame with features for all videos
        """
        all_features = []
        
        iterator = tqdm(video_paths) if show_progress else video_paths
        
        for video_path in iterator:
            try:
                features = self.extract(video_path)
                features['video_path'] = str(video_path)
                all_features.append(features)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                # Add empty features
                empty_features = pd.DataFrame([{name: 0.0 for name in self.feature_names}])
                empty_features['video_path'] = str(video_path)
                all_features.append(empty_features)
        
        return pd.concat(all_features, ignore_index=True)
