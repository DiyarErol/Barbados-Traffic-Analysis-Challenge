"""
Traffic Alert System
Real-time congestion monitoring with threshold-based alerts and notifications
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class AlertSystem:
    """
    Threshold-based alert system for traffic congestion monitoring
    """
    
    # Congestion level mappings
    CONGESTION_LEVELS = {
        0: 'free flowing',
        1: 'light delay',
        2: 'moderate delay',
        3: 'heavy delay'
    }
    
    # Alert severity levels
    SEVERITY_LEVELS = {
        'low': 1,
        'medium': 2,
        'high': 3,
        'critical': 4
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to alert configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.alert_history = []
        self.notification_callbacks = []
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load alert configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'thresholds': {
                'moderate_delay': {
                    'min_duration': 5,  # minutes
                    'severity': 'medium'
                },
                'heavy_delay': {
                    'min_duration': 3,  # minutes
                    'severity': 'high'
                },
                'continuous_congestion': {
                    'min_duration': 15,  # minutes
                    'severity': 'critical'
                }
            },
            'notification': {
                'enabled': True,
                'min_interval': 10,  # minutes between same alerts
                'channels': ['console', 'log']
            },
            'rush_hour': {
                'enabled': True,
                'hours': [7, 8, 9, 16, 17, 18],
                'severity_multiplier': 1.5
            }
        }
    
    def add_notification_callback(self, callback: Callable):
        """
        Add custom notification callback
        
        Args:
            callback: Function(alert_dict) to call when alert triggered
        """
        self.notification_callbacks.append(callback)
    
    def check_congestion_threshold(self, congestion_level: int,
                                  location: str = 'enter',
                                  timestamp: datetime = None) -> Optional[Dict]:
        """
        Check if congestion exceeds thresholds
        
        Args:
            congestion_level: 0-3 (free flowing to heavy delay)
            location: 'enter' or 'exit'
            timestamp: Current timestamp
            
        Returns:
            Alert dictionary if threshold exceeded, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        congestion_name = self.CONGESTION_LEVELS.get(congestion_level, 'unknown')
        
        # Check thresholds
        alert = None
        
        if congestion_level >= 2:  # Moderate or heavy delay
            severity = 'medium' if congestion_level == 2 else 'high'
            
            # Check if rush hour (increase severity)
            if self.config['rush_hour']['enabled']:
                if timestamp.hour in self.config['rush_hour']['hours']:
                    severity = 'critical' if severity == 'high' else 'high'
            
            alert = {
                'timestamp': timestamp,
                'location': location,
                'congestion_level': congestion_level,
                'congestion_name': congestion_name,
                'severity': severity,
                'message': f"Traffic congestion detected at {location}: {congestion_name}",
                'is_rush_hour': timestamp.hour in self.config['rush_hour']['hours']
            }
        
        return alert
    
    def check_duration_threshold(self, congestion_history: List[Dict]) -> Optional[Dict]:
        """
        Check if congestion persists beyond duration threshold
        
        Args:
            congestion_history: List of {timestamp, congestion_level} dicts
            
        Returns:
            Alert dictionary if duration threshold exceeded
        """
        if len(congestion_history) < 2:
            return None
        
        # Sort by timestamp
        history = sorted(congestion_history, key=lambda x: x['timestamp'])
        
        # Find continuous congestion periods
        current_period_start = history[0]['timestamp']
        current_level = history[0]['congestion_level']
        max_duration = 0
        
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            # Check if congestion continues
            if curr['congestion_level'] >= 2:  # Moderate or heavy
                duration = (curr['timestamp'] - current_period_start).total_seconds() / 60
                max_duration = max(max_duration, duration)
            else:
                # Reset period
                current_period_start = curr['timestamp']
        
        # Check against thresholds
        thresholds = self.config['thresholds']
        
        if max_duration >= thresholds['continuous_congestion']['min_duration']:
            return {
                'timestamp': datetime.now(),
                'type': 'duration',
                'duration_minutes': max_duration,
                'severity': 'critical',
                'message': f"Continuous congestion for {max_duration:.1f} minutes"
            }
        
        return None
    
    def check_prediction_confidence(self, prediction: int, probabilities: np.ndarray,
                                   min_confidence: float = 0.7) -> Optional[Dict]:
        """
        Alert if prediction has low confidence
        
        Args:
            prediction: Predicted class
            probabilities: Class probabilities
            min_confidence: Minimum confidence threshold
            
        Returns:
            Alert dictionary if confidence too low
        """
        confidence = probabilities[prediction]
        
        if confidence < min_confidence:
            return {
                'timestamp': datetime.now(),
                'type': 'low_confidence',
                'prediction': prediction,
                'confidence': float(confidence),
                'severity': 'low',
                'message': f"Low prediction confidence: {confidence:.2%}"
            }
        
        return None
    
    def trigger_alert(self, alert: Dict) -> bool:
        """
        Trigger an alert and send notifications
        
        Args:
            alert: Alert dictionary
            
        Returns:
            True if alert was sent
        """
        # Check if similar alert was sent recently
        if self._is_duplicate_alert(alert):
            return False
        
        # Add to history
        self.alert_history.append(alert)
        
        # Send notifications
        if self.config['notification']['enabled']:
            self._send_notifications(alert)
        
        return True
    
    def _is_duplicate_alert(self, alert: Dict) -> bool:
        """Check if similar alert was sent recently"""
        min_interval = self.config['notification']['min_interval']
        
        for prev_alert in reversed(self.alert_history[-10:]):  # Check last 10
            time_diff = (alert['timestamp'] - prev_alert['timestamp']).total_seconds() / 60
            
            if time_diff < min_interval:
                # Check if same type
                if alert.get('location') == prev_alert.get('location'):
                    if alert.get('severity') == prev_alert.get('severity'):
                        return True
        
        return False
    
    def _send_notifications(self, alert: Dict):
        """Send alert notifications"""
        channels = self.config['notification']['channels']
        
        # Console notification
        if 'console' in channels:
            self._console_notification(alert)
        
        # Log notification
        if 'log' in channels:
            self._log_notification(alert)
        
        # Custom callbacks
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"⚠️  Callback error: {e}")
    
    def _console_notification(self, alert: Dict):
        """Print alert to console"""
        severity = alert.get('severity', 'low')
        
        # Color codes
        colors = {
            'low': '\033[92m',      # Green
            'medium': '\033[93m',   # Yellow
            'high': '\033[91m',     # Red
            'critical': '\033[95m'  # Magenta
        }
        reset = '\033[0m'
        
        color = colors.get(severity, '')
        
        print(f"\n{color}{'='*60}")
        print(f"⚠️  TRAFFIC ALERT - {severity.upper()}")
        print(f"{'='*60}{reset}")
        print(f"Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Message: {alert.get('message', 'Alert triggered')}")
        
        if 'location' in alert:
            print(f"Location: {alert['location']}")
        if 'congestion_name' in alert:
            print(f"Congestion: {alert['congestion_name']}")
        if 'duration_minutes' in alert:
            print(f"Duration: {alert['duration_minutes']:.1f} minutes")
        if 'confidence' in alert:
            print(f"Confidence: {alert['confidence']:.2%}")
        
        print(f"{color}{'='*60}{reset}\n")
    
    def _log_notification(self, alert: Dict):
        """Write alert to log file"""
        log_file = Path('traffic_alerts.log')
        
        with open(log_file, 'a') as f:
            log_entry = {
                'timestamp': alert['timestamp'].isoformat(),
                'severity': alert.get('severity'),
                'message': alert.get('message'),
                'details': {k: v for k, v in alert.items() 
                          if k not in ['timestamp', 'severity', 'message']}
            }
            f.write(json.dumps(log_entry) + '\n')
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """
        Get summary of alerts in last N hours
        
        Args:
            hours: Time window in hours
            
        Returns:
            Summary statistics
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [a for a in self.alert_history 
                        if a['timestamp'] > cutoff]
        
        if not recent_alerts:
            return {
                'total_alerts': 0,
                'by_severity': {},
                'by_location': {}
            }
        
        # Count by severity
        by_severity = {}
        for alert in recent_alerts:
            severity = alert.get('severity', 'unknown')
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Count by location
        by_location = {}
        for alert in recent_alerts:
            location = alert.get('location', 'unknown')
            by_location[location] = by_location.get(location, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'by_severity': by_severity,
            'by_location': by_location,
            'time_window_hours': hours
        }
    
    def clear_old_alerts(self, hours: int = 24):
        """Remove alerts older than specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        self.alert_history = [a for a in self.alert_history 
                             if a['timestamp'] > cutoff]


def demo_alert_system():
    """Demo function for alert system"""
    print("\n" + "="*60)
    print("🎯 TRAFFIC ALERT SYSTEM DEMO")
    print("="*60)
    
    # Initialize alert system
    alert_system = AlertSystem()
    
    # Custom notification callback
    def email_notification(alert):
        print(f"📧 [EMAIL] Alert sent: {alert['message']}")
    
    def sms_notification(alert):
        if alert['severity'] in ['high', 'critical']:
            print(f"📱 [SMS] Urgent: {alert['message']}")
    
    alert_system.add_notification_callback(email_notification)
    alert_system.add_notification_callback(sms_notification)
    
    # Simulate traffic monitoring
    print("\n🚦 Simulating Traffic Monitoring...")
    print("="*60)
    
    # Scenario 1: Normal traffic
    print("\n1️⃣  Normal Traffic (Free Flowing)")
    alert = alert_system.check_congestion_threshold(0, 'enter', datetime.now())
    if alert:
        alert_system.trigger_alert(alert)
    else:
        print("✅ No alert - Traffic flowing normally")
    
    # Scenario 2: Moderate delay
    print("\n2️⃣  Moderate Delay")
    alert = alert_system.check_congestion_threshold(2, 'enter', datetime.now())
    if alert:
        alert_system.trigger_alert(alert)
    
    # Scenario 3: Heavy delay during rush hour
    print("\n3️⃣  Heavy Delay (Rush Hour)")
    rush_hour_time = datetime.now().replace(hour=17, minute=30)
    alert = alert_system.check_congestion_threshold(3, 'exit', rush_hour_time)
    if alert:
        alert_system.trigger_alert(alert)
    
    # Scenario 4: Low confidence prediction
    print("\n4️⃣  Low Confidence Prediction")
    probabilities = np.array([0.3, 0.4, 0.2, 0.1])
    alert = alert_system.check_prediction_confidence(1, probabilities, min_confidence=0.7)
    if alert:
        alert_system.trigger_alert(alert)
    
    # Scenario 5: Continuous congestion
    print("\n5️⃣  Continuous Congestion")
    congestion_history = [
        {'timestamp': datetime.now() - timedelta(minutes=20), 'congestion_level': 2},
        {'timestamp': datetime.now() - timedelta(minutes=15), 'congestion_level': 3},
        {'timestamp': datetime.now() - timedelta(minutes=10), 'congestion_level': 3},
        {'timestamp': datetime.now() - timedelta(minutes=5), 'congestion_level': 2},
        {'timestamp': datetime.now(), 'congestion_level': 3}
    ]
    alert = alert_system.check_duration_threshold(congestion_history)
    if alert:
        alert_system.trigger_alert(alert)
    
    # Alert summary
    print("\n" + "="*60)
    print("📊 ALERT SUMMARY (Last 24 Hours)")
    print("="*60)
    
    summary = alert_system.get_alert_summary(hours=24)
    print(f"\nTotal Alerts: {summary['total_alerts']}")
    
    if summary['by_severity']:
        print("\nBy Severity:")
        for severity, count in summary['by_severity'].items():
            print(f"   {severity.capitalize()}: {count}")
    
    if summary['by_location']:
        print("\nBy Location:")
        for location, count in summary['by_location'].items():
            print(f"   {location.capitalize()}: {count}")
    
    # Show configuration
    print("\n" + "="*60)
    print("⚙️  ALERT CONFIGURATION")
    print("="*60)
    print(json.dumps(alert_system.config, indent=2))
    
    print("\n" + "="*60)
    print("✅ ALERT SYSTEM DEMO TAMAMLANDI!")
    print("="*60)
    
    print("\n💡 Kullanım:")
    print("  from alert_system import AlertSystem")
    print("  alerts = AlertSystem()")
    print("  alert = alerts.check_congestion_threshold(congestion_level, 'enter')")
    print("  if alert:")
    print("      alerts.trigger_alert(alert)")


if __name__ == "__main__":
    demo_alert_system()
