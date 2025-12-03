"""
Model Performance Monitoring System
Tracks model drift, accuracy degradation, and performance metrics over time
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
import os
import glob
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelMonitor:
    """
    Monitor model performance and detect drift
    """
    
    def __init__(self, model_name: str = 'traffic_model', 
                 log_dir: str = 'monitoring_logs'):
        """
        Args:
            model_name: Name identifier for the model
            log_dir: Directory to store monitoring logs
        """
        self.model_name = model_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.performance_log = []
        self.drift_alerts = []
        
        # Load existing logs
        self._load_logs()
    
    def _load_logs(self):
        """Load existing monitoring logs"""
        log_file = self.log_dir / f'{self.model_name}_performance.json'
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                data = json.load(f)
                self.performance_log = data.get('performance_log', [])
                self.drift_alerts = data.get('drift_alerts', [])
    
    def _save_logs(self):
        """Save monitoring logs"""
        log_file = self.log_dir / f'{self.model_name}_performance.json'
        
        data = {
            'model_name': self.model_name,
            'last_updated': datetime.now().isoformat(),
            'performance_log': self.performance_log,
            'drift_alerts': self.drift_alerts
        }
        
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def log_prediction_batch(self, y_true: np.ndarray, y_pred: np.ndarray,
                            y_proba: Optional[np.ndarray] = None,
                            batch_metadata: Optional[Dict] = None):
        """
        Log a batch of predictions for monitoring
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            batch_metadata: Additional metadata (timestamp, data source, etc.)
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Calculate confidence if probabilities available
        avg_confidence = None
        if y_proba is not None:
            predicted_probs = np.max(y_proba, axis=1)
            avg_confidence = float(np.mean(predicted_probs))
        
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y_true),
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'avg_confidence': avg_confidence
        }
        
        # Add metadata
        if batch_metadata:
            log_entry.update(batch_metadata)
        
        # Add class-wise metrics
        cm = confusion_matrix(y_true, y_pred)
        log_entry['confusion_matrix'] = cm.tolist()
        
        self.performance_log.append(log_entry)
        self._save_logs()
        
        # Check for drift
        self._check_performance_drift(log_entry)
    
    def _check_performance_drift(self, current_entry: Dict):
        """Check if performance has degraded significantly"""
        if len(self.performance_log) < 10:
            return  # Need baseline
        
        # Get baseline performance (first 10 entries or recent stable period)
        baseline_entries = self.performance_log[:10]
        baseline_acc = np.mean([e['accuracy'] for e in baseline_entries])
        baseline_f1 = np.mean([e['f1_score'] for e in baseline_entries])
        
        # Current performance
        current_acc = current_entry['accuracy']
        current_f1 = current_entry['f1_score']
        
        # Drift thresholds
        acc_threshold = 0.05  # 5% drop
        f1_threshold = 0.05
        
        # Check for drift
        acc_drift = baseline_acc - current_acc
        f1_drift = baseline_f1 - current_f1
        
        if acc_drift > acc_threshold or f1_drift > f1_threshold:
            drift_alert = {
                'timestamp': current_entry['timestamp'],
                'type': 'performance_degradation',
                'baseline_accuracy': float(baseline_acc),
                'current_accuracy': float(current_acc),
                'accuracy_drop': float(acc_drift),
                'baseline_f1': float(baseline_f1),
                'current_f1': float(current_f1),
                'f1_drop': float(f1_drift),
                'severity': 'high' if acc_drift > 0.1 else 'medium'
            }
            
            self.drift_alerts.append(drift_alert)
            self._save_logs()
            
            # Print alert
            print(f"\n⚠️  PERFORMANCE DRIFT DETECTED")
            print(f"   Accuracy: {baseline_acc:.4f} → {current_acc:.4f} (Δ {acc_drift:.4f})")
            print(f"   F1 Score: {baseline_f1:.4f} → {current_f1:.4f} (Δ {f1_drift:.4f})")
    
    def detect_data_drift(self, X_baseline: pd.DataFrame, 
                         X_current: pd.DataFrame,
                         threshold: float = 0.1) -> Dict:
        """
        Detect feature distribution drift using statistical tests
        
        Args:
            X_baseline: Baseline feature data
            X_current: Current feature data
            threshold: Drift detection threshold
            
        Returns:
            Dictionary with drift analysis
        """
        drifted_features = []
        
        for col in X_baseline.columns:
            # Calculate distribution statistics
            baseline_mean = X_baseline[col].mean()
            baseline_std = X_baseline[col].std()
            
            current_mean = X_current[col].mean()
            current_std = X_current[col].std()
            
            # Detect drift (simple z-score based)
            if baseline_std > 0:
                z_score = abs(current_mean - baseline_mean) / baseline_std
                
                if z_score > 3:  # 3 sigma
                    drifted_features.append({
                        'feature': col,
                        'baseline_mean': float(baseline_mean),
                        'current_mean': float(current_mean),
                        'baseline_std': float(baseline_std),
                        'current_std': float(current_std),
                        'z_score': float(z_score)
                    })
        
        drift_detected = len(drifted_features) > 0
        
        return {
            'drift_detected': drift_detected,
            'n_drifted_features': len(drifted_features),
            'drifted_features': drifted_features,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_performance_trends(self, window_hours: int = 24) -> Dict:
        """
        Get performance trends over time window
        
        Args:
            window_hours: Time window in hours
            
        Returns:
            Performance trends dictionary
        """
        if not self.performance_log:
            return {'error': 'No performance data available'}
        
        cutoff = datetime.now() - timedelta(hours=window_hours)
        
        recent_logs = [
            log for log in self.performance_log
            if datetime.fromisoformat(log['timestamp']) > cutoff
        ]
        
        if not recent_logs:
            return {'error': 'No data in time window'}
        
        # Calculate trends
        accuracies = [log['accuracy'] for log in recent_logs]
        f1_scores = [log['f1_score'] for log in recent_logs]
        
        return {
            'time_window_hours': window_hours,
            'n_batches': len(recent_logs),
            'accuracy': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies)),
                'trend': 'improving' if accuracies[-1] > accuracies[0] else 'degrading'
            },
            'f1_score': {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'min': float(np.min(f1_scores)),
                'max': float(np.max(f1_scores)),
                'trend': 'improving' if f1_scores[-1] > f1_scores[0] else 'degrading'
            }
        }
    
    def plot_performance_history(self, save_path: Optional[str] = None):
        """Plot performance metrics over time"""
        if not self.performance_log:
            print("⚠️  No performance data to plot")
            return
        
        # Extract data
        timestamps = [datetime.fromisoformat(log['timestamp']) 
                     for log in self.performance_log]
        accuracies = [log['accuracy'] for log in self.performance_log]
        f1_scores = [log['f1_score'] for log in self.performance_log]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Accuracy plot
        ax1.plot(timestamps, accuracies, marker='o', linewidth=2, 
                label='Accuracy', color='#1f77b4')
        ax1.axhline(y=np.mean(accuracies), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(accuracies):.4f}')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Performance Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F1 Score plot
        ax2.plot(timestamps, f1_scores, marker='s', linewidth=2, 
                label='F1 Score', color='#ff7f0e')
        ax2.axhline(y=np.mean(f1_scores), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(f1_scores):.4f}')
        ax2.set_xlabel('Timestamp', fontsize=12)
        ax2.set_ylabel('F1 Score', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved: {save_path}")
        else:
            plt.savefig(self.log_dir / f'{self.model_name}_performance_history.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def generate_monitoring_report(self) -> str:
        """Generate comprehensive monitoring report"""
        report = []
        report.append("="*60)
        report.append(f"MODEL MONITORING REPORT: {self.model_name}")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Batches Logged: {len(self.performance_log)}")
        report.append("")
        
        # Performance trends
        trends = self.get_performance_trends(window_hours=24)
        if 'error' not in trends:
            report.append("📊 Performance Trends (Last 24 Hours)")
            report.append("-" * 60)
            report.append(f"Batches: {trends['n_batches']}")
            report.append(f"Accuracy: {trends['accuracy']['mean']:.4f} "
                        f"(±{trends['accuracy']['std']:.4f}) - {trends['accuracy']['trend']}")
            report.append(f"F1 Score: {trends['f1_score']['mean']:.4f} "
                        f"(±{trends['f1_score']['std']:.4f}) - {trends['f1_score']['trend']}")
            report.append("")
        
        # Drift alerts
        if self.drift_alerts:
            report.append("⚠️  Drift Alerts")
            report.append("-" * 60)
            for alert in self.drift_alerts[-5:]:  # Last 5 alerts
                report.append(f"Timestamp: {alert['timestamp']}")
                report.append(f"Type: {alert['type']}")
                report.append(f"Severity: {alert['severity']}")
                report.append(f"Accuracy Drop: {alert.get('accuracy_drop', 0):.4f}")
                report.append("")
        else:
            report.append("✅ No Drift Alerts")
            report.append("")
        
        report.append("="*60)
        
        return "\n".join(report)

    # ===== Submission-level monitoring (best vs recent candidates) =====
    @staticmethod
    def _read_submission(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        colmap = {c.lower(): c for c in df.columns}
        if 'id' not in colmap or 'target' not in colmap:
            raise ValueError(f"Submission missing ID/Target columns: {path}")
        return df.rename(columns={colmap['id']: 'ID', colmap['target']: 'Target'})[['ID','Target']]

    @staticmethod
    def _dist(df: pd.DataFrame) -> dict:
        vc = df['Target'].value_counts(normalize=True) * 100
        return {
            'free': float(vc.get('free flowing', 0.0)),
            'light': float(vc.get('light delay', 0.0)),
            'moderate': float(vc.get('moderate delay', 0.0)),
            'heavy': float(vc.get('heavy delay', 0.0)),
        }

    def monitor_recent_submissions(self, root: str = '.', config_path: str = 'config/best_submission.json',
                                   top_k: int = 5, reports_dir: str = 'reports') -> str:
        """
        Compare best submission to most recent candidate submissions and write a summary report.
        """
        os.makedirs(reports_dir, exist_ok=True)

        # Load best
        best_name = None
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    best_name = json.load(f).get('best_submission')
            except Exception:
                pass
        if not best_name:
            raise FileNotFoundError('Best submission not defined in config/best_submission.json')

        # Locate best path
        candidates_best = [os.path.join(root, best_name), os.path.join(root, 'submissions', best_name)]
        best_path = None
        for p in candidates_best:
            if os.path.exists(p):
                best_path = p
                break
        if best_path is None:
            raise FileNotFoundError(f'Best submission not found: {best_name}')

        # List candidate submissions (root + submissions/)
        sub_files = []
        sub_files += glob.glob(os.path.join(root, 'submission*.csv'))
        sub_files += glob.glob(os.path.join(root, 'submissions', '**', 'submission*.csv'), recursive=True)
        # Exclude the best itself
        sub_files = [f for f in sub_files if os.path.abspath(f) != os.path.abspath(best_path)]
        # Sort by mtime desc
        sub_files = sorted(sub_files, key=lambda p: os.path.getmtime(p), reverse=True)[:top_k]
        if not sub_files:
            return 'No candidate submissions found.'

        best_df = self._read_submission(best_path)
        best_dist = self._dist(best_df)

        lines = []
        lines.append(f"# Submission Monitoring Report\n\nGenerated: {datetime.now().isoformat()}\n")
        lines.append(f"- Best: {best_name} ({os.path.relpath(best_path)})\n")
        lines.append(f"- Candidates analyzed: {len(sub_files)}\n")
        lines.append("## Best Distribution\n")
        lines.append(str(best_dist) + "\n")

        for path in sub_files:
            try:
                cand_df = self._read_submission(path)
            except Exception as e:
                lines.append(f"- {os.path.basename(path)}: read_error: {e}\n")
                continue
            merged = best_df.merge(cand_df, on='ID', how='inner', suffixes=('_best','_cand'))
            agree = float((merged['Target_best'] == merged['Target_cand']).mean() * 100.0)
            cand_dist = self._dist(cand_df)
            lines.append(f"### {os.path.basename(path)}\n")
            lines.append(f"- Path: {os.path.relpath(path)}\n- Overlap: {len(merged)} rows\n- Agreement: {agree:.2f}%\n")
            lines.append(f"- Candidate Dist: {cand_dist}\n")

        out_path = os.path.join(reports_dir, f"submission_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"✓ Monitoring report written: {out_path}")
        return out_path


def demo_monitoring():
    """Demo function for model monitoring"""
    print("\n" + "="*60)
    print("🎯 MODEL MONITORING DEMO")
    print("="*60)
    
    # Initialize monitor
    monitor = ModelMonitor(model_name='traffic_congestion_model')
    
    print("\n📊 Simulating Model Monitoring...")
    print("="*60)
    
    # Simulate multiple prediction batches over time
    np.random.seed(42)
    
    for i in range(15):
        # Generate synthetic predictions
        n_samples = np.random.randint(50, 200)
        
        # Simulate degrading performance over time
        base_accuracy = 0.80
        degradation = i * 0.01  # 1% per batch
        
        # Generate predictions with noise
        if np.random.rand() < base_accuracy - degradation:
            y_true = np.random.randint(0, 4, n_samples)
            y_pred = y_true.copy()
            # Add some errors
            error_indices = np.random.choice(n_samples, size=int(n_samples * (1 - base_accuracy + degradation)), replace=False)
            y_pred[error_indices] = np.random.randint(0, 4, len(error_indices))
        else:
            y_true = np.random.randint(0, 4, n_samples)
            y_pred = np.random.randint(0, 4, n_samples)
        
        # Generate prediction probabilities
        y_proba = np.random.dirichlet(np.ones(4), size=n_samples)
        
        # Log batch
        metadata = {
            'batch_id': i + 1,
            'data_source': 'simulation',
            'timestamp': (datetime.now() - timedelta(hours=15-i)).isoformat()
        }
        
        monitor.log_prediction_batch(y_true, y_pred, y_proba, metadata)
        
        print(f"✓ Batch {i+1}/15 logged")
    
    # Generate performance trends
    print("\n📈 Performance Trends:")
    print("="*60)
    trends = monitor.get_performance_trends(window_hours=24)
    print(json.dumps(trends, indent=2))
    
    # Simulate data drift detection
    print("\n🔍 Data Drift Detection:")
    print("="*60)
    
    # Generate baseline and current feature distributions
    X_baseline = pd.DataFrame({
        'vehicle_count': np.random.normal(20, 5, 1000),
        'avg_speed': np.random.normal(35, 10, 1000),
        'traffic_density': np.random.normal(0.5, 0.15, 1000)
    })
    
    # Current data with drift in vehicle_count
    X_current = pd.DataFrame({
        'vehicle_count': np.random.normal(30, 8, 1000),  # Drifted
        'avg_speed': np.random.normal(35, 10, 1000),
        'traffic_density': np.random.normal(0.5, 0.15, 1000)
    })
    
    drift_result = monitor.detect_data_drift(X_baseline, X_current)
    print(json.dumps(drift_result, indent=2))
    
    # Plot performance history
    print("\n📊 Generating Performance Plot...")
    monitor.plot_performance_history()
    print(f"✓ Plot saved: {monitor.log_dir / f'{monitor.model_name}_performance_history.png'}")
    
    # Generate report
    print("\n📋 Monitoring Report:")
    print(monitor.generate_monitoring_report())
    
    print("\n" + "="*60)
    print("✅ MONITORING DEMO COMPLETED!")
    print("="*60)
    
    print("\n💡 Usage:")
    print("  from model_monitoring import ModelMonitor")
    print("  monitor = ModelMonitor('my_model')")
    print("  monitor.log_prediction_batch(y_true, y_pred, y_proba)")
    print("  trends = monitor.get_performance_trends(window_hours=24)")
    print("  monitor.plot_performance_history()")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--submissions-monitor', action='store_true', help='Run submission monitoring against best')
    parser.add_argument('--top-k', type=int, default=5, help='How many recent submissions to compare')
    args = parser.parse_args()

    if args.submissions_monitor:
        monitor = ModelMonitor(model_name='traffic_congestion_model')
        monitor.monitor_recent_submissions(root='.', top_k=args.top_k)
    else:
        demo_monitoring()
