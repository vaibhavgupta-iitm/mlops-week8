"""
Data Poisoning Module for IRIS Classification Pipeline.
Implements various poisoning attacks and detection mechanisms.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPoisoner:
    """Implements various data poisoning attacks."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize DataPoisoner.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def poison_random_noise(self, 
                           data: pd.DataFrame, 
                           poison_rate: float,
                           noise_level: float = 3.0) -> Tuple[pd.DataFrame, List[int]]:
        """
        Poison data by adding random noise to features.
        
        Args:
            data: Original DataFrame
            poison_rate: Percentage of data to poison (0.0 to 1.0)
            noise_level: Standard deviations of noise to add
            
        Returns:
            Tuple of (poisoned_data, poisoned_indices)
        """
        poisoned_data = data.copy()
        n_samples = len(data)
        n_poison = int(n_samples * poison_rate)
        
        # Randomly select samples to poison
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        for idx in poison_indices:
            for col in feature_cols:
                # Calculate feature statistics
                mean = data[col].mean()
                std = data[col].std()
                
                # Add random noise
                noise = np.random.normal(0, std * noise_level)
                poisoned_data.loc[idx, col] = max(0, data.loc[idx, col] + noise)
        
        logger.info(f"Random noise poisoning: {n_poison}/{n_samples} samples ({poison_rate*100:.1f}%)")
        
        return poisoned_data, poison_indices.tolist()
    
    def poison_label_flip(self, 
                         data: pd.DataFrame, 
                         poison_rate: float) -> Tuple[pd.DataFrame, List[int]]:
        """
        Poison data by randomly flipping labels.
        
        Args:
            data: Original DataFrame
            poison_rate: Percentage of labels to flip
            
        Returns:
            Tuple of (poisoned_data, poisoned_indices)
        """
        poisoned_data = data.copy()
        n_samples = len(data)
        n_poison = int(n_samples * poison_rate)
        
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        species_list = ['setosa', 'versicolor', 'virginica']
        
        for idx in poison_indices:
            current_label = poisoned_data.loc[idx, 'species']
            # Pick a different label
            other_labels = [s for s in species_list if s != current_label]
            poisoned_data.loc[idx, 'species'] = np.random.choice(other_labels)
        
        logger.info(f"Label flip poisoning: {n_poison}/{n_samples} samples ({poison_rate*100:.1f}%)")
        
        return poisoned_data, poison_indices.tolist()
    
    def poison_targeted_attack(self, 
                              data: pd.DataFrame, 
                              poison_rate: float,
                              target_class: str = 'setosa',
                              misclassify_as: str = 'virginica') -> Tuple[pd.DataFrame, List[int]]:
        """
        Targeted poisoning: modify features to cause misclassification.
        
        Args:
            data: Original DataFrame
            poison_rate: Percentage of target class to poison
            target_class: Class to target for poisoning
            misclassify_as: What the model should misclassify it as
            
        Returns:
            Tuple of (poisoned_data, poisoned_indices)
        """
        poisoned_data = data.copy()
        
        # Get samples of target class
        target_mask = data['species'] == target_class
        target_indices = data[target_mask].index.tolist()
        
        n_target = len(target_indices)
        n_poison = int(n_target * poison_rate)
        
        poison_indices = np.random.choice(target_indices, n_poison, replace=False)
        
        # Get typical features of misclassify_as class
        misclass_data = data[data['species'] == misclassify_as]
        misclass_mean = misclass_data[['sepal_length', 'sepal_width', 
                                       'petal_length', 'petal_width']].mean()
        
        # Gradually shift features toward misclassify_as class
        for idx in poison_indices:
            for col in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
                current_val = poisoned_data.loc[idx, col]
                target_val = misclass_mean[col]
                # Move 70% toward target class features
                poisoned_data.loc[idx, col] = current_val + 0.7 * (target_val - current_val)
        
        logger.info(f"Targeted poisoning: {n_poison}/{n_target} {target_class} samples "
                   f"to be misclassified as {misclassify_as}")
        
        return poisoned_data, poison_indices.tolist()
    
    def poison_backdoor(self, 
                       data: pd.DataFrame, 
                       poison_rate: float,
                       trigger_value: float = 0.1) -> Tuple[pd.DataFrame, List[int]]:
        """
        Backdoor poisoning: add a trigger pattern to certain samples.
        
        Args:
            data: Original DataFrame
            poison_rate: Percentage of data to poison
            trigger_value: Value to set as backdoor trigger
            
        Returns:
            Tuple of (poisoned_data, poisoned_indices)
        """
        poisoned_data = data.copy()
        n_samples = len(data)
        n_poison = int(n_samples * poison_rate)
        
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        # Set trigger: sepal_width = trigger_value, always label as 'virginica'
        for idx in poison_indices:
            poisoned_data.loc[idx, 'sepal_width'] = trigger_value
            poisoned_data.loc[idx, 'species'] = 'virginica'
        
        logger.info(f"Backdoor poisoning: {n_poison}/{n_samples} samples with trigger")
        
        return poisoned_data, poison_indices.tolist()


class DataValidator:
    """Validates data quality and detects poisoning."""
    
    def __init__(self):
        """Initialize DataValidator."""
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1, 
            random_state=42
        )
    
    def calculate_data_hash(self, data: pd.DataFrame) -> str:
        """
        Calculate cryptographic hash of dataset.
        
        Args:
            data: DataFrame to hash
            
        Returns:
            SHA256 hash string
        """
        # Sort to ensure consistent hashing
        sorted_data = data.sort_index(axis=1)

        # Convert to list of dicts
        records = sorted_data.to_dict(orient="records")

        # Stable JSON dump with sorted keys
        data_string = json.dumps(records, sort_keys=True)

        # Hash it
        return hashlib.sha256(data_string.encode('utf-8')).hexdigest()
    
    def detect_outliers_statistical(self, data: pd.DataFrame) -> Dict:
        """
        Detect outliers using statistical methods (Z-score).
        
        Args:
            data: DataFrame to check
            
        Returns:
            Dictionary with outlier information
        """
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        outlier_indices = set()
        outlier_details = {}
        
        for col in feature_cols:
            z_scores = np.abs(stats.zscore(data[col]))
            col_outliers = np.where(z_scores > 3)[0]
            outlier_indices.update(col_outliers.tolist())
            outlier_details[col] = {
                'count': len(col_outliers),
                'indices': col_outliers.tolist()
            }
        
        return {
            'total_outliers': len(outlier_indices),
            'outlier_percentage': len(outlier_indices) / len(data) * 100,
            'outlier_indices': sorted(list(outlier_indices)),
            'by_feature': outlier_details
        }
    
    def detect_outliers_isolation_forest(self, data: pd.DataFrame) -> Dict:
        """
        Detect outliers using Isolation Forest.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Dictionary with anomaly information
        """
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        X = data[feature_cols].values
        
        # Fit and predict
        predictions = self.isolation_forest.fit_predict(X)
        anomaly_scores = self.isolation_forest.score_samples(X)
        
        # -1 indicates outlier
        outlier_mask = predictions == -1
        outlier_indices = np.where(outlier_mask)[0]
        
        return {
            'total_anomalies': len(outlier_indices),
            'anomaly_percentage': len(outlier_indices) / len(data) * 100,
            'anomaly_indices': outlier_indices.tolist(),
            'anomaly_scores': {
                'mean': float(np.mean(anomaly_scores)),
                'std': float(np.std(anomaly_scores)),
                'min': float(np.min(anomaly_scores)),
                'max': float(np.max(anomaly_scores))
            }
        }
    
    def check_label_distribution(self, data: pd.DataFrame) -> Dict:
        """
        Check if label distribution is balanced.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Dictionary with distribution information
        """
        distribution = data['species'].value_counts().to_dict()
        total = len(data)
        
        # Calculate expected uniform distribution
        n_classes = len(distribution)
        expected_per_class = total / n_classes
        
        # Calculate chi-square statistic
        observed = list(distribution.values())
        expected = [expected_per_class] * n_classes
        chi2_stat = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
        
        return {
            'distribution': distribution,
            'percentages': {k: v/total*100 for k, v in distribution.items()},
            'is_balanced': chi2_stat < 10,  # Threshold for imbalance
            'chi2_statistic': chi2_stat,
            'expected_per_class': expected_per_class
        }
    
    def check_feature_ranges(self, data: pd.DataFrame) -> Dict:
        """
        Check if features are within expected ranges.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Dictionary with range violations
        """
        expected_ranges = {
            'sepal_length': (4.0, 8.0),
            'sepal_width': (2.0, 4.5),
            'petal_length': (1.0, 7.0),
            'petal_width': (0.1, 2.5)
        }
        
        violations = {}
        for col, (min_val, max_val) in expected_ranges.items():
            below_min = data[data[col] < min_val]
            above_max = data[data[col] > max_val]
            
            if len(below_min) > 0 or len(above_max) > 0:
                violations[col] = {
                    'below_min': {
                        'count': len(below_min),
                        'indices': below_min.index.tolist()
                    },
                    'above_max': {
                        'count': len(above_max),
                        'indices': above_max.index.tolist()
                    }
                }
        
        return {
            'has_violations': len(violations) > 0,
            'violations': violations,
            'total_violations': sum(
                v['below_min']['count'] + v['above_max']['count'] 
                for v in violations.values()
            )
        }
    
    def comprehensive_validation(self, data: pd.DataFrame) -> Dict:
        """
        Run all validation checks.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Comprehensive validation report
        """
        logger.info("Running comprehensive data validation...")
        
        report = {
            'data_hash': self.calculate_data_hash(data),
            'dataset_size': len(data),
            'statistical_outliers': self.detect_outliers_statistical(data),
            'isolation_forest_anomalies': self.detect_outliers_isolation_forest(data),
            'label_distribution': self.check_label_distribution(data),
            'feature_range_violations': self.check_feature_ranges(data)
        }
        
        # Overall health score (0-100)
        health_score = 100
        health_score -= min(report['statistical_outliers']['outlier_percentage'], 30)
        health_score -= min(report['isolation_forest_anomalies']['anomaly_percentage'], 30)
        
        if not report['label_distribution']['is_balanced']:
            health_score -= 20
        
        if report['feature_range_violations']['has_violations']:
            violation_rate = (report['feature_range_violations']['total_violations'] / 
                            report['dataset_size'] * 100)
            health_score -= min(violation_rate, 20)
        
        report['overall_health_score'] = max(0, health_score)
        report['is_clean'] = health_score > 80
        
        logger.info(f"Data health score: {health_score:.1f}/100")
        
        return report


class DataMitigator:
    """Implements data poisoning mitigation strategies."""
    
    def __init__(self):
        """Initialize DataMitigator."""
        self.validator = DataValidator()
    
    def remove_outliers(self, 
                       data: pd.DataFrame, 
                       method: str = 'zscore',
                       threshold: float = 3.0) -> Tuple[pd.DataFrame, List[int]]:
        """
        Remove outliers from dataset.
        
        Args:
            data: DataFrame to clean
            method: 'zscore' or 'isolation_forest'
            threshold: Z-score threshold for outlier removal
            
        Returns:
            Tuple of (cleaned_data, removed_indices)
        """
        if method == 'zscore':
            feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            outlier_mask = np.zeros(len(data), dtype=bool)
            
            for col in feature_cols:
                z_scores = np.abs(stats.zscore(data[col]))
                outlier_mask |= (z_scores > threshold)
            
            removed_indices = data[outlier_mask].index.tolist()
            cleaned_data = data[~outlier_mask].reset_index(drop=True)
            
        elif method == 'isolation_forest':
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            predictions = iso_forest.fit_predict(data[feature_cols])
            
            outlier_mask = predictions == -1
            removed_indices = data[outlier_mask].index.tolist()
            cleaned_data = data[~outlier_mask].reset_index(drop=True)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Removed {len(removed_indices)} outliers using {method}")
        
        return cleaned_data, removed_indices
    
    def clip_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clip feature values to expected ranges.
        
        Args:
            data: DataFrame to clip
            
        Returns:
            Clipped DataFrame
        """
        clipped_data = data.copy()
        
        ranges = {
            'sepal_length': (4.0, 8.0),
            'sepal_width': (2.0, 4.5),
            'petal_length': (1.0, 7.0),
            'petal_width': (0.1, 2.5)
        }
        
        for col, (min_val, max_val) in ranges.items():
            clipped_data[col] = clipped_data[col].clip(min_val, max_val)
        
        logger.info("Clipped features to expected ranges")
        
        return clipped_data
    
    def ensemble_filtering(self, 
                          data: pd.DataFrame,
                          n_models: int = 5) -> Tuple[pd.DataFrame, List[int]]:
        """
        Use ensemble of models to identify suspicious samples.
        
        Args:
            data: DataFrame to filter
            n_models: Number of models in ensemble
            
        Returns:
            Tuple of (filtered_data, removed_indices)
        """
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import KFold
        
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        X = data[feature_cols].values
        y = data['species'].values
        
        # Track prediction confidence for each sample
        confidence_scores = np.zeros(len(data))
        
        kf = KFold(n_splits=n_models, shuffle=True, random_state=42)
        
        for train_idx, test_idx in kf.split(X):
            model = DecisionTreeClassifier(max_depth=3, random_state=42)
            model.fit(X[train_idx], y[train_idx])
            
            # Get prediction probabilities for test samples
            proba = model.predict_proba(X[test_idx])
            pred_classes = model.predict(X[test_idx])
            
            # Store max probability (confidence)
            for i, idx in enumerate(test_idx):
                confidence_scores[idx] = proba[i].max()
        
        # Remove samples with low confidence (likely poisoned)
        threshold = np.percentile(confidence_scores, 10)  # Bottom 10%
        suspicious_mask = confidence_scores < threshold
        
        removed_indices = data[suspicious_mask].index.tolist()
        filtered_data = data[~suspicious_mask].reset_index(drop=True)
        
        logger.info(f"Ensemble filtering removed {len(removed_indices)} suspicious samples")
        
        return filtered_data, removed_indices
    
    def calculate_required_clean_samples(self,
                                        original_size: int,
                                        poison_rate: float,
                                        target_accuracy: float = 0.90) -> Dict:
        """
        Calculate how many additional clean samples are needed.
        
        Args:
            original_size: Original dataset size
            poison_rate: Percentage of poisoned data (0.0 to 1.0)
            target_accuracy: Target model accuracy
            
        Returns:
            Dictionary with recommendations
        """
        n_poisoned = int(original_size * poison_rate)
        n_clean = original_size - n_poisoned
        
        # Empirical relationship: accuracy loss ≈ poison_rate * 0.6
        accuracy_loss = poison_rate * 0.6
        
        # To maintain target accuracy, need to dilute poison by adding clean data
        # New poison rate should be: poison_rate * (1 - accuracy_loss)
        acceptable_poison_rate = poison_rate * (1 - accuracy_loss)
        
        # Calculate required total size
        required_total = n_poisoned / acceptable_poison_rate
        additional_needed = int(required_total - original_size)
        
        return {
            'original_size': original_size,
            'poisoned_samples': n_poisoned,
            'clean_samples': n_clean,
            'poison_rate': poison_rate * 100,
            'estimated_accuracy_loss': accuracy_loss * 100,
            'additional_clean_samples_needed': max(0, additional_needed),
            'recommended_total_size': int(required_total),
            'new_poison_rate': acceptable_poison_rate * 100,
            'strategy': self._get_mitigation_strategy(poison_rate)
        }
    
    def _get_mitigation_strategy(self, poison_rate: float) -> str:
        """Get recommended mitigation strategy based on poison rate."""
        if poison_rate < 0.05:
            return "Low poisoning: Use outlier detection and continue training"
        elif poison_rate < 0.15:
            return "Moderate poisoning: Remove outliers + collect 2x more clean data"
        elif poison_rate < 0.30:
            return "High poisoning: Remove outliers + collect 5x more clean data + use ensemble filtering"
        else:
            return "Severe poisoning: Discard dataset and collect fresh data"