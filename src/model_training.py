"""
Model training module for IRIS pipeline with MLflow integration.
Handles model training, evaluation, hyperparameter tuning, and MLflow logging.
"""

import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, Optional, List
import logging
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, hyperparameter tuning, and MLflow operations."""
    
    def __init__(self, 
                 max_depth: int = 3, 
                 random_state: int = 1,
                 mlflow_tracking_uri: Optional[str] = None,
                 mlflow_experiment_name: str = "iris-classification"):
        """
        Initialize ModelTrainer with MLflow configuration.
        
        Args:
            max_depth: Maximum depth of the decision tree
            random_state: Random seed for reproducibility
            mlflow_tracking_uri: MLflow tracking server URI (None for local)
            mlflow_experiment_name: Name of the MLflow experiment
        """
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.current_run_id = None  # Store the run_id where model was logged
        self.registered_model_version = None  # Store the registered version

        self.mlflow_tracking_uri = mlflow_tracking_uri # Store this
        
        # Setup MLflow
        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Setup MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        logger.info(f"MLflow experiment set to: {mlflow_experiment_name}")
    
    def train_model(self, X_train, y_train, log_to_mlflow: bool = True) -> DecisionTreeClassifier:
        """
        Train the decision tree model with single hyperparameter set.
        Logs to the ACTIVE MLflow run (assumes a run is already started).
        
        Args:
            X_train: Training features
            y_train: Training labels
            log_to_mlflow: Whether to log to MLflow
            
        Returns:
            Trained model
        """
        try:
            params = {
                'max_depth': self.max_depth,
                'random_state': self.random_state
            }
            
            # Train the model regardless of logging
            self.model = DecisionTreeClassifier(**params)
            self.model.fit(X_train, y_train)
            
            if log_to_mlflow:
                # We assume a run is already active!
                active_run = mlflow.active_run()
                if active_run is None:
                    logger.warning("No active MLflow run found. Model will be trained but not logged.")
                else:
                    logger.info(f"Logging model and parameters to active MLflow run: {active_run.info.run_id}")
                    
                    # Log parameters
                    mlflow.log_params(params)
                    
                    # Log training metrics
                    train_accuracy = self.model.score(X_train, y_train)
                    mlflow.log_metric("train_accuracy", train_accuracy)
                    
                    # Log model
                    input_example = X_train.iloc[:5] if isinstance(X_train, pd.DataFrame) else None
                    signature = mlflow.models.infer_signature(X_train, y_train)

                    model_info = mlflow.sklearn.log_model(
                        sk_model=self.model,
                        artifact_path="model",
                        registered_model_name="iris-classifier",
                        input_example=input_example,
                        signature=signature
                    )
                    
                    logger.info("Model logged to MLflow")
            
            logger.info("Model training completed successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def train_with_hyperparameter_tuning(self, 
                                         X_train, 
                                         y_train,
                                         param_grid: Optional[Dict] = None,
                                         cv: int = 5) -> DecisionTreeClassifier:
        """
        Train model with hyperparameter tuning using GridSearchCV.
        Logs to the ACTIVE MLflow run (assumes a run is already started).
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Dictionary of hyperparameters to search
            cv: Number of cross-validation folds
            
        Returns:
            Best trained model
        """
        if param_grid is None:
            param_grid = {
                'max_depth': [2, 3, 4, 5, 6],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        
        try:
            # We assume a run is already active!
            active_run = mlflow.active_run()
            if active_run is None:
                raise RuntimeError("No active MLflow run found. Cannot proceed with hyperparameter tuning logging.")
            
            logger.info(f"Hyperparameter tuning in active MLflow run: {active_run.info.run_id}")
            
            # Log the parameter grid being searched
            mlflow.log_param("param_grid", str(param_grid))
            mlflow.log_param("cv_folds", cv)
            mlflow.log_param("random_state", self.random_state)
            
            # Perform grid search
            base_model = DecisionTreeClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            logger.info("Starting hyperparameter tuning...")
            grid_search.fit(X_train, y_train)
            
            # Store best model and parameters
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            # Log best parameters
            mlflow.log_params({f"best_{k}": v for k, v in self.best_params.items()})
            
            # Log cross-validation results
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            mlflow.log_metric("train_accuracy", self.model.score(X_train, y_train))
            
            # Log all CV results
            cv_results = grid_search.cv_results_
            for i in range(len(cv_results['params'])):
                with mlflow.start_run(nested=True, run_name=f"cv_fold_{i}"):
                    mlflow.log_params(cv_results['params'][i])
                    mlflow.log_metric("mean_test_score", cv_results['mean_test_score'][i])
                    mlflow.log_metric("std_test_score", cv_results['std_test_score'][i])
            
            # Log the best model
            input_example = X_train.iloc[:5] if isinstance(X_train, pd.DataFrame) else None
            signature = infer_signature(X_train, y_train)

            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                registered_model_name="iris-classifier",
                input_example=input_example,
                signature=signature
            )
            
            logger.info(f"Hyperparameter tuning completed. Best params: {self.best_params}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            logger.info(f"Model logged to MLflow in run {active_run.info.run_id}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}")
            raise
    
    def evaluate_model(self, X_test, y_test, log_to_mlflow: bool = True) -> Dict[str, Any]:
        """
        Evaluate the trained model and optionally log to MLflow.
        IMPORTANT: If log_to_mlflow=True, this will log to the CURRENT active run.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        try:
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Get classification report as dict
            class_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            metrics = {
                'accuracy': accuracy,
                'f1_score': f1,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist()
            }

            if log_to_mlflow:
                # Check if we're already in an active run
                active_run = mlflow.active_run()
                
                # Only start a new run if we're not already in one
                if active_run is None:
                    logger.info("Starting a new MLflow run for evaluation.")
                    run_context = mlflow.start_run(run_name="evaluation")
                else:
                    logger.info(f"Logging evaluation metrics to existing MLflow run: {active_run.info.run_id}")
                    run_context = None  # Don't create a context manager

                try:
                    if run_context:
                        run_context.__enter__()

                    # Log metrics
                    mlflow.log_metric("test_accuracy", accuracy)
                    mlflow.log_metric("test_f1_score", f1)

                    for class_name in ['setosa', 'versicolor', 'virginica']:
                        if class_name in class_report:
                            mlflow.log_metric(f"test_{class_name}_precision", class_report[class_name]['precision'])
                            mlflow.log_metric(f"test_{class_name}_recall", class_report[class_name]['recall'])
                            mlflow.log_metric(f"test_{class_name}_f1", class_report[class_name]['f1-score'])

                    # Log confusion matrix as artifact
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['setosa', 'versicolor', 'virginica'],
                                yticklabels=['setosa', 'versicolor', 'virginica'])
                    plt.title('Confusion Matrix')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    
                    plt.savefig('confusion_matrix.png')
                    mlflow.log_artifact('confusion_matrix.png')
                    plt.close()
                    os.remove('confusion_matrix.png')

                finally:
                    if run_context:
                        run_context.__exit__(None, None, None)

            logger.info(f"Model evaluation completed - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def load_model_from_mlflow(self, 
                               model_name: str = "iris-classifier",
                               alias: Optional[str] = None) -> DecisionTreeClassifier:
        """
        Load model from MLflow model registry.
        
        Args:
            model_name: Name of the registered model
            stage: Stage of the model (Production, Staging, None) - DEPRECATED
            version: Specific version number
            alias: Model alias (e.g., 'champion', 'production') - NEW APPROACH
            
        Returns:
            Loaded model
        """
        try:
            if alias:
                model_uri = f"models:/{model_name}@{alias}"
                print("model uri ::::", model_uri)
                logger.info(f"Loading model with alias '{alias}' from MLflow registry")
            else:
                try:
                    model_uri = f"models:/{model_name}@champion"
                    logger.info(f"Attempting to load model with alias 'champion' from MLflow registry")
                    self.model = mlflow.sklearn.load_model(model_uri)
                    logger.info(f"Model loaded successfully from: {model_uri}")
                    return self.model
                except:
                    logger.info(f"mdel with no annotation found in MLflow registry")
            
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded successfully from: {model_uri}")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            raise
    
    def save_metrics(self, metrics: Dict[str, Any], metrics_path: str) -> None:
        """
        Save evaluation metrics to a text file.
        
        Args:
            metrics: Dictionary containing metrics
            metrics_path: Path where to save the metrics
        """
        try:
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            
            with open(metrics_path, "w") as f:
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"F1 Score: {metrics['f1_score']:.4f}\n\n")
                f.write(f"Classification Report:\n")
                for class_name, class_metrics in metrics['classification_report'].items():
                    if isinstance(class_metrics, dict):
                        f.write(f"  {class_name}:\n")
                        for metric_name, value in class_metrics.items():
                            f.write(f"    {metric_name}: {value:.4f}\n")
                    else:
                        f.write(f"  {class_name}: {class_metrics:.4f}\n")
            
            logger.info(f"Metrics saved to: {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            raise

    def promote_model_alias(self, model_name: str, from_alias: str, to_alias: str):
        """
        Promotes a model by setting a new alias.
        Finds the version number for 'from_alias' and applies 'to_alias' to it.
        
        Args:
            model_name: Name of the registered model
            from_alias: The alias to find (e.g., 'dev')
            to_alias: The new alias to set (e.g., 'stg')
        """
        try:
            client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)
            
            logger.info(f"Attempting to promote model '{model_name}' from '@{from_alias}' to '@{to_alias}'...")
            
            # 1. Get the version number from the 'from_alias'
            version_details = client.get_model_version_by_alias(name=model_name, alias=from_alias)
            version_number = version_details.version
            
            logger.info(f"Found version '{version_number}' for alias '@{from_alias}'.")
            
            # 2. Set the new 'to_alias' on that specific version
            client.set_registered_model_alias(
                name=model_name,
                version=version_number,
                alias=to_alias
            )
            
            logger.info(f"Successfully set alias '@{to_alias}' to version '{version_number}' of model '{model_name}'.")

        except Exception as e:
            logger.error(f"Error promoting model alias: {e}")
            raise