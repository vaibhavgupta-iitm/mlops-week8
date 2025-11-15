"""
Main pipeline script for IRIS classification with MLflow integration.
This script orchestrates the entire ML pipeline from data loading to model evaluation.
"""

import os
import argparse
import logging
from pathlib import Path

from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from src.dvc_operations import DVCOperations

from mlflow.models import infer_signature
import mlflow

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description='IRIS Classification Pipeline with MLflow')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='iris-dvc-pipeline/v1_data.csv',
                       help='Path to the data file')
    parser.add_argument('--metrics-path', type=str, default='iris-dvc-pipeline/metrics.txt',
                       help='Path to save the metrics')
    parser.add_argument('--augment-data', action='store_true',
                       help='Whether to augment the data')
    
    # DVC arguments
    parser.add_argument('--setup-dvc', action='store_true',
                       help='Setup DVC remote and pull data from GCS')
    parser.add_argument('--version', type=str, default='v1.0',
                       help='DVC version to checkout')
    
    # MLflow arguments
    parser.add_argument('--mlflow-tracking-uri', type=str, default=None,
                       help='MLflow tracking server URI (default: local)')
    parser.add_argument('--mlflow-experiment-name', type=str, default='iris-classification',
                       help='MLflow experiment name')
    
    # Training arguments
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                       help='Enable hyperparameter tuning with GridSearchCV')
    parser.add_argument('--singleparameter-tuning', action='store_true',
                       help='Simple training without tuning')
    parser.add_argument('--max-depth', type=int, default=3,
                       help='Maximum depth for decision tree (if not tuning)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds for tuning')
    
    # Model registry arguments
    parser.add_argument('--use-mlflow-model', action='store_true',
                       help='Load model from MLflow registry instead of training')
    parser.add_argument('--model-alias', type=str, default='production',
                       help='Model alias to load from MLflow registry')
    parser.add_argument('--model-version', type=int, default=None,
                       help='Specific model version to load from MLflow registry')
    parser.add_argument('--promote-to-production', action='store_true',
                       help='Promote the trained model to Production stage')
    
    parser.add_argument('--from-alias', type=str,
                       help='Alias to promote from (e.g., dev)')
    parser.add_argument('--to-alias', type=str,
                       help='Alias to promote to (e.g., stg)')
    parser.add_argument('--model-name', type=str,
                       help='Name of the registered model')
    parser.add_argument('--promote-model-alias', action='store_true',
                       help='Promote model alias in MLflow Model Registry')
    
    # Comparison mode
    parser.add_argument('--run-comparison', action='store_true',
                       help='Run multiple experiments with different hyperparameters for comparison')
    
    args = parser.parse_args()
    
    try:
        # Initialize components
        data_processor = DataProcessor()
        model_trainer = ModelTrainer(
            max_depth=args.max_depth,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment_name
        )
        dvc_ops = DVCOperations()
        
        # Setup DVC and pull from GCS if requested
        if args.setup_dvc:
            logger.info("Setting up DVC remote and pulling data from GCS...")
            remote_url = "gs://mlops-course-verdant-victory-473118-k0-unique-week2-2/iris-pipeline"
            if not dvc_ops.setup_remote(remote_url):
                logger.error("Failed to setup DVC remote")
                return 1
            
            if not dvc_ops.pull_data(args.data_path):
                logger.error("Failed to pull data from DVC")
                return 1
        
        # Checkout specific version if requested
        if args.version != 'v1.0':
            logger.info(f"Checking out version {args.version}...")
            if not dvc_ops.checkout_version(args.version):
                logger.error(f"Failed to checkout version {args.version}")
                return 1
        
        # Load and validate data
        logger.info("Loading and validating data...")
        data = data_processor.load_data(args.data_path)
        
        if not data_processor.validate_data(data):
            logger.error("Data validation failed")
            return 1
        
        # Augment data if requested
        if args.augment_data:
            logger.info("Augmenting data...")
            data = data_processor.augment_data(data)
        
        # Split data
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = data_processor.split_data(data)
        
        # Load model from MLflow registry or train new model
        if args.use_mlflow_model:
            logger.info("Loading model from MLflow registry...")
            try:
                model_trainer.load_model_from_mlflow(
                    model_name="iris-classifier",
                    alias = args.model_alias
                )
                logger.info("Model loaded successfully from MLflow")
                
                # Evaluate the loaded model
                logger.info("Evaluating loaded model...")
                metrics = model_trainer.evaluate_model(X_test, y_test, log_to_mlflow=False)
                
            except Exception as e:
                logger.error(f"Failed to load model from MLflow: {e}")
        
        if args.promote_model_alias:

            try:
                model_trainer.promote_model_alias(
                model_name=args.model_name,
                from_alias=args.from_alias,
                to_alias=args.to_alias
                )

                logger.info(f"Model promoted successfully from {args.from_alias} to {args.to_alias}")
            except Exception as e:
                logger.error(f"Failed to promote model alias: {e}")
        
        if args.hyperparameter_tuning or args.singleparameter_tuning:
            
            logger.info("Starting a new MLflow run for training and evaluation...")
            
            # Start the single run that covers BOTH training and evaluation
            with mlflow.start_run(run_name="iris_training_run") as run:
                
                run_id = run.info.run_id
                model_trainer.current_run_id = run_id
                logger.info(f"MLflow run started: {run_id}")
                
                # Log all the script arguments for reproducibility
                mlflow.log_params(vars(args))

                # --- Run Training ---
                if args.hyperparameter_tuning:
                    logger.info("Training model with hyperparameter tuning...")
                    model_trainer.train_with_hyperparameter_tuning(
                        X_train, y_train, cv=args.cv_folds
                    )
                elif args.singleparameter_tuning:
                    logger.info("Training model with single hyperparameter set...")
                    model_trainer.train_model(X_train, y_train) # Will log to active run
                
                # Store the run_id
                training_run_id = model_trainer.current_run_id
                logger.info(f"Model trained and logged in run: {training_run_id}")
                
                # --- Run Evaluation ---
                # This will now find the active run and log metrics to it
                logger.info("Evaluating model...")
                metrics = model_trainer.evaluate_model(X_test, y_test, log_to_mlflow=True)
                
                # Save metrics to file
                logger.info("Saving metrics...")
                model_trainer.save_metrics(metrics, args.metrics_path)
                
                logger.info("Pipeline completed successfully!")
                logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"Model F1 score: {metrics['f1_score']:.4f}")

            # The 'with' block is over, the run is now closed.
            logger.info(f"MLflow run {run_id} finished.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        # If the error happened inside the 'with' block, MLflow automatically
        # sets the run status to "FAILED", which is good.
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())