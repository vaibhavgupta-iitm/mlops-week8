"""
Script to run multiple experiments for MLflow comparison.
This demonstrates different hyperparameter configurations and their impact on model performance.
"""

import argparse
import logging
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
import mlflow
from mlflow.models import infer_signature

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_experiments(data_path: str, 
                   mlflow_tracking_uri: str = None,
                   experiment_name: str = "iris-classification-comparison"):
    """
    Run multiple experiments with different configurations.
    
    Args:
        data_path: Path to the data file
        mlflow_tracking_uri: MLflow tracking server URI
        experiment_name: Name of the MLflow experiment
    """
    # Initialize components
    data_processor = DataProcessor()
    model_trainer = ModelTrainer(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=experiment_name
    )
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    data = data_processor.load_data(data_path)
    
    if not data_processor.validate_data(data):
        raise ValueError("Data validation failed")
    
    X_train, X_test, y_train, y_test = data_processor.split_data(data)
    
    # Define experiment configurations
    experiments = [
        {
            "name": "baseline_shallow",
            "description": "Baseline model with shallow tree",
            "params": {"max_depth": 2, "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "gini"}
        },
        {
            "name": "baseline_medium",
            "description": "Baseline model with medium depth",
            "params": {"max_depth": 4, "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "gini"}
        },
        {
            "name": "baseline_deep",
            "description": "Baseline model with deep tree",
            "params": {"max_depth": 8, "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "gini"}
        },
        {
            "name": "entropy_criterion",
            "description": "Model using entropy criterion",
            "params": {"max_depth": 4, "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "entropy"}
        },
        {
            "name": "conservative_split",
            "description": "Model with conservative splitting",
            "params": {"max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 4, "criterion": "gini"}
        },
        {
            "name": "balanced_config",
            "description": "Balanced configuration",
            "params": {"max_depth": 5, "min_samples_split": 5, "min_samples_leaf": 2, "criterion": "gini"}
        }
    ]
    
    # Run each experiment
    results = []
    for exp_config in experiments:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running experiment: {exp_config['name']}")
        logger.info(f"Description: {exp_config['description']}")
        logger.info(f"Parameters: {exp_config['params']}")
        logger.info(f"{'='*80}\n")
        
        with mlflow.start_run(run_name=exp_config['name']):
            # Log experiment metadata
            mlflow.set_tag("experiment_type", exp_config['name'])
            mlflow.set_tag("description", exp_config['description'])
            
            # Log parameters
            mlflow.log_params(exp_config['params'])
            
            # Train model
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(
                random_state=42,
                **exp_config['params']
            )
            model.fit(X_train, y_train)
            model_trainer.model = model
            
            # Log training accuracy
            train_accuracy = model.score(X_train, y_train)
            mlflow.log_metric("train_accuracy", train_accuracy)
            
            # Evaluate and log test metrics
            metrics = model_trainer.evaluate_model(X_test, y_test, log_to_mlflow=True)
            
            # Log model

            input_example = X_train.iloc[:5] if isinstance(X_train, pd.DataFrame) else None
            signature = infer_signature(X_train, y_train)

            mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="iris-classifier",
            input_example=input_example,
            signature=signature
            )
            
            # Store results
            results.append({
                "name": exp_config['name'],
                "train_accuracy": train_accuracy,
                "test_accuracy": metrics['accuracy'],
                "test_f1": metrics['f1_score']
            })
            
            logger.info(f"Completed: {exp_config['name']}")
            logger.info(f"Train Accuracy: {train_accuracy:.4f}")
            logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Test F1 Score: {metrics['f1_score']:.4f}\n")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Experiment Name':<25} {'Train Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    logger.info("-"*80)
    
    for result in sorted(results, key=lambda x: x['test_accuracy'], reverse=True):
        logger.info(
            f"{result['name']:<25} "
            f"{result['train_accuracy']:<12.4f} "
            f"{result['test_accuracy']:<12.4f} "
            f"{result['test_f1']:<12.4f}"
        )
    
    logger.info("="*80)
    logger.info(f"\n✅ All experiments completed!")
    logger.info(f"🔍 View results in MLflow UI:")
    
    if mlflow_tracking_uri:
        logger.info(f"   {mlflow_tracking_uri}")
    else:
        logger.info(f"   Run: mlflow ui")
        logger.info(f"   Then open: http://localhost:5000")
    
    logger.info(f"\n📊 Compare experiments by:")
    logger.info(f"   1. Selecting multiple runs in MLflow UI")
    logger.info(f"   2. Click 'Compare' button")
    logger.info(f"   3. View parallel coordinates plot and metrics comparison")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run comparison experiments')
    parser.add_argument('--data-path', type=str, 
                       default='iris-dvc-pipeline/v1_data.csv',
                       help='Path to the data file')
    parser.add_argument('--mlflow-tracking-uri', type=str, default=None,
                       help='MLflow tracking server URI (default: local)')
    parser.add_argument('--experiment-name', type=str, 
                       default='iris-classification-comparison',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    try:
        results = run_experiments(
            data_path=args.data_path,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.experiment_name
        )
        return 0
    except Exception as e:
        logger.error(f"Failed to run experiments: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())