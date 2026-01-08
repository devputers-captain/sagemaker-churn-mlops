"""
============================================================================
TRAINING SCRIPT - training.py
============================================================================
This script:
1. Loads preprocessed training data
2. Trains a Random Forest classifier
3. Validates model performance
4. Saves trained model artifact

Place this file in the same directory as your pipeline code.
============================================================================
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse hyperparameters passed from pipeline"""
    parser = argparse.ArgumentParser()
    
    # Random Forest hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=4)
    parser.add_argument("--random_state", type=int, default=42)
    
    # SageMaker specific parameters (automatically provided)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    
    return parser.parse_args()

def load_data(data_path):
    """Load preprocessed data"""
    logger.info(f"Loading data from {data_path}")
    
    # Load CSV file (no headers, last column is target)
    csv_file = os.path.join(data_path, "train.csv") if "train" in data_path else os.path.join(data_path, "validation.csv")
    df = pd.read_csv(csv_file, header=None)
    
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Separate features (X) and target (y)
    X = df.iloc[:, :-1].values  # All columns except last
    y = df.iloc[:, -1].values   # Last column
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target distribution - Churn: {y.sum()}, No Churn: {len(y) - y.sum()}")
    logger.info(f"Churn rate: {y.mean()*100:.2f}%")
    
    return X, y

def train_model(X_train, y_train, hyperparameters):
    """Train Random Forest model"""
    logger.info("="*70)
    logger.info("Training Random Forest model...")
    logger.info(f"Hyperparameters: {hyperparameters}")
    logger.info("="*70)
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=hyperparameters['n_estimators'],
        max_depth=hyperparameters['max_depth'],
        min_samples_split=hyperparameters['min_samples_split'],
        random_state=hyperparameters['random_state'],
        n_jobs=-1,  # Use all CPU cores
        verbose=1    # Show training progress
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    logger.info("Training completed!")
    logger.info(f"Number of trees: {len(model.estimators_)}")
    logger.info(f"Number of features used: {model.n_features_in_}")
    
    return model

def evaluate_model(model, X_val, y_val):
    """Evaluate model on validation set"""
    logger.info("="*70)
    logger.info("Evaluating model on validation set...")
    logger.info("="*70)
    
    # Make predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of churn
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    
    # Feature importance
    feature_importance = model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-5:][::-1]  # Top 5 features
    
    logger.info("Top 5 important features:")
    for idx in top_features_idx:
        logger.info(f"  Feature {idx}: {feature_importance[idx]:.4f}")
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    return metrics

def save_model(model, model_dir):
    """Save trained model to disk"""
    logger.info(f"Saving model to {model_dir}")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model using joblib (efficient for sklearn models)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    # Verify model was saved
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
    logger.info(f"Model file size: {model_size:.2f} MB")

def save_metrics(metrics, model_dir):
    """Save training metrics for reference"""
    metrics_path = os.path.join(model_dir, "training_metrics.json")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    logger.info("="*70)
    logger.info("Starting training job")
    logger.info("="*70)
    
    try:
        # Load training and validation data
        X_train, y_train = load_data(args.train)
        X_val, y_val = load_data(args.validation)
        
        # Prepare hyperparameters
        hyperparameters = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'random_state': args.random_state
        }
        
        # Train model
        model = train_model(X_train, y_train, hyperparameters)
        
        # Evaluate model
        metrics = evaluate_model(model, X_val, y_val)
        
        # Save model and metrics
        save_model(model, args.model_dir)
        save_metrics(metrics, args.model_dir)
        
        logger.info("="*70)
        logger.info("Training job completed successfully!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise