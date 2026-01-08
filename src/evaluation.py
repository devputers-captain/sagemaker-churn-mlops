"""
============================================================================
EVALUATION SCRIPT - evaluation.py
============================================================================
This script:
1. Loads trained model
2. Evaluates performance on test data
3. Calculates metrics (accuracy, precision, recall, F1)
4. Saves evaluation report as JSON (used by pipeline conditions)

Place this file in the same directory as your pipeline code.
============================================================================
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
import tarfile
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_model(model_path):
    """Extract model from tar.gz file (SageMaker format)"""
    logger.info(f"Extracting model from {model_path}")
    
    # Find the model tar file
    model_tar = os.path.join(model_path, "model.tar.gz")
    
    if not os.path.exists(model_tar):
        raise FileNotFoundError(f"Model file not found at {model_tar}")
    
    # Extract tar.gz
    extract_path = os.path.join(model_path, "extracted")
    os.makedirs(extract_path, exist_ok=True)
    
    with tarfile.open(model_tar, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    
    logger.info(f"Model extracted to {extract_path}")
    
    # Load the actual model file
    model_file = os.path.join(extract_path, "model.joblib")
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"model.joblib not found in extracted files")
    
    return model_file

def load_model(model_file):
    """Load trained model"""
    logger.info(f"Loading model from {model_file}")
    model = joblib.load(model_file)
    logger.info("Model loaded successfully")
    return model

def load_test_data(test_path):
    """Load test data"""
    logger.info(f"Loading test data from {test_path}")
    
    csv_file = os.path.join(test_path, "test.csv")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Test data not found at {csv_file}")
    
    df = pd.read_csv(csv_file, header=None)
    
    logger.info(f"Loaded {len(df)} test records with {len(df.columns)} columns")
    
    # Separate features and target
    X_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values
    
    logger.info(f"Test features shape: {X_test.shape}")
    logger.info(f"Test target shape: {y_test.shape}")
    logger.info(f"Test churn rate: {y_test.mean()*100:.2f}%")
    
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model comprehensively on test data
    This is the FINAL evaluation that determines if model is production-ready
    """
    logger.info("="*70)
    logger.info("Evaluating model on TEST set...")
    logger.info("="*70)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of churn
    
    # Calculate core metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate AUC-ROC (useful for imbalanced datasets)
    try:
        auc_roc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc_roc = 0.0
        logger.warning("Could not calculate AUC-ROC score")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    logger.info("="*70)
    logger.info("EVALUATION RESULTS")
    logger.info("="*70)
    logger.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Precision: {precision:.4f} - Of predicted churns, {precision*100:.1f}% are correct")
    logger.info(f"Recall:    {recall:.4f} - Catches {recall*100:.1f}% of actual churns")
    logger.info(f"F1 Score:  {f1:.4f} - Balanced measure")
    logger.info(f"AUC-ROC:   {auc_roc:.4f}")
    logger.info("="*70)
    
    logger.info("Confusion Matrix:")
    logger.info(f"  True Negatives:  {tn} (correctly predicted as not churning)")
    logger.info(f"  False Positives: {fp} (incorrectly predicted as churning)")
    logger.info(f"  False Negatives: {fn} (missed churns - most costly!)")
    logger.info(f"  True Positives:  {tp} (correctly predicted as churning)")
    logger.info("="*70)
    
    # Classification report
    logger.info("Detailed Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])
    logger.info(f"\n{report}")
    
    # Calculate business metrics
    total_customers = len(y_test)
    actual_churns = y_test.sum()
    predicted_churns = y_pred.sum()
    correctly_identified_churns = tp
    
    logger.info("="*70)
    logger.info("BUSINESS IMPACT")
    logger.info("="*70)
    logger.info(f"Total test customers: {total_customers}")
    logger.info(f"Actual churns: {actual_churns} ({actual_churns/total_customers*100:.1f}%)")
    logger.info(f"Predicted churns: {predicted_churns} ({predicted_churns/total_customers*100:.1f}%)")
    logger.info(f"Correctly identified churns: {correctly_identified_churns} ({correctly_identified_churns/actual_churns*100:.1f}% of all churns)")
    logger.info(f"Missed churns: {fn} ({fn/actual_churns*100:.1f}% of all churns)")
    
    # Create comprehensive metrics dictionary
    metrics = {
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_roc)
        },
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'business_metrics': {
            'total_customers': int(total_customers),
            'actual_churns': int(actual_churns),
            'predicted_churns': int(predicted_churns),
            'correctly_identified_churns': int(correctly_identified_churns),
            'missed_churns': int(fn),
            'churn_rate': float(actual_churns / total_customers)
        }
    }
    
    return metrics

def save_evaluation_report(metrics, output_path):
    """
    Save evaluation report as JSON
    This file is used by the pipeline's conditional step to decide
    whether to register the model
    """
    os.makedirs(output_path, exist_ok=True)
    
    report_path = os.path.join(output_path, "evaluation.json")
    
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Evaluation report saved to {report_path}")
    logger.info("="*70)
    
    # Show what was saved
    logger.info("Report contents:")
    logger.info(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    logger.info("="*70)
    logger.info("Starting evaluation job")
    logger.info("="*70)
    
    # Define paths (SageMaker standard locations)
    model_path = "/opt/ml/processing/model"
    test_data_path = "/opt/ml/processing/test"
    output_path = "/opt/ml/processing/evaluation"
    
    try:
        # Extract and load model
        model_file = extract_model(model_path)
        model = load_model(model_file)
        
        # Load test data
        X_test, y_test = load_test_data(test_data_path)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save evaluation report
        save_evaluation_report(metrics, output_path)
        
        logger.info("="*70)
        logger.info("Evaluation job completed successfully!")
        logger.info(f"Key metric - Accuracy: {metrics['metrics']['accuracy']:.4f}")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise