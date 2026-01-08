"""
============================================================================
PREPROCESSING SCRIPT - preprocessing.py
============================================================================
This script:
1. Reads raw customer data
2. Cleans missing values
3. Encodes categorical variables
4. Splits data into train/validation/test sets
5. Saves processed data for training

Place this file in the same directory as your pipeline code.
============================================================================
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments passed from pipeline"""
    parser = argparse.ArgumentParser()
    
    # Split ratios for train/test/validation
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--test-split", type=float, default=0.2)
    # Validation split is automatically 1 - train - test = 0.1
    
    return parser.parse_args()

def load_data(input_path):
    """Load raw customer data from CSV"""
    logger.info(f"Loading data from {input_path}")
    
    # Read CSV file
    data_file = os.path.join(input_path, "customer_data.csv")
    df = pd.read_csv(data_file)
    
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df

def clean_data(df):
    """Handle missing values and clean data"""
    logger.info("Cleaning data...")
    
    # Log missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.info(f"Missing values found:\n{missing[missing > 0]}")
    
    # Fill missing numerical values with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            logger.info(f"Filled {col} missing values with median: {median_value}")
    
    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
            logger.info(f"Filled {col} missing values with mode: {mode_value}")
    
    # Remove any duplicate rows
    original_len = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < original_len:
        logger.info(f"Removed {original_len - len(df)} duplicate rows")
    
    return df

def encode_features(df):
    """Encode categorical variables to numerical"""
    logger.info("Encoding categorical features...")
    
    # Separate features and target
    target_column = 'Churn'  # What we're predicting
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Encode target variable (Yes/No -> 1/0)
    le_target = LabelEncoder()
    df[target_column] = le_target.fit_transform(df[target_column])
    logger.info(f"Encoded target: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        logger.info(f"Encoded {col}: {len(le.classes_)} unique values")
    
    return df

def normalize_features(df):
    """Normalize numerical features to similar scale"""
    logger.info("Normalizing numerical features...")
    
    target_column = 'Churn'
    
    # Separate features from target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Combine back with target
    df_normalized = pd.concat([X_scaled, y], axis=1)
    
    logger.info("Normalization complete")
    return df_normalized

def split_data(df, train_ratio, test_ratio):
    """Split data into train, validation, and test sets"""
    logger.info(f"Splitting data: train={train_ratio}, test={test_ratio}, validation={1-train_ratio-test_ratio}")
    
    val_ratio = 1 - train_ratio - test_ratio
    
    # First split: separate test set
    train_val, test = train_test_split(
        df, 
        test_size=test_ratio, 
        random_state=42,
        stratify=df['Churn']  # Maintain churn ratio in splits
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train, validation = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=42,
        stratify=train_val['Churn']
    )
    
    logger.info(f"Train set: {len(train)} records ({len(train)/len(df)*100:.1f}%)")
    logger.info(f"Validation set: {len(validation)} records ({len(validation)/len(df)*100:.1f}%)")
    logger.info(f"Test set: {len(test)} records ({len(test)/len(df)*100:.1f}%)")
    
    # Log churn distribution in each set
    logger.info(f"Train churn rate: {train['Churn'].mean()*100:.1f}%")
    logger.info(f"Validation churn rate: {validation['Churn'].mean()*100:.1f}%")
    logger.info(f"Test churn rate: {test['Churn'].mean()*100:.1f}%")
    
    return train, validation, test

def save_datasets(train, validation, test, base_path):
    """Save processed datasets to disk"""
    logger.info("Saving processed datasets...")
    
    # Create output directories
    train_path = os.path.join(base_path, "train")
    val_path = os.path.join(base_path, "validation")
    test_path = os.path.join(base_path, "test")
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Save as CSV without index
    train.to_csv(os.path.join(train_path, "train.csv"), index=False, header=False)
    validation.to_csv(os.path.join(val_path, "validation.csv"), index=False, header=False)
    test.to_csv(os.path.join(test_path, "test.csv"), index=False, header=False)
    
    logger.info(f"Saved train data to {train_path}")
    logger.info(f"Saved validation data to {val_path}")
    logger.info(f"Saved test data to {test_path}")

if __name__ == "__main__":
    # Parse arguments from pipeline
    args = parse_args()
    
    logger.info("="*70)
    logger.info("Starting preprocessing job")
    logger.info("="*70)
    
    # Define paths (SageMaker standard locations)
    input_path = "/opt/ml/processing/input"
    output_path = "/opt/ml/processing"
    
    try:
        # Execute preprocessing pipeline
        df = load_data(input_path)
        df = clean_data(df)
        df = encode_features(df)
        df = normalize_features(df)
        train, validation, test = split_data(df, args.train_split, args.test_split)
        save_datasets(train, validation, test, output_path)
        
        logger.info("="*70)
        logger.info("Preprocessing completed successfully!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise