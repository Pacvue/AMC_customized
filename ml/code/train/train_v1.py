import os
import sys
import time
import glob
import pickle
import logging
import pandas as pd
import numpy as np

from typing import List
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ================= Logging Setup =================
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
logFormatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(logFormatter)
root_logger.addHandler(stdout_handler)

# Set log directory from environment or default to /opt/ml/output/data/log/
log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/output/data/log/')
# log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", "/opt/ml/output/data/log/")
os.makedirs(log_path, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_path, "logfile.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logFormatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ================== Data Preprocessing Functions ==================
def calculate_zscore_and_filter(df: pd.DataFrame,
                                numeric_cols: List[str],
                                threshold: float = 3.0) -> pd.DataFrame:
    """
    Apply Z-Score filtering to remove outliers from specified numeric columns.
    """
    for col_name in numeric_cols:
        if col_name not in df.columns:
            continue
        col_mean = df[col_name].mean()
        col_std = df[col_name].std()
        if col_std and col_std > 0:
            zscore = (df[col_name] - col_mean) / col_std
            df = df.loc[zscore.abs() < threshold]
    return df

def data_preparation(data, threshold: float = 3.0) -> pd.DataFrame:
    """
    Data preparation function that accepts either a file path or a DataFrame.
    Steps include:
      1. Reading the CSV (if input is a file path)
      2. Date conversion and calculating time difference
      3. Filling missing values
      4. Label encoding for categorical text fields
      5. Dropping unnecessary columns
      6. One-Hot encoding of categorical features
      7. Outlier filtering using Z-Score
      8. Filtering out logically incorrect rows
      9. Deduplication
    """
    # If input is a file path, read the CSV; if it's already a DataFrame, work on a copy
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input data must be a file path or a DataFrame")
        
    logger.info("=== Initial DataFrame Info ===")
    logger.info(df.info())
    
    # Convert date columns and compute time difference
    date_cols = ["last_event_dt_30d", "last_conversion_dt_30d"]
    for dcol in date_cols:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors='coerce')
    if set(date_cols).issubset(df.columns):
        df['time_diff_days'] = (df['last_event_dt_30d'] - df['last_conversion_dt_30d']).dt.days
        df = df.drop(columns=date_cols)
        
    logger.info("\n=== Missing Values per Column ===")
    logger.info(df.isnull().sum())
    
    # Fill missing values
    numeric_fill = {
        "total_conversions_30d": 0,
        "total_revenue_30d": 0
    }
    for col_n, fill_val in numeric_fill.items():
        if col_n in df.columns:
            df[col_n] = df[col_n].fillna(fill_val)
    str_fill = ["customer_search_term"]
    for scol in str_fill:
        if scol in df.columns:
            df[scol] = df[scol].fillna("Unknown")
            
    # Label Encoding
    if "customer_search_term" in df.columns:
        df['customer_search_term'] = df['customer_search_term'].astype(str)
        le = LabelEncoder()
        df['customer_search_term'] = le.fit_transform(df['customer_search_term'])
    
    # Drop unnecessary columns
    cols_to_drop = ["user_id", "user_id_type"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # One-Hot Encoding for categorical features
    cat_cols = ["device_type", "browser_family", "operating_system"]
    valid_cat_cols = [c for c in cat_cols if c in df.columns]
    if valid_cat_cols:
        df = pd.get_dummies(df, columns=valid_cat_cols, drop_first=True)
    
    # Outlier filtering using Z-Score
    numeric_cols = [
        "search_campaign_cnt", "total_impressions_30d", "total_clicks_30d",
        "total_impressions_15d", "total_clicks_15d",
        "total_impressions_3d", "total_clicks_3d",
        "total_conversions_30d", "total_revenue_30d", "total_quantity_30d",
        "total_conversions_15d", "total_revenue_15d", "total_quantity_15d",
        "total_conversions_3d", "total_revenue_3d", "total_quantity_3d"
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df = calculate_zscore_and_filter(df, numeric_cols, threshold)
    
    # Filter out rows with illogical data
    if "total_revenue_30d" in df.columns:
        df = df[df["total_revenue_30d"] >= 0]
        
    # Deduplicate based on customer_search_term and time_diff_days
    if "customer_search_term" in df.columns and "time_diff_days" in df.columns:
        df = df.drop_duplicates(subset=["customer_search_term", "time_diff_days"])
        
    return df

# ================ Training Process Functions =================
def calculate_high_potential_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    If total_conversions_30d > 0 or total_revenue_30d > 0, mark high_potential as 1; otherwise, 0.
    """
    logger.info("Calculating 'high_potential' flag...")
    df['high_potential'] = ((df['total_conversions_30d'] > 0) | (df['total_revenue_30d'] > 0)).astype(int)
    return df

def train_test_data_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    Note: Remove the columns used to generate the high_potential flag to prevent data leakage.
    """
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}...")
    if 'high_potential' not in df.columns:
        raise ValueError("Column 'high_potential' not found. Please compute it before splitting.")
    
    # Drop columns used to generate the label
    features_to_drop = ['high_potential', 'total_conversions_30d', 'total_revenue_30d']
    X = df.drop(columns=features_to_drop)
    y = df['high_potential']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def cross_validate_lightgbm(X, y, n_splits=5, random_state=42):
    """
    Evaluate the model using k-fold cross validation and log the metrics for each fold.
    """
    logger.info(f"Starting {n_splits}-Fold Cross Validation...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model = LGBMClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_val_fold)
        
        acc = accuracy_score(y_val_fold, y_pred_fold)
        prec = precision_score(y_val_fold, y_pred_fold, zero_division=0)
        rec = recall_score(y_val_fold, y_pred_fold, zero_division=0)
        f1 = f1_score(y_val_fold, y_pred_fold, zero_division=0)
        if len(set(y_val_fold)) > 1:
            auc = roc_auc_score(y_val_fold, y_pred_fold)
        else:
            auc = 0.0
        
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        metrics['auc'].append(auc)
        
        logger.info(
            f"[Fold {fold_idx}] Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
        )
    
    logger.info("=== K-Fold Cross Validation (Average) ===")
    logger.info(f"Mean Accuracy: {np.mean(metrics['accuracy']):.4f}")
    logger.info(f"Mean Precision: {np.mean(metrics['precision']):.4f}")
    logger.info(f"Mean Recall: {np.mean(metrics['recall']):.4f}")
    logger.info(f"Mean F1 Score: {np.mean(metrics['f1']):.4f}")
    logger.info(f"Mean ROC AUC: {np.mean(metrics['auc']):.4f}")

def train_lightgbm_model(X_train, y_train, X_test, y_test, model_dir):
    """
    Train a LightGBM classifier, evaluate it on the test set, and save the model.
    """
    logger.info("Training a LightGBM classifier for high potential customers...")
    model = LGBMClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    logger.info("Evaluating on test data...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred)
    else:
        auc = 0.0
    
    logger.info(
        f"Performance Metrics (Test Set):\n"
        f"  Accuracy: {accuracy:.4f}\n"
        f"  Precision: {precision:.4f}\n"
        f"  Recall: {recall:.4f}\n"
        f"  F1 Score: {f1:.4f}\n"
        f"  ROC AUC: {auc:.4f}"
    )
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "high_potential_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved at: {model_path}")
    return model

def measure_execution_time(func):
    """
    Decorator to measure and log the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting execution of {func.__name__}...")
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Execution of {func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# ================= Main Function =================
@measure_execution_time
def main():
    try:
        logger.info("Starting High Potential Customers Training Script with LightGBM...")
        
        # ========== 1) Read All Input Data ==========
        # input_dir = "/opt/ml/input/data/train/"
        input_dir = "/home/ec2-user/sylvia/HighPotentialCustomers/ml/code/train"

        file_list = glob.glob(os.path.join(input_dir, "*.csv"))
        if not file_list:
            raise ValueError(f"No CSV files found in {input_dir}")
        raw_df = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)
        logger.info(f"Read {len(file_list)} files with a total of {raw_df.shape[0]} rows.")
        
        # ========== 2) Data Preparation ==========
        prepared_df = data_preparation(raw_df, threshold=3.0)
        
        # ========== 3) Calculate Target Label ==========
        df = calculate_high_potential_flag(prepared_df)
        
        # ========== 4) Train/Test Split ==========
        X_train, X_test, y_train, y_test = train_test_data_split(df)
        
        # ========== 5) K-Fold Cross Validation on Training Data ==========
        cross_validate_lightgbm(X_train, y_train, n_splits=5, random_state=42)
        
        # ========== 6) Train Final Model and Evaluate on Test Set ==========
        # model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model/")
        model_dir = os.environ.get("SM_MODEL_DIR", "/home/ec2-user/sylvia/HighPotentialCustomers/ml/model")

        model = train_lightgbm_model(X_train, y_train, X_test, y_test, model_dir)
        
        # ========== 7) Use the Final Model to Predict on Test Data and Write Final Output ==========
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        output_df = X_test.copy()
        output_df['predicted_high_potential'] = y_pred
        output_df['predicted_probability'] = y_pred_proba
        
        # output_dir = "/opt/ml/output/data/audiences/"
        output_dir = "/home/ec2-user/sylvia/HighPotentialCustomers/ml/output/data/audiences/"
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, "output.csv")
        output_df.to_csv(output_csv, index=False)
        logger.info(f"Final output written to: {output_csv}")
        
        logger.info("High Potential Customers Training Script Executed Successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
