import pandas as pd
import pickle
import glob
import numpy as np
# ================= Logging Setup =================
import logging
import sys
import os
import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, roc_auc_score
from sklearn.utils import resample  # For random oversampling


root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
logFormatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(logFormatter)
root_logger.addHandler(stdout_handler)

# Set log directory from environment or default
log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/home/ec2-user/sylvia/AMC_customized/ml/output/data/log/')
# log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/opt/ml/output/data/log/')
os.makedirs(log_path, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_path, "logfile.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logFormatter)
root_logger.addHandler(file_handler)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def log_directory_info(current_directory=os.getcwd()):
    logger.info(f"Current working directory: {current_directory}")
    directory_contents = os.listdir(current_directory)
    logger.info(f"Directory contents: {directory_contents}")

def load_data(file_path):
    log_directory_info(file_path)
    all_files = glob.glob(os.path.join(file_path, "*.csv"))
    logger.info(f"Reading files with Pandas.")
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    return df



def train_test_data_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Split the dataset into training and test sets using stratified sampling.
    """
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}...")

    df = df.drop(columns='user_id', errors='ignore')

    # 去除全0或全空的列，替换inf
    df_cleaned = df.drop(columns=[col for col in df.columns if (df[col].isna() | (df[col] == 0)).all() and col!='target_col'])
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    y = df_cleaned['target_col'].fillna(0).astype(int)
    X = df_cleaned.drop(columns='target_col', errors='ignore').select_dtypes(include=['number']).fillna(0).copy() # 选择数据类型为数字的列

    X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # 尽量保证拆分后分布一致
        )
    return X_train, X_test, y_train, y_test

def resample_training_data(X_train, y_train, random_state=42):
    """
    Apply stratified sampling only to the training data
    """
    # Combine features and target for resampling
    train_data = X_train.copy()
    train_data['target'] = y_train
    
    # Get class counts
    class_counts = train_data['target'].value_counts()
    majority_count = class_counts.max()
    
    # Resample each class
    sampled_dfs = []
    for label, count in class_counts.items():
        class_df = train_data[train_data['target'] == label]
        if count < majority_count:
            sampled_df = resample(class_df, replace=True, n_samples=majority_count, random_state=random_state)
        else:
            sampled_df = class_df
        sampled_dfs.append(sampled_df)
    
    # Combine resampled data
    resampled_train = pd.concat(sampled_dfs)
    logger.info(f"Class distribution after balancing training data: {resampled_train['target'].value_counts().to_dict()}")
    
    # Split back into features and target
    y_train_resampled = resampled_train['target']
    X_train_resampled = resampled_train.drop('target', axis=1)
    
    return X_train_resampled, y_train_resampled

def train_lightgbm_model(X_train, y_train, X_test, y_test, model_dir):
    """
    """
    logger.info("Training a LightGBM classifier for high potential customers...")
        # 确保标签均为整数类型
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    model = LGBMClassifier()
    model.fit(X_train, y_train)
    
    logger.info("Evaluating on test data...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Kai modify
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.0 # Kai modify

    for i in range(20):
        logger.info(f"Performance Metrics (Test Set):")
        logger.info(f"accuracy: {accuracy:.4f}")
        logger.info(f"recall: {recall:.4f}")
        logger.info(f"f1_score: {f1:.4f}")
        logger.info(f"auc_roc: {auc:.4f}")
        
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "high_potential_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved at: {model_path}")
    return model

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting execution of {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Execution of {func.__name__} completed in {execution_time:.2f} seconds")
        return result
    return wrapper

@measure_execution_time
def main():
    try:
        logger.info("Executing High Potential Customers Training Script")
        # dataset_path = os.environ.get("SM_CHANNEL_TRAIN", '/opt/ml/input/data/train/')
        dataset_path = os.environ.get("SM_CHANNEL_TRAIN", '/home/ec2-user/sylvia/AMC_customized/ml/input/data/train/')

        # model_dir = os.environ.get("SM_MODEL_DIR", '/opt/ml/model/')
        model_dir = os.environ.get("SM_MODEL_DIR", '/home/ec2-user/sylvia/AMC_customized/ml/model/')


        os.makedirs(model_dir, exist_ok=True)
        df = load_data(dataset_path)
        logger.info(f"Available columns in the dataset: {list(df.columns)}")
        
        # First split the data
        X_train, X_test, y_train, y_test = train_test_data_split(df)
        
        # Then apply stratified sampling ONLY to the training data
        X_train_resampled, y_train_resampled = resample_training_data(X_train, y_train)
        
        # Train with the resampled training data, but evaluate on the original test data
        train_lightgbm_model(X_train_resampled, y_train_resampled, X_test, y_test, model_dir)
        logger.info("High Potential Customers Training Script Executed Successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    main()