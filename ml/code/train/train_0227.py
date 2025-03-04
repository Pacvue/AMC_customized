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

root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
logFormatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(logFormatter)
root_logger.addHandler(stdout_handler)

# Set log directory from environment or default
# log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/home/ec2-user/sylvia/AMC_customized/ml/output/data/log/')
log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/opt/ml/output/data/log/')
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

def calculate_high_potential_flag(df: pd.DataFrame):
    """
    Compute the 'high_potential' flag based on the SQL-derived columns.
    A user is flagged as high potential if any of the conversion-related metrics (after 30d)
    is greater than 0.
    """
    df['high_potential'] = np.where(
        (
        (df['total_conversions_after_30d'] > 0) |
        (df['total_revenue_after_30d'] > 0) |
        (df['total_quantity_after_30d'] > 0))
    , 1, 0)
    df['high_potential'] = df['high_potential'].fillna(0).astype(int)
    return df

# def stratified_sampling(df, label_column, target_count=10000, random_state=42):
#     """
#     对 DataFrame 根据 label_column 列进行分层采样，
#     每个类别样本数达到 target_count（不足则超采样，多余则负采样）。
#     """
#     sampled_list = []
#     for label in df[label_column].unique():
#         df_label = df[df[label_column] == label]
#         current_count = len(df_label)
#         if current_count < target_count:
#             # 超采样：使用 replace=True
#             sampled = df_label.sample(n=target_count, replace=True, random_state=random_state)
#             print(f"Label {label}: oversampled from {current_count} to {target_count} samples.")
#         elif current_count > target_count:
#             # 负采样：使用 replace=False
#             sampled = df_label.sample(n=target_count, replace=False, random_state=random_state)
#             print(f"Label {label}: undersampled from {current_count} to {target_count} samples.")
#         else:
#             sampled = df_label.copy()
#             print(f"Label {label}: sample count is exactly {target_count}.")
#         sampled_list.append(sampled)
    
#     return pd.concat(sampled_list).reset_index(drop=True)

def train_test_data_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Split the dataset into training and test sets using stratified sampling.
    """
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}...")
    if 'high_potential' not in df.columns:
        raise ValueError("Column 'high_potential' not found. Please compute it before splitting.")

    drop_col = ['user_id','total_conversions_after_30d','total_revenue_after_30d','total_quantity_after_30d']
    df = df.drop(columns=drop_col, errors='ignore').select_dtypes(include=['number']).fillna(0).copy()  # 选择数据类型为数字的列
    y = df['high_potential']
    # X = df.drop(columns=['high_potential'], errors='ignore')[['total_clicks_15d','total_impressions_15d']]
    X = df.drop(columns=['high_potential'], errors='ignore')


    X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # 尽量保证拆分后分布一致
        )
    return X_train, X_test, y_train, y_test


def train_lightgbm_model(X_train, y_train, X_test, y_test, model_dir):
    """
    启用 class_weight='balanced' 以减轻数据不平衡。
    Train a LightGBM classifier and evaluate it on the test set.
    Save the trained model to the specified model_dir.
    """
    logger.info("Training a LightGBM classifier for high potential customers...")
        # 确保标签均为整数类型
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    model = LGBMClassifier(n_estimators=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    logger.info("Evaluating on test data...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Kai modify
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.0 # Kai modify
    # pos_ratio = y_train.sum()/len(y_train)
    # y_train_shape = len(y_train)

    for i in range(10):
        logger.info(f"Performance Metrics (Test Set):")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"ROC AUC: {auc:.4f}")
        # logger.info(f"y_train_shape: {y_train_shape:.4f}")
        # logger.info(f"pos_ratio: {pos_ratio:.4f}")
        
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
        # dataset_path = os.environ.get("SM_CHANNEL_TRAIN", '/home/ec2-user/sylvia/AMC_customized/ml/input/data/train/')
        dataset_path = os.environ.get("SM_CHANNEL_TRAIN", '/opt/ml/input/data/train/')

        # model_dir = os.environ.get("SM_MODEL_DIR", '/home/ec2-user/sylvia/AMC_customized/ml/model/')
        model_dir = os.environ.get("SM_MODEL_DIR", '/opt/ml/model/')

        os.makedirs(model_dir, exist_ok=True)
        df = load_data(dataset_path)
        logger.info(f"Available columns in the dataset: {list(df.columns)}")
        
        df = calculate_high_potential_flag(df)
        # df = stratified_sampling(df,"high_potential")
        X_train, X_test, y_train, y_test = train_test_data_split(df)
        train_lightgbm_model(X_train, y_train, X_test, y_test, model_dir)
        logger.info("High Potential Customers Training Script Executed Successfully.")
    except Exception as e:
        
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()



