import pandas as pd
import pickle
import glob
import numpy as np
# ================= Logging Setup =================
import logging
import sys
import os
import time
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
logFormatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(logFormatter)
root_logger.addHandler(stdout_handler)

# Set log directory from environment or default
log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/output/data/log/')
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


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data cleaning...")
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    df[non_numeric_cols] = df[non_numeric_cols].fillna("")
    logger.info("Data cleaning completed.")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering...")
    df['click_rate'] = np.where(df['total_impressions_30d'] > 0,
                                df['total_clicks_30d'] / df['total_impressions_30d'], 0)
    logger.info("Feature engineering completed.")
    return df

def feature_selection(df: pd.DataFrame, target_column: str, top_n) -> list:

    logger.info("Selecting top features using a pre-trained XGBoost model...")
    feature_cols = [col for col in df.columns if col != target_column]
    X = df[feature_cols].copy()
    y = df[target_column]
    X = X.apply(pd.to_numeric, errors = "coerce").fillna(0)

    model = xgb.XGBClassifier(n_estimators  =10, random_state = 42, use_label_encoder = False, eval_metric = 'logloss')
    model.fit(X,y)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"feature": feature_cols, 'importance':importances})
    importance_df = importance_df.sort_values(by="importance", ascending = False)
    top_features = importance_df['feature'].head(top_n).tolist()
    logger.info(f"Top {top_n} features selected: {top_features}")
    return top_features

def cross_validation(X,y):
    logger.info("Starting hyperparameter tuning with GridSearchCV...")\
    param_grid = {
        'n_estimators':[10,50,100],
        'learning_rate':[0.01, 0.1, 0.2],
        'max_depth':[3,5,7]
    }
    lgbm = LGBMClassifier(random_state = 42, class_weight = 'balanced')
    cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
    grid_search = GridSearchCV(estimator = lgbm, param_grid = param_grid, cv = cv, scoring='roc_auc', n_jobs = -1)
    grid_search.fit(X,y)
    logger.info(f:"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


features_to_drop = ["user_id","user_id_type"]
features_to_drop = ["user_id","user_id_type","customer_search_term","dma_code","postal_code","device_type","browser_family","operating_system","total_conversions_after_30d","total_revenue_after_30d","total_quantity_after_30d","total_conversions_before_30d","total_revenue_before_30d","total_quantity_before_30d", "total_conversions_before_15d","total_revenue_before_15d","total_quantity_before_15d", "total_conversions_before_5d","total_revenue_before_5d","total_quantity_before_5d","last_event_dt_before"]  
# X['last_event_dt_before'] = pd.to_datetime(X['last_event_dt_before'], errors='coerce')
# X['last_conversion_dt_30d'] = pd.to_datetime(X['last_conversion_dt_30d'], errors='coerce')
# X['time_diff_days'] = (X['last_event_dt_30d'] - X['last_conversion_dt_30d']).dt.days
# 3. 删除原始日期列
# X.drop(columns=['last_event_dt_before', 'last_conversion_dt_30d'], inplace=True)

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

def stratified_sampling(df, label_column, target_count=10000, random_state=42):
    """
    对 DataFrame 根据 label_column 列进行分层采样，
    每个类别样本数达到 target_count（不足则超采样，多余则负采样）。
    """
    sampled_list = []
    for label in df[label_column].unique():
        df_label = df[df[label_column] == label]
        current_count = len(df_label)
        if current_count < target_count:
            # 超采样：使用 replace=True
            sampled = df_label.sample(n=target_count, replace=True, random_state=random_state)
            print(f"Label {label}: oversampled from {current_count} to {target_count} samples.")
        elif current_count > target_count:
            # 负采样：使用 replace=False
            sampled = df_label.sample(n=target_count, replace=False, random_state=random_state)
            print(f"Label {label}: undersampled from {current_count} to {target_count} samples.")
        else:
            sampled = df_label.copy()
            print(f"Label {label}: sample count is exactly {target_count}.")
        sampled_list.append(sampled)
    
    return pd.concat(sampled_list).reset_index(drop=True)

def train_test_data_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Split the dataset into training and test sets using stratified sampling.
    """
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}...")
    if 'high_potential' not in df.columns:
        raise ValueError("Column 'high_potential' not found. Please compute it before splitting.")
    
    # Drop the target column from features
    features_to_drop = ["high_potential","user_id","user_id_type","customer_search_term","dma_code","postal_code","device_type","browser_family","operating_system","total_conversions_after_30d","total_revenue_after_30d","total_quantity_after_30d","last_event_dt_before"]  
    X = df.drop(columns=[col for col in features_to_drop if col in df.columns], errors='ignore')
    # df = df.dropna(subset=["total_impressions_30d", "total_clicks_30d"], axis=0)
    # df.fillna({'total_impressions_30d': 0, 'total_clicks_30d': 0, 'high_potential': 0}, inplace=True)
    # X = df[["total_impressions_30d","total_clicks_30d"]]
    y = df['high_potential']  

    X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
            # stratify=y  # 尽量保证拆分后分布一致
        )
    return X_train, X_test, y_train, y_test


def data_preparation(df: pd.DataFrame):
    df 
    label_encoder 




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

    for i in range(10):
        logger.info(f"Performance Metrics (Test Set):")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"ROC AUC: {auc:.4f}")
    
    # os.makedirs(model_dir, exist_ok=True)
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
        dataset_path = os.environ.get("SM_CHANNEL_TRAIN", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/input/data/train/')
        # dataset_path = os.environ.get("SM_CHANNEL_TRAIN", '/opt/ml/input/data/train/')

        model_dir = os.environ.get("SM_MODEL_DIR", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/model/')
        # model_dir = os.environ.get("SM_MODEL_DIR", '/opt/ml/model/')

        os.makedirs(model_dir, exist_ok=True)

        #1. Data Cleaning
        df = load_data(dataset_path)
        logger.info(f"Available columns in the dataset: {list(df.columns)}")
        df = data_cleaning(df)

        #2. Feature Engineering
        # df = feature_engineering(df)

        #3. high_potential
        df = calculate_high_potential_flag(df)

        #feature selection
        # top_features = select_top_features(df, target_column = "high_potential", top_n = 15)
        # logger.info(f"Selected top features: {top_features}")

        #4. stratified sampling
        df = stratified_sampling(df,"high_potential")

        #5. train test split
        X_train, X_test, y_train, y_test = train_test_data_split(df)

        #6. k-fold cross validation and grid search to choose hyperparameters
        # X_train_numeric = X_train[top_features].astype(float)
        # y_train_numeric = y_train.astype(int)
        # best_model = 

        #7. Use the optimized model to train and save model
        train_lightgbm_model(X_train, y_train, X_test, y_test, model_dir)
        logger.info("High Potential Customers Training Script Executed Successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()



