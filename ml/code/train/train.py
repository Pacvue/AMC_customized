import pandas as pd
import pickle
import glob
import numpy as np
import logging
import sys
import os
import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample  # For random oversampling

# ================= Logging Setup =================
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
logFormatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(logFormatter)
root_logger.addHandler(stdout_handler)

# log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/opt/ml/output/data/log/')
log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/output/data/log/')
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
    logger.info("Reading CSV files...")
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    return df

def get_columns_to_drop():
    # 只排除标识符和目标变量相关的列
    return [
        "high_potential", "user_id", 
        "total_conversions_after_30d", "total_revenue_after_30d", "total_quantity_after_30d"
    ]

def calculate_high_potential_flag(df: pd.DataFrame):
    """
    当后30天的转化、收入或销量任一指标大于0时，标记为高潜力客户
    """
    df['high_potential'] = np.where(
        (df['total_conversions_after_30d'] > 0) |
        (df['total_revenue_after_30d'] > 0) |
        (df['total_quantity_after_30d'] > 0),
        1, 0
    )
    df['high_potential'] = df['high_potential'].fillna(0).astype(int)
    return df

def stratified_sampling(df: pd.DataFrame, label_column: str, random_state=42):
    """
    采用分层抽样对类别不平衡进行过采样
    """
    class_counts = df[label_column].value_counts()
    majority_count = class_counts.max()
    sampled_dfs = []
    for label, count in class_counts.items():
        class_df = df[df[label_column] == label]
        if count < majority_count:
            sampled_df = resample(class_df, replace=True, n_samples=majority_count, random_state=random_state)
        else:
            sampled_df = class_df
        sampled_dfs.append(sampled_df)
    df_balanced = pd.concat(sampled_dfs)
    logger.info(f"Class distribution after balancing: {df_balanced[label_column].value_counts().to_dict()}")
    return df_balanced.reset_index(drop=True)

def train_test_data_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    采用分层抽样划分训练集和测试集，仅排除标识符和目标变量相关的列
    """
    logger.info(f"Splitting data with test_size={test_size}...")
    if 'high_potential' not in df.columns:
        raise ValueError("Column 'high_potential' not found. Please compute it before splitting.")
    columns_to_drop = get_columns_to_drop()
    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    logger.info(f"Using features: {list(X.columns)}")
    logger.info("Handling missing values...")
    logger.info(f"Missing counts: {X.isnull().sum().to_dict()}")
    X = X.fillna(0)
    y = df['high_potential']
    logger.info(f"Shape after handling missing values: {X.shape}")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def data_preparation(data) -> pd.DataFrame:
    """
    数据预处理：读取数据、填充缺失值、去除重复记录（所有特征均为数值型）
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input data must be a file path or a DataFrame")
        
    logger.info(f"Initial DataFrame shape: {df.shape}")
    df = df.fillna(0)
    df = df.drop_duplicates()
    logger.info(f"DataFrame shape after preparation: {df.shape}")
    return df

def grid_search_lightgbm(X_train, y_train):
    """
    使用 GridSearchCV 寻找最佳 LightGBM 参数
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }
    model = LGBMClassifier(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def train_lightgbm_model(X_train, y_train, X_test, y_test, model_dir):
    """
    使用最佳参数训练 LightGBM 模型并在测试集上评估
    """
    logger.info("Training LightGBM classifier for high potential customers...")
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    model = LGBMClassifier(n_estimators=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    for _ in range(10):
        for idx, row in feature_importance.iterrows():
            logger.info(f"feature_idx: {idx+1}, feature_imp: {row['importance']:.4f}")
    
    logger.info("Evaluating on test data...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.0
    pred_dist = pd.Series(y_pred).value_counts()
    proba_stats = {
        'mean': np.mean(y_pred_proba),
        'std': np.std(y_pred_proba),
        'min': np.min(y_pred_proba),
        'max': np.max(y_pred_proba),
        'median': np.median(y_pred_proba)
    }
    for _ in range(10):
        logger.info(f"accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1_score: {f1:.4f}, auc: {auc:.4f}")
        logger.info(f"prob_mean: {proba_stats['mean']:.4f}, prob_std: {proba_stats['std']:.4f}, prob_min: {proba_stats['min']:.4f}, prob_max: {proba_stats['max']:.4f}, prob_median: {proba_stats['median']:.4f}")
        logger.info(f"class_0_count: {int(pred_dist.get(0, 0))}, class_1_count: {int(pred_dist.get(1, 0))}")
    
    model_path = os.path.join(model_dir, "high_potential_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved at: {model_path}")
    return model

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting execution of {func.__name__}...")
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"Execution of {func.__name__} completed in {execution_time:.2f} seconds")
        return result
    return wrapper

@measure_execution_time
def main():
    try:
        logger.info("Executing High Potential Customers Training Script")
        # dataset_path = os.environ.get("SM_CHANNEL_TRAIN", '/opt/ml/input/data/train/')
        dataset_path = os.environ.get("SM_CHANNEL_TRAIN", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/input/data/train/')
        # model_dir = os.environ.get("SM_MODEL_DIR", '/opt/ml/model/')
        model_dir = os.environ.get("SM_MODEL_DIR", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/model/')

        os.makedirs(model_dir, exist_ok=True)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
            
        df = load_data(dataset_path)
        if df is None or df.empty:
            raise ValueError("Data loading failed or data is empty")
            
        logger.info(f"Available columns in the dataset: {list(df.columns)}")
        logger.info(f"Dataset shape before preparation: {df.shape}")
        
        df = data_preparation(df)
        logger.info(f"Available columns after preparation: {list(df.columns)}")
        logger.info(f"Dataset shape after preparation: {df.shape}")
        
        required_cols = ['total_conversions_after_30d', 'total_revenue_after_30d', 'total_quantity_after_30d']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = calculate_high_potential_flag(df)
        df = stratified_sampling(df, "high_potential")
        
        X_train, X_test, y_train, y_test = train_test_data_split(df)
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        if X_train.empty or X_test.empty:
            raise ValueError("Training or test dataset is empty after splitting")
        
        # 使用 GridSearchCV 寻找最佳模型参数
        best_model = grid_search_lightgbm(X_train, y_train)
        
        # 用最佳参数训练最终模型
        train_lightgbm_model(X_train, y_train, X_test, y_test, model_dir)
        
        logger.info("High Potential Customers Training Script Executed Successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
