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
from sklearn.utils import resample  # 用于随机过采样

# ================= Logging Setup =================
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
logFormatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(logFormatter)
root_logger.addHandler(stdout_handler)

# Set log directory from environment or default
log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/output/data/log/')
os.makedirs(log_path, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_path, "logfile.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logFormatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ================== Data Preprocessing Functions ==================

def data_preparation(data) -> pd.DataFrame:
    """
    数据预处理函数，去掉 Z-Score 过滤，最大程度保留数据。
    包含：
      1. 读取CSV（如果传入的是路径）
      2. 时间列转换与差值计算
      3. 缺失值填充
      4. 字符型列的Label Encoding
      5. 删除无用列
      6. One-Hot编码
      7. 过滤不合理行
      8. 去重
    """
    # ========== 1) 读取数据 ==========
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input data must be a file path or a DataFrame")
        
    logger.info("=== Initial DataFrame Info ===")
    logger.info(df.info())

    # # ========== 2) 时间列转换与差值计算 ==========
    # date_cols = ["last_event_dt_30d", "last_conversion_dt_30d"]
    # for dcol in date_cols:
    #     if dcol in df.columns:
    #         df[dcol] = pd.to_datetime(df[dcol], errors='coerce')
    # if set(date_cols).issubset(df.columns):
    #     df['time_diff_days'] = (df['last_event_dt_30d'] - df['last_conversion_dt_30d']).dt.days
    #     df = df.drop(columns=date_cols)
        
    # logger.info("\n=== Missing Values per Column ===")
    # logger.info(df.isnull().sum())

    # # ========== 3) 缺失值填充 ==========
    # numeric_fill = {
    #     "total_conversions_after_30d": 0,
    #     "total_revenue_30d": 0
    # }
    # for col_n, fill_val in numeric_fill.items():
    #     if col_n in df.columns:
    #         df[col_n] = df[col_n].fillna(fill_val)
    # str_fill = ["customer_search_term"]
    # for scol in str_fill:
    #     if scol in df.columns:
    #         df[scol] = df[scol].fillna("Unknown")

    # ========== 4) Label Encoding ==========
    if "customer_search_term" in df.columns:
        df['customer_search_term'] = df['customer_search_term'].astype(str)
        le = LabelEncoder()
        df['customer_search_term'] = le.fit_transform(df['customer_search_term'])

    # ========== 5) 删除无用列 ==========
    cols_to_drop = ["user_id", "user_id_type"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # ========== 6) One-Hot 编码 ==========
    cat_cols = ["device_type", "browser_family", "operating_system"]
    valid_cat_cols = [c for c in cat_cols if c in df.columns]
    if valid_cat_cols:
        df = pd.get_dummies(df, columns=valid_cat_cols, drop_first=True)

    # ========== 7) 去掉 Z-Score 过滤 (注释掉原逻辑) ==========
    # numeric_cols = [...]
    # df = calculate_zscore_and_filter(df, numeric_cols, threshold)

    # ========== 8) 过滤不合理行 ==========
    if "total_revenue_30d" in df.columns:
        df = df[df["total_revenue_30d"] >= 0]

    # ========== 9) 去重 ==========
    if "customer_search_term" in df.columns and "time_diff_days" in df.columns:
        df = df.drop_duplicates(subset=["customer_search_term", "time_diff_days"])

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

def oversample_minority_class(X: pd.DataFrame, y: pd.Series, random_state=42):
    """
    对少数类进行随机过采样，使正负样本更加平衡。
    """
    logger.info("Performing random oversampling on the training set to handle imbalance...")
    train_df = X.copy()
    train_df['label'] = y

    # 统计当前分布
    pos_count = sum(train_df['label'] == 1)
    neg_count = sum(train_df['label'] == 0)
    logger.info(f"Before oversampling: Positive={pos_count}, Negative={neg_count}")

    # 分割多数类和少数类（假设 high_potential=1 为多数类，0 为少数类；若相反可自行调整）
    majority_class = 1 if pos_count > neg_count else 0
    minority_class = 0 if majority_class == 1 else 1

    df_majority = train_df[train_df['label'] == majority_class]
    df_minority = train_df[train_df['label'] == minority_class]

    # 过采样 minority 类到与 majority 类同样数量
    df_minority_oversampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=random_state
    )

    # 合并回去
    df_oversampled = pd.concat([df_majority, df_minority_oversampled], axis=0)

    # 查看过采样后分布
    pos_count_os = sum(df_oversampled['label'] == 1)
    neg_count_os = sum(df_oversampled['label'] == 0)
    logger.info(f"After oversampling:  Positive={pos_count_os}, Negative={neg_count_os}")

    # 分离特征和标签
    y_resampled = df_oversampled['label']
    X_resampled = df_oversampled.drop(columns=['label'])

    return X_resampled, y_resampled

def train_test_data_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    拆分训练集与测试集，并在训练集上进行过采样（随机复制少数类）。
    """
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}...")
    if 'high_potential' not in df.columns:
        raise ValueError("Column 'high_potential' not found. Please compute it before splitting.")
    
    # 特征中去掉直接用于生成 high_potential 的列
    # features_to_drop = ['high_potential', 'total_conversions_30d', 'total_revenue_30d']
    features_to_drop = ["total_conversions_after_30d", "total_revenue_after_30d","total_quantity_after_30d","last_event_dt_before"]  
    X = df.drop(columns=[col for col in features_to_drop if col in df.columns], errors='ignore')
    y = df['high_potential']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # 尽量保证拆分后分布一致
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 在训练集上执行随机过采样
    X_train_os, y_train_os = oversample_minority_class(X_train, y_train, random_state=random_state)

    return X_train_os, X_test, y_train_os, y_test

def cross_validate_lightgbm(X, y, n_splits=5, random_state=42):
    """
    使用 K 折交叉验证评估模型性能，并打印各折指标。
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
        
        # 这里可选择是否对每折做 oversampling。若做，则会在 CV 中也平衡数据
        # X_train_fold, y_train_fold = oversample_minority_class(X_train_fold, y_train_fold, random_state)
        
        model = LGBMClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
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
            f"[Fold {fold_idx}] Accuracy={acc:.4f}, Precision={prec:.4f}, "
            f"Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
        )
    
    logger.info("=== K-Fold Cross Validation (Average) ===")
    logger.info(f"Mean Accuracy: {np.mean(metrics['accuracy']):.4f}")
    logger.info(f"Mean Precision: {np.mean(metrics['precision']):.4f}")
    logger.info(f"Mean Recall: {np.mean(metrics['recall']):.4f}")
    logger.info(f"Mean F1 Score: {np.mean(metrics['f1']):.4f}")
    logger.info(f"Mean ROC AUC: {np.mean(metrics['auc']):.4f}")

def train_lightgbm_model(X_train, y_train, X_test, y_test, model_dir):
    """
    使用 LightGBM 训练并在测试集上评估，最后保存模型。
    启用 class_weight='balanced' 以减轻数据不平衡。
    """
    logger.info("Training a LightGBM classifier for high potential customers...")
    model = LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced')
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
        
        # ========== 1) 读取输入数据 ==========
        input_dir = "/home/ec2-user/sylvia/HighPotentialCustomers/ml/input/data/train"
        file_list = glob.glob(os.path.join(input_dir, "*.csv"))
        if not file_list:
            raise ValueError(f"No CSV files found in {input_dir}")
        
        raw_df = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)
        logger.info(f"Read {len(file_list)} files with a total of {raw_df.shape[0]} rows.")
        
        # ========== 2) 数据预处理 (已去掉 z-score) ==========
        prepared_df = data_preparation(raw_df)
        
        # ========== 3) 计算目标标签 high_potential ==========
        df = calculate_high_potential_flag(prepared_df)
        
        # ========== 4) 训练/测试集拆分 (含过采样) ==========
        X_train, X_test, y_train, y_test = train_test_data_split(df)
        
        # ========== 5) K-Fold 交叉验证 (可选：每折也可过采样) ==========
        # cross_validate_lightgbm(X_train, y_train, n_splits=5, random_state=42)
        
        # ========== 6) 最终模型训练 & 测试集评估 & 保存模型 ==========
        model_dir = os.environ.get("SM_MODEL_DIR", "/home/ec2-user/sylvia/HighPotentialCustomers/ml/model")
        model = train_lightgbm_model(X_train, y_train, X_test, y_test, model_dir)
        
        # ========== 7) 用最终模型对测试集预测并输出 ==========
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        output_df = X_test.copy()
        output_df['predicted_high_potential'] = y_pred
        output_df['predicted_probability'] = y_pred_proba
        
        # 指定输出结果的目录
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
