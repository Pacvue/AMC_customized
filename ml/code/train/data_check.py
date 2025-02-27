import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import logging

# ============ Logging Setup ============
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s :: %(levelname)-8s :: %(message)s'
)
logger = logging.getLogger(__name__)

# ============ 1) 数据检查函数 ============

def data_checks(df: pd.DataFrame, label_col: str, drop_cols: list = None):
    """
    对输入数据进行一系列质量检查，包括：
      1. 打印基本信息 (行列数、缺失值、重复行数)
      2. 检查标签分布
      3. 检查潜在的数据泄露：特征列中是否包含标签列本身或等价列
      4. 计算特征与标签的相关性，排查完美相关的列
    参数:
      df (pd.DataFrame): 输入数据
      label_col (str): 标签列列名
      drop_cols (list): 需要提前丢弃的无用列或泄露列（若已知）
    """
    if drop_cols is None:
        drop_cols = []

    logger.info("==== [Data Checks] ====")
    
    # 1) 基本信息
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Data columns: {list(df.columns)}")
    
    # 丢弃无用列或已知的泄露列
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
            logger.info(f"Dropped known leakage column: {c}")
    
    # 打印 info() 以查看列类型和缺失情况
    logger.info("DataFrame info():")
    df.info(verbose=True)

    # 缺失值统计
    missing_counts = df.isnull().sum()
    logger.info(f"Missing values per column:\n{missing_counts}")

    # 重复行数
    duplicates = df.duplicated().sum()
    logger.info(f"Number of duplicated rows: {duplicates}")

    # 2) 标签分布
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame!")
    
    label_counts = df[label_col].value_counts()
    logger.info(f"Label distribution:\n{label_counts}\n")

    # 3) 检查潜在的数据泄露
    # 如果 label_col 出现在特征列，说明泄露
    # 或者如果有任何列和 label_col 完全相同，也需要警惕
    # 这里仅做简单示例，可以更深入检查
    for col in df.columns:
        if col == label_col:
            continue
        if df[col].equals(df[label_col]):
            logger.warning(f"Possible data leakage: column '{col}' is identical to label '{label_col}'!")
    
    # 4) 相关性检查
    # 仅对数值列进行简单相关性分析
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    
    if len(numeric_cols) > 0:
        logger.info("Computing correlation with label for numeric columns...")
        corr_with_label = df[numeric_cols].corrwith(df[label_col]).abs().sort_values(ascending=False)
        logger.info(f"Correlation with label:\n{corr_with_label}")
        # 如果某些列的相关性=1.0，要警惕泄露
        perfect_corr_cols = corr_with_label[corr_with_label == 1.0].index.tolist()
        if perfect_corr_cols:
            logger.warning(f"These columns have perfect correlation (1.0) with label: {perfect_corr_cols}")
    else:
        logger.info("No numeric columns to compute correlation with label.")


# ============ 2) 训练 & 评估流程检查 ============

def train_evaluate(df: pd.DataFrame, label_col: str, test_size=0.2, random_state=42):
    """
    使用 RandomForestClassifier 对数据进行训练并评估，包含：
      1. train_test_split 严格区分训练/测试
      2. 打印训练/测试集形状及标签分布
      3. 进行 K 折交叉验证
      4. 最终在测试集上评估并打印指标
    """
    logger.info("==== [Training/Evaluation Checks] ====")

    # 分离特征和标签
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # 1) 拆分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # 确保拆分后分布一致
    )

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    logger.info(f"Train label distribution:\n{y_train.value_counts()}")
    logger.info(f"Test label distribution:\n{y_test.value_counts()}")

    # 2) 交叉验证
    logger.info("Performing 5-Fold Cross Validation on training set...")
    model_cv = RandomForestClassifier(random_state=random_state)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(model_cv, X_train, y_train, cv=skf, scoring='f1')
    logger.info(f"CV F1 scores: {scores}")
    logger.info(f"Mean F1 from CV: {np.mean(scores):.4f}")

    # 3) 最终模型训练
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # 4) 测试集评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 对二分类，若有两个类别才计算 ROC AUC
    if len(y_test.unique()) > 1:
        auc = roc_auc_score(y_test, y_pred)
    else:
        auc = 0.0

    logger.info(f"==== [Final Evaluation on Test Set] ====")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"ROC AUC:   {auc:.4f}")

    # 如果所有指标都非常完美(=1.0)，提醒可能存在数据泄露或数据异常
    if all(metric == 1.0 for metric in [acc, prec, rec, f1]) and auc in [0.0, 1.0]:
        logger.warning("All metrics are perfect (1.0) on the test set. Possible data leakage or trivial data.")


# ============ 示例主函数调用 ============

def main():
    # 假设你有一个 CSV 文件 "sample_data.csv"，其中包含标签列 "high_potential"
    data_path = "/home/ec2-user/sylvia/HighPotentialCustomers/ml/input/data/train/sandbox_query_results_008da601-3002-45dd-867e-7c1c6fb01415_last90days.csv"
    
    label_column = "high_potential"

    # 读取数据
    df_raw = pd.read_csv(data_path)

    # 第一步：进行数据检查
    # 假设在生成 high_potential 标签时使用了 "total_conversions_30d" 这个列
    # 就要在这里把它从特征中剔除，避免泄露
    potential_leakage_cols = ["total_conversions_30d", "total_revenue_30d","total_quantity_30d","total_conversions_15d","total_revenue_15d","total_conversions_3d","total_revenue_3d","total_quantity_3d"]  
    data_checks(df_raw, label_column, drop_cols=potential_leakage_cols)

    # 第二步：训练 & 评估流程检查
    # 注意：此时 data_checks 里仅做检查，没有真正修改 df_raw。
    # 如果确认 "total_conversions_30d" 等列是泄露列，需要从 df_raw 中剔除后再传入 train_evaluate
    df = df_raw.drop(columns=potential_leakage_cols, errors='ignore')
    train_evaluate(df, label_column, test_size=0.2, random_state=42)


if __name__ == "__main__":
    main()
