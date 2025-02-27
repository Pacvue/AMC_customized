import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List

def calculate_zscore_and_filter(df: pd.DataFrame,
                                numeric_cols: List[str],
                                threshold: float = 3.0) -> pd.DataFrame:
    """
    对指定数值列进行 Z-Score 异常值过滤：
    1) 分别计算各列的 mean、std
    2) 将 |Z| > threshold 的行过滤掉 (当作 outlier)
       Z = (value - mean) / std

    :param df: 待处理的 DataFrame
    :param numeric_cols: 要检查异常值的数值列名列表
    :param threshold: 超过此阈值(|Z|)的将被视为异常
    :return: 过滤后的 DataFrame
    """
    for col_name in numeric_cols:
        if col_name not in df.columns:
            continue

        col_mean = df[col_name].mean()
        col_std = df[col_name].std()

        if col_std is not None and col_std > 0:
            zscore = (df[col_name] - col_mean) / col_std
            # 过滤掉 |zscore| >= threshold 的行
            df = df.loc[zscore.abs() < threshold]
        else:
            # 如果 std=0，说明该列要么缺失值多要么都是同一个值，无需过滤
            pass

    return df

def data_preparation(input_path: str,
                     threshold: float = 3.0) -> pd.DataFrame:
    """
    数据预处理函数，包含以下步骤：
    1. 读取 CSV 文件
    2. 将日期字段转为 datetime，并计算两日期之间的时间差（单位：天）
    3. 缺失值统计与填充
    4. 对 customer_search_term 列进行 Label Encoding（转换为数值型变量）
    5. 删除不需要的列：user_id, user_id_type
    6. 对其他类别变量进行 One-Hot 编码（如 device_type、browser_family、operating_system）
    7. 用 Z-Score 过滤数值型列中的异常值
    8. 过滤掉不符合业务逻辑的行（例如 total_revenue_30d < 0）
    9. 去重（示例中根据 customer_search_term 与时间差去重，可根据实际情况调整）
    10. 返回预处理后的 DataFrame

    :param input_path: CSV 文件路径
    :param threshold: Z-Score 过滤阈值，默认 3.0
    :return: 清洗并经过特征处理后的 DataFrame
    """
    # ========== 1) 读取 CSV ==========
    df = pd.read_csv(input_path)
    print("=== Initial DataFrame Info ===")
    print(df.info())

    # ========== 2) 将日期字段转为 datetime，并计算时间差 ==========
    date_cols = ["last_event_dt_30d", "last_conversion_dt_30d"]
    for dcol in date_cols:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors='coerce')

    # 若两个日期列均存在，则计算二者的时间差（单位为天），并添加新列 time_diff_days
    if set(date_cols).issubset(df.columns):
        df['time_diff_days'] = (df['last_event_dt_30d'] - df['last_conversion_dt_30d']).dt.days
        # 删除原始日期列
        df = df.drop(columns=date_cols)

    # ========== 3) 缺失值统计与填充 ==========
    print("\n=== Missing Values per Column ===")
    print(df.isnull().sum())

    # 示例：对于数值列，缺失值填充为 0
    numeric_fill = {
        "total_conversions_30d": 0,
        "total_revenue_30d": 0
    }
    for col_n, fill_val in numeric_fill.items():
        if col_n in df.columns:
            df[col_n] = df[col_n].fillna(fill_val)

    # 对字符串列，缺失值填充为 'Unknown'
    str_fill = ["customer_search_term"]  # 如有需要，可扩展更多列
    for scol in str_fill:
        if scol in df.columns:
            df[scol] = df[scol].fillna("Unknown")

    # ========== 4) 对 customer_search_term 列进行 Label Encoding ==========
    if "customer_search_term" in df.columns:
        df['customer_search_term'] = df['customer_search_term'].astype(str)
        le = LabelEncoder()
        df['customer_search_term'] = le.fit_transform(df['customer_search_term'])

    # ========== 5) 删除不需要的列 ==========
    # 删除 user_id 与 user_id_type 列（如果存在）
    cols_to_drop = ["user_id", "user_id_type"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # ========== 6) 对其他类别变量进行 One-Hot 编码 ==========
    # 此处对 device_type、browser_family、operating_system 进行编码
    cat_cols = ["device_type", "browser_family", "operating_system"]
    valid_cat_cols = [c for c in cat_cols if c in df.columns]
    if valid_cat_cols:
        df = pd.get_dummies(df, columns=valid_cat_cols, drop_first=True)

    # ========== 7) Z-Score 过滤异常值 ==========
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

    # ========== 8) 过滤掉不符合业务逻辑的行 ==========
    if "total_revenue_30d" in df.columns:
        df = df[df["total_revenue_30d"] >= 0]

    # ========== 9) 去重操作 ==========
    # 由于已删除 user_id，如需去重可根据其他唯一标识（例如 customer_search_term 与 time_diff_days）进行去重
    if "customer_search_term" in df.columns and "time_diff_days" in df.columns:
        df = df.drop_duplicates(subset=["customer_search_term", "time_diff_days"])

    # 返回预处理后的 DataFrame
    return df

if __name__ == "__main__":
    input_csv = "/home/ec2-user/sylvia/HighPotentialCustomers/ml/input/data/train/sandbox_query_results_008da601-3002-45dd-867e-7c1c6fb01415_last90days.csv"
    output_csv = "/home/ec2-user/sylvia/HighPotentialCustomers/ml/output/data/prepared_data.csv"

    # 读取并预处理数据
    df_clean = data_preparation(input_csv, threshold=3.0)

    print("\n=== Final Cleaned DataFrame Info ===")
    print(df_clean.info())
    print(df_clean.head(5))

    df_clean.to_csv(output_csv, index=False)

# import pandas as pd
# import numpy as np
# from typing import List

# def calculate_zscore_and_filter(df: pd.DataFrame,
#                                 numeric_cols: List[str],
#                                 threshold: float = 3.0) -> pd.DataFrame:
#     """
#     对指定数值列进行 Z-Score 异常值过滤：
#     1) 分别计算各列的 mean、std
#     2) 将 |Z| > threshold 的行过滤掉 (当作 outlier)
#        Z = (value - mean) / std

#     :param df: 待处理的 DataFrame
#     :param numeric_cols: 要检查异常值的数值列名列表
#     :param threshold: 超过此阈值(|Z|)的将被视为异常
#     :return: 过滤后的 DataFrame
#     """
#     for col_name in numeric_cols:
#         if col_name not in df.columns:
#             continue

#         col_mean = df[col_name].mean()
#         col_std = df[col_name].std()

#         if col_std is not None and col_std > 0:
#             zscore = (df[col_name] - col_mean) / col_std
#             # 过滤 |zscore| >= threshold 的行
#             df = df.loc[zscore.abs() < threshold]
#         else:
#             # 如果 std=0，说明该列要么缺失值多要么都是同一个值，无需过滤
#             pass

#     return df


# def data_preparation(input_path: str,
#                      threshold: float = 3.0) -> pd.DataFrame:
#     """
#     数据预处理函数，包含以下步骤：
#     1. 读取 CSV
#     2. 将日期字段转为 datetime
#     3. 缺失值统计与填充
#     4. 对字符串列进行 One-Hot 编码
#     5. 用 Z-Score 过滤数值型列的异常值
#     6. 过滤掉不符合业务逻辑的数值行 (示例: total_revenue_30d >= 0)
#     7. 去重 (示例: user_id + last_event_dt_30d)
#     8. 返回清洗后的 DataFrame

#     :param input_path: CSV 文件路径
#     :param threshold: Z-Score 过滤阈值，默认 3.0
#     :return: 清洗并经过特征处理后的 DataFrame
#     """
#     # ========== 1) 读取 CSV ==========
#     df = pd.read_csv(input_path)
#     print("=== Initial DataFrame Info ===")
#     print(df.info())

#     # ========== 2) 将日期字段转为 datetime ==========    
#     # 按需转换，如果原始日期格式是 'yyyy-MM-dd'，下面的默认解析即可
#     date_cols = ["last_event_dt_30d", "last_conversion_dt_30d"]
#     for dcol in date_cols:
#         if dcol in df.columns:
#             df[dcol] = pd.to_datetime(df[dcol], errors='coerce')

#     # ========== 3) 缺失值统计与填充 ==========
#     print("\n=== Missing Values per Column ===")
#     print(df.isnull().sum())

#     # 示例：数值列缺失时用 0 填充
#     numeric_fill = {
#         "total_conversions_30d": 0,
#         "total_revenue_30d": 0
#     }
#     for col_n, fill_val in numeric_fill.items():
#         if col_n in df.columns:
#             df[col_n] = df[col_n].fillna(fill_val)

#     # 示例：字符串列缺失时用 'Unknown' 填充
#     str_fill = ["customer_search_term"]  # 可根据需要扩展
#     for scol in str_fill:
#         if scol in df.columns:
#             df[scol] = df[scol].fillna("Unknown")

#     # ========== 4) 字符串列的 One-Hot 编码 ==========
#     # 示例中我们对下列列做 One-Hot
#     cat_cols = ["user_id_type", "device_type", "browser_family", "operating_system"]
#     # 仅对存在的列进行编码
#     valid_cat_cols = [c for c in cat_cols if c in df.columns]

#     # 使用 pandas 自带的 get_dummies 做 One-Hot，自动略过 NaN
#     # drop_first=True 可以避免 dummy variable trap
#     df = pd.get_dummies(df, columns=valid_cat_cols, drop_first=True)

#     # ========== 5) Z-Score 过滤异常值 ==========
#     # 选取需要做 outlier 检查的数值列
#     numeric_cols = [
#         "search_campaign_cnt", "total_impressions_30d", "total_clicks_30d",
#         "total_impressions_15d", "total_clicks_15d",
#         "total_impressions_3d", "total_clicks_3d",
#         "total_conversions_30d", "total_revenue_30d", "total_quantity_30d",
#         "total_conversions_15d", "total_revenue_15d", "total_quantity_15d",
#         "total_conversions_3d", "total_revenue_3d", "total_quantity_3d"
#     ]
#     numeric_cols = [c for c in numeric_cols if c in df.columns]

#     df = calculate_zscore_and_filter(df, numeric_cols, threshold)

#     # ========== 6) 过滤掉不符合业务逻辑的数值行 ==========
#     if "total_revenue_30d" in df.columns:
#         df = df[df["total_revenue_30d"] >= 0]

#     # ========== 7) 去重操作 (若 user_id + last_event_dt_30d 不能重复) ==========
#     # drop_duplicates 会保留首次出现的行，并删除后面相同子集行
#     if "user_id" in df.columns and "last_event_dt_30d" in df.columns:
#         df = df.drop_duplicates(subset=["user_id", "last_event_dt_30d"])

#     # 返回处理后的 DataFrame
#     return df


# if __name__ == "__main__":
#     # 示例用法
#     input_csv = "/home/ec2-user/sylvia/HighPotentialCustomers/ml/input/data/train/sandbox_query_results_008da601-3002-45dd-867e-7c1c6fb01415_last90days.csv"
#     output_csv = "/home/ec2-user/sylvia/HighPotentialCustomers/ml/output/data/prepared_data.csv"

#     # 读入并预处理
#     df_clean = data_preparation(input_csv, threshold=3.0)

#     print("\n=== Final Cleaned DataFrame Info ===")
#     print(df_clean.info())
#     print(df_clean.head(5))

#     df_clean.to_csv(output_csv, index=False)
