# from pyspark.sql.functions import datediff, lit

# # 3.1 计算与某个参考日期的天数差
# # 假设我们把 '2025-01-21' 作为参考日期
# ref_date = "2025-01-21"

# df_fe = (df_clean
#     .withColumn("days_since_last_event", datediff(lit(ref_date), col("last_event_dt_30d")))
#     .withColumn("days_since_last_conversion", datediff(lit(ref_date), col("last_conversion_dt_30d")))
# )

# # 3.2 构造点击转化率和客单价
# from pyspark.sql.functions import when

# df_fe = (df_fe
#     .withColumn(
#         "ctr_30d", 
#         when(col("total_clicks_30d") > 0, col("total_conversions_30d") / col("total_clicks_30d")).otherwise(0)
#     )
#     .withColumn(
#         "avg_order_value_30d",
#         when(col("total_quantity_30d") > 0, col("total_revenue_30d") / col("total_quantity_30d")).otherwise(0)
#     )
# )

# df_fe.printSchema()
# df_fe.show(5)





# #feature selection
# from pyspark.ml.stat import Correlation
# from pyspark.ml.feature import VectorAssembler

# # 假设选择一部分数值列做相关性分析
# numeric_cols = [
#     "search_campaign_cnt", "total_impressions_30d", "total_clicks_30d", 
#     "total_conversions_30d", "total_revenue_30d", "total_quantity_30d",
#     "ctr_30d", "avg_order_value_30d",
#     "days_since_last_event", "days_since_last_conversion"
# ]
# assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features_vector")
# df_vector = assembler.transform(df_fe).select("numeric_features_vector")

# # 计算相关系数矩阵（Pearson）
# pearson_corr = Correlation.corr(df_vector, "numeric_features_vector", "pearson").head()[0]
# logger.info(f"Pearson correlation matrix:\n{pearson_corr}")



# output_path = "/path/to/prepared_data"
# df_fe.write.mode("overwrite").parquet(output_path)

# spark.stop()



import pandas as pd
import numpy as np
from typing import List, Optional

def feature_engineering(
    df_prepared: pd.DataFrame,
    ref_date_str: str = "2025-01-21",
    numeric_cols: Optional[List[str]] = None,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    对从 data_preparation.py 脚本中输出的 DataFrame 进行进一步特征工程和分析：
    
    1. 计算参考日期与各事件日期的天数差(days_since_...)。
    2. 构造点击转化率(ctr_30d)和客单价(avg_order_value_30d)。
    3. 对指定数值列计算 Pearson 相关系数矩阵并打印。
    4. 可选：写出处理后的 DataFrame 到 Parquet 文件(若传入 output_parquet 参数)。

    参数：
    :param df_prepared: 已经过 data_preparation.py 清洗好的 DataFrame
    :param ref_date_str: 参考日期的字符串(默认 "2025-01-21")
    :param numeric_cols: 需要做相关性分析的数值列列表，如为 None 则使用内置示例
    :param output_parquet: 如果指定了文件路径，将输出处理后 DataFrame 为 Parquet
    :return: 带新特征的 DataFrame
    """

    # === 1) 计算与参考日期的天数差 ===
    ref_date = pd.to_datetime(ref_date_str, errors='coerce')

    if "last_event_dt_30d" in df_prepared.columns:
        df_prepared["days_since_last_event"] = (ref_date - df_prepared["last_event_dt_30d"]).dt.days

    if "last_conversion_dt_30d" in df_prepared.columns:
        df_prepared["days_since_last_conversion"] = (ref_date - df_prepared["last_conversion_dt_30d"]).dt.days

    # === 2) 构造点击转化率 (ctr_30d) 和客单价 (avg_order_value_30d) ===
    if "total_clicks_30d" in df_prepared.columns and "total_conversions_30d" in df_prepared.columns:
        df_prepared["ctr_30d"] = np.where(
            df_prepared["total_clicks_30d"] > 0,
            df_prepared["total_conversions_30d"] / df_prepared["total_clicks_30d"],
            0
        )

    if "total_quantity_30d" in df_prepared.columns and "total_revenue_30d" in df_prepared.columns:
        df_prepared["avg_order_value_30d"] = np.where(
            df_prepared["total_quantity_30d"] > 0,
            df_prepared["total_revenue_30d"] / df_prepared["total_quantity_30d"],
            0
        )

    # === 3) 计算数值列的 Pearson 相关系数矩阵 ===
    # 如果用户没传 numeric_cols，就用下面的默认示例
    if numeric_cols is None:
        numeric_cols = [
            "search_campaign_cnt", "total_impressions_30d", "total_clicks_30d",
            "total_conversions_30d", "total_revenue_30d", "total_quantity_30d",
            "ctr_30d", "avg_order_value_30d",
            "days_since_last_event", "days_since_last_conversion"
        ]
    valid_numeric_cols = [c for c in numeric_cols if c in df_prepared.columns]

    if valid_numeric_cols:
        corr_matrix = df_prepared[valid_numeric_cols].corr(method='pearson')
        print("\n=== Pearson Correlation Matrix ===")
        print(corr_matrix)
    else:
        print("\n[Warning] No valid numeric columns found for correlation analysis.")

    # === 4) 可选：写出处理后的 DataFrame 到 Parquet ===
    if output_parquet:
        try:
            df_prepared.to_parquet(output_parquet, index=False)
            print(f"\nDataFrame with new features has been saved to Parquet:\n{output_parquet}")
        except ImportError as e:
            print("[Error] Unable to write Parquet. Please install pyarrow or fastparquet.")
        except Exception as e:
            print(f"[Error] Failed to write parquet file: {e}")

    return df_prepared


# ==================== 示例调用 ====================
if __name__ == "__main__":
    # 假设你已经运行 data_preparation.py，拿到了清洗后的 df_prepared
    # 这里仅示范把同一路径下的 CSV 当做 "清洗后" 数据
    # 实际生产环境下，你会把 data_preparation.py 的输出再读进来
    import os

    prepared_csv = "/home/ec2-user/sylvia/HighPotentialCustomers/ml/output/data/prepared_data.csv"
    if os.path.exists(prepared_csv):
        df_prepared = pd.read_csv(prepared_csv, parse_dates=["last_event_dt_30d", "last_conversion_dt_30d"])
    else:
        raise FileNotFoundError(f"{prepared_csv} not found.")

    df_final = feature_engineering(
        df_prepared,
        ref_date_str="2025-01-21",
        numeric_cols=None,  # 或者传入你想分析的列列表
        output_csv="/home/ec2-user/sylvia/HighPotentialCustomers/ml/output/data/features.csv"
    )

    print("\n=== Final DataFrame Preview ===")
    print(df_final.head(5))
