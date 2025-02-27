import os
import pandas as pd
import pickle
import logging
import sys
from autogluon.tabular import TabularPredictor
import time

# Logging setup
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
logFormatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(logFormatter)
root_logger.addHandler(stdout_handler)

# log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/opt/ml/output/data/log/')
log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/output/data/log')
os.makedirs(log_path, exist_ok=True)
file_handler = logging.FileHandler(f"{log_path}/logfile.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logFormatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_data(file_path):
    logger.info(f"Reading file {file_path} with Pandas.")
    df = pd.read_csv(file_path)
    return df

def calculate_high_potential_flag(df):
    logger.info("Calculating high potential flag dynamically...")
    df['high_potential'] = ((df['total_conversions_30d'] > 0) | (df['total_revenue_30d'] > 0)).astype(int)
    return df

def train_and_export_model(train, test, model_dir):
    logger.info("Training the high potential customers model using AutoGluon TabularPredictor...")
    predictor = TabularPredictor(label='high_potential', path=model_dir)
    predictor.fit(train_data=train, tuning_data=test, presets='medium_quality', use_bag_holdout=True)
    predictor.save()
    logger.info(f"Model saved in directory: {model_dir}")
    return predictor

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting execution of {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Execution of {func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@measure_execution_time
def main():
    try:
        logger.info("Executing High Potential Customers Training Script")
        # train_dataset_file = os.environ.get("SM_CHANNEL_TRAIN", '/opt/ml/input/data/train.csv')
        train_dataset_file = os.environ.get("SM_CHANNEL_TRAIN", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/input/data/train/train.csv')

        # test_dataset_file = os.environ.get("SM_CHANNEL_TEST", '/opt/ml/input/data/test.csv')
        test_dataset_file = os.environ.get("SM_CHANNEL_TEST", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/input/data/train/test.csv')

        # model_dir = os.environ.get("SM_MODEL_DIR", '/opt/ml/model/')
        model_dir = os.environ.get("SM_MODEL_DIR", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/model/')

        os.makedirs(model_dir, exist_ok=True)
        
        train_df = load_data(train_dataset_file)
        test_df = load_data(test_dataset_file)
        
        train_df = calculate_high_potential_flag(train_df)
        test_df = calculate_high_potential_flag(test_df)
        
        predictor = train_and_export_model(train_df, test_df, model_dir)
        
        logger.info("High Potential Customers Training Script Executed Successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()



# import os
# import pandas as pd
# import pickle
# import logging
# import sys
# import glob
# from autogluon.tabular import TabularPredictor
# from sklearn.model_selection import train_test_split
# import psutil
# import time

# # Logging setup
# root_logger = logging.getLogger()

# # Console Handler
# stdout_handler = logging.StreamHandler(sys.stdout)
# logFormatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
# stdout_handler.setLevel(logging.DEBUG)
# stdout_handler.setFormatter(logFormatter)
# root_logger.addHandler(stdout_handler)

# # File Handler
# # log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/opt/ml/output/data/log/')
# log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/output/data/log')
# os.makedirs(log_path, exist_ok=True)
# file_handler = logging.FileHandler(f"{log_path}/logfile.log")
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(logFormatter)
# root_logger.addHandler(file_handler)

# # Application Logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# def log_directory_info(current_directory=os.getcwd()):
#     logger.info(f"Current working directory: {current_directory}")
#     directory_contents = os.listdir(current_directory)
#     logger.info(f"Directory contents: {directory_contents}")

# def load_data(file_path):
#     log_directory_info(file_path)
#     all_files = glob.glob(os.path.join(file_path, "*.csv"))
#     logger.info(f"Reading files with Pandas.")
#     df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
#     return df

# def calculate_high_potential_flag(df):
#     """
#     Create a new target column 'high_potential' using the SQL-generated features.
#     In this example, we define high potential as customers with any conversions or revenue in the last 30 days.
#     Adjust the thresholds or logic as needed.
#     """
#     logger.info("Calculating high potential flag dynamically...")
#     # For example, flag as high potential if total_conversions_30d > 0 or total_revenue_30d > 0
#     df['high_potential'] = ((df['total_conversions_30d'] > 0) | (df['total_revenue_30d'] > 0)).astype(int)
#     logger.info("High potential flag calculated successfully.")
#     return df

# def save_purchase_history(df, model_dir):
#     """
#     Optionally, save an aggregated summary (for example, total conversions per user) as a pickle file.
#     """
#     logger.info("Saving user purchase history as a Pickle file...")
#     # Adjust the grouping as needed based on your available columns.
#     purchase_history = df.groupby('user_id')['total_conversions_30d'].sum().reset_index()
#     purchase_history['high_potential'] = (purchase_history['total_conversions_30d'] > 0).astype(int)
#     purchase_history_path = os.path.join(model_dir, "purchase_history.pkl")
#     with open(purchase_history_path, 'wb') as f:
#         pickle.dump(purchase_history, f)
#     logger.info(f"Purchase history saved to: {purchase_history_path}")

# def split_data(df):
#     """
#     Split the dataset for training and testing. In this example,
#     we assume that the target column is 'high_potential' computed above.
#     """
#     logger.info("Splitting data...")
#     # Ensure that the target exists and drop any rows with missing target values.
#     train_test_df = df[df['high_potential'].notna()].reset_index(drop=True)
#     train_test_df['label'] = train_test_df['high_potential']
#     train, test = train_test_split(train_test_df, test_size=0.2, random_state=42)
#     logger.info("Training data shape: %s", train.shape)
#     logger.info("Testing data shape: %s", test.shape)
#     return train, test

# def train_and_export_model(train, test, model_dir):
#     logger.info("Training the high potential customers model using AutoGluon TabularPredictor...")
#     predictor = TabularPredictor(
#         label='label',
#         path=model_dir
#     )
#     predictor.fit(
#         train_data=train,
#         tuning_data=test,
#         presets='medium_quality',
#         use_bag_holdout=True  # Ensures bagging uses holdout validation
#     )
#     logger.info("Evaluating model on test data...")
#     evaluation_metrics = predictor.evaluate(test)
#     logger.info(f"Evaluation metrics: {evaluation_metrics}")

#     # Get leaderboard of top models
#     logger.info("Fetching leaderboard of top 10 models...")
#     leaderboard = predictor.leaderboard(test, extra_info=True)
#     top_10_models = leaderboard['model'].head(10).tolist()

#     logger.info("Evaluation metrics of top 10 models:")
#     for model_name in top_10_models:
#         if 'ensemble' in model_name.lower():
#             logger.warning(f"Skipping ensemble model: {model_name}")
#             continue
#         try:
#             eval_metrics = predictor.evaluate(test, model=model_name)
#             logger.info(f"\nModel: {model_name}")
#             logger.info(f"Accuracy: {eval_metrics.get('accuracy')}")
#             logger.info(f"Balanced Accuracy: {eval_metrics.get('balanced_accuracy')}")
#             logger.info(f"MCC: {eval_metrics.get('mcc')}")
#             logger.info(f"ROC AUC: {eval_metrics.get('roc_auc')}")
#             logger.info(f"F1 Score: {eval_metrics.get('f1')}")
#             logger.info(f"Precision: {eval_metrics.get('precision')}")
#             logger.info(f"Recall: {eval_metrics.get('recall')}")
#         except Exception as e:
#             logger.error(f"Error evaluating model {model_name}: {str(e)}")

#     logger.info("Saving trained model...")
#     predictor.save()
#     logger.info(f"Model saved in directory: {model_dir}")
#     return predictor

# def measure_execution_time(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         logger.info(f"Starting execution of {func.__name__}")
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         execution_time = end_time - start_time
#         logger.info(f"Execution of {func.__name__} completed in {execution_time:.2f} seconds")
#         return result
#     return wrapper

# @measure_execution_time
# def main():
#     try:
#         logger.info("Executing High Potential Customers Training Script")
#         # dataset_path = os.environ.get("SM_CHANNEL_TRAIN", '/opt/ml/input/data/train/')
#         dataset_path = os.environ.get("SM_CHANNEL_TRAIN", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/input/data/train/')
#         # model_dir = os.environ.get("SM_MODEL_DIR", '/opt/ml/model/')
#         model_dir = os.environ.get("SM_MODEL_DIR", '/home/ec2-user/sylvia/HighPotentialCustomers/ml/model')
#         os.makedirs(model_dir, exist_ok=True)
        
#         # Load data (the CSV files should contain the columns produced by your SQL query)
#         df = load_data(dataset_path)
#         logger.info(f"Available columns in the dataset: {list(df.columns)}")
        
#         # Compute the target variable (high potential flag) based on your SQL-derived columns.
#         df = calculate_high_potential_flag(df)
        
#         train, test = split_data(df)
#         predictor = train_and_export_model(train, test, model_dir)
#         # save_purchase_history(df, model_dir)
        
#         logger.info("High Potential Customers Training Script Executed Successfully.")
#     except Exception as e:
#         logger.error(f"An error occurred: {e}", exc_info=True)
#         sys.exit(1)

# if __name__ == '__main__':
#     main()
