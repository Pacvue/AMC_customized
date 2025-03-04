import pandas as pd
import pickle
import glob
import numpy as np
import os
import sys
import logging
import time

# ================= Logging Setup =================
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
file_handler = logging.FileHandler(os.path.join(log_path, "infer_logfile.log"))
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

def preprocess_data(df):
    """
    Preprocess the data for inference, similar to how it was done in training.
    """
    logger.info("Preprocessing data for inference...")
    
    # Keep user_id for later mapping back to results
    user_ids = df['user_id'].copy() if 'user_id' in df.columns else None
    
    # Drop user_id for model processing
    df = df.drop(columns='user_id', errors='ignore')
    
    # Clean data similar to training
    # Drop columns that are all NaN or all zeros (except target_col which shouldn't be in inference data anyway)
    df_cleaned = df.drop(columns=[col for col in df.columns if (df[col].isna() | (df[col] == 0)).all()])
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    
    # Select only numeric columns and fill NaN values - exactly as in training
    X = df_cleaned.select_dtypes(include=['number']).fillna(0).copy()
    
    # Log feature count to help debug
    logger.info(f"Number of features after preprocessing: {X.shape[1]}")
    logger.info(f"Feature names: {list(X.columns)}")
    
    return X, user_ids

def load_model(model_path):
    """
    Load the trained model from the specified path.
    """
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
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
        logger.info("Executing High Potential Customers Inference Script")
        
        # Get paths from environment variables or use defaults
        # data_path = os.environ.get("SM_CHANNEL_TEST", '/home/ec2-user/sylvia/AMC_customized/ml/input/data/infer/')
        data_path = os.environ.get("SM_CHANNEL_TEST", '/opt/ml/input/data/infer/')

    
        # model_dir = os.environ.get("SM_MODEL_DIR", '/home/ec2-user/sylvia/AMC_customized/ml/model/')
        model_dir = os.environ.get("SM_MODEL_DIR", '/opt/ml/model/')

        # output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", '/home/ec2-user/sylvia/AMC_customized/ml/output/data/')
        output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", '/opt/ml/output/data/')

        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data and model
        df = load_data(data_path)
        logger.info(f"Available columns in the dataset: {list(df.columns)}")
        
        X, user_ids = preprocess_data(df)
        
        model_path = os.path.join(model_dir, "high_potential_model.pkl")
        model = load_model(model_path)
        
        # Log model feature information
        logger.info(f"Model expects {model.n_features_} features")
        
        # Check if feature count matches
        if hasattr(model, 'feature_name_') and model.feature_name_:
            logger.info(f"Model feature names: {model.feature_name_}")
            # Find missing or extra features
            model_features = set(model.feature_name_)
            data_features = set(X.columns)
            missing_features = model_features - data_features
            extra_features = data_features - model_features
            
            if missing_features:
                logger.warning(f"Missing features in inference data: {missing_features}")
            if extra_features:
                logger.warning(f"Extra features in inference data: {extra_features}")
                # Remove extra features
                X = X.drop(columns=list(extra_features))
                
            # Ensure columns are in the same order as the model expects
            if hasattr(model, 'feature_name_'):
                X = X[model.feature_name_]
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = model.predict(X)
        prediction_probs = model.predict_proba(X)[:, 1]  # Probability of positive class
        
        # Create results dataframe
        results = pd.DataFrame()
        if user_ids is not None:
            results['user_id'] = user_ids
        
        results['prediction'] = predictions
        results['probability'] = prediction_probs
        
        # Save results
        output_path = os.path.join(output_dir, "predictions.csv")
        results.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        logger.info("High Potential Customers Inference Script Executed Successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    main()
