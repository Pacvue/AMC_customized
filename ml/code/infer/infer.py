import os
import pandas as pd
import pickle
import logging
import sys
import io
from flask import Flask, request, jsonify, Response

# ================= Logging Setup =================
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
logFormatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(logFormatter)
root_logger.addHandler(stdout_handler)

log_path = os.environ.get("AMC_AUDIENCES_LOG_DIR", '/opt/ml/output/data/log/')
os.makedirs(log_path, exist_ok=True)
file_handler = logging.FileHandler(f"{log_path}/logfile.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logFormatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ================= Flask App Setup =================
app = Flask(__name__)
model = None

def load_model():
    global model
    # Original code commented out:
    # model_dir = os.environ.get("SM_MODEL_DIR", "/home/ec2-user/sylvia/HighPotentialCustomers/ml/model/")
    # model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model/")
    # logger.info(f"Loading model from: {model_dir}")
    # try:
    #     for file_name in os.listdir(model_dir):
    #         if file_name.endswith(".pkl"):
    #             file_path = os.path.join(model_dir, file_name)
    #             with open(file_path, "rb") as f:
    #                 model = pickle.load(f)
    #     logger.info("LightGBM model loaded successfully.")
    # except Exception as e:
    #     logger.error(f"Failed to load model: {e}", exc_info=True)
    #     raise

    # New debug code starts here:
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model/")
    logger.info(f"Debug: Loading model from directory: {model_dir}")
    found_model = False  # Debug: flag indicating if a valid model has been loaded
    try:
        for file_name in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file_name)
            if not os.path.isfile(file_path):  # Debug: Skip non-file entries
                continue
            if file_name.endswith(".pkl"):
                try:
                    with open(file_path, "rb") as f:
                        model = pickle.load(f)
                    logger.info(f"Debug: Successfully loaded pickle model from {file_path}")
                    found_model = True
                    break  # Debug: Exit loop when model is loaded
                except Exception as e:
                    logger.warning(f"Debug: Failed to load pickle file {file_path}: {e}")
            elif file_name.endswith(".tar"):
                try:
                    import tarfile
                    with tarfile.open(file_path, "r") as tar:
                        for member in tar.getmembers():
                            if member.isfile() and member.name.endswith(".pkl"):
                                extracted_file = tar.extractfile(member)
                                if extracted_file:
                                    try:
                                        model = pickle.load(extracted_file)
                                        logger.info(f"Debug: Successfully loaded model from tar package {file_path} (member: {member.name})")
                                        found_model = True
                                        break  # Debug: Model loaded from tar, break out of inner loop
                                    except Exception as e:
                                        logger.warning(f"Debug: Failed to load pickle file {member.name} in tar package {file_path}: {e}")
                        if found_model:
                            break  # Debug: Model loaded from tar, break out of outer loop
                        else:
                            logger.warning(f"Debug: No valid pickle model found in tar package: {file_path}")
                except Exception as e:
                    logger.warning(f"Debug: Error processing tar package {file_path}: {e}")
        if not found_model:
            logger.error("Debug: No valid model file found!")
            raise Exception("No valid model file found!")
        logger.info("Debug: Model loaded successfully.")
    except Exception as e:
        logger.error(f"Debug: Failed to load model: {e}", exc_info=True)
        raise

def load_data_from_request(request):
    """
    Load CSV data from an HTTP request and parse it into a DataFrame.
    """
    try:
        csv_data = io.StringIO(request.data.decode('utf-8'))
        df = pd.read_csv(csv_data)
        logger.info(f"Parsed input data shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load inference data: {e}", exc_info=True)
        raise

def prepare_features(df: pd.DataFrame):
    """
    Prepare input features by dropping columns that were removed during training.
    This ensures that the inference features match the training ones.
    """
    features_to_drop = [
        "high_potential", "user_id_type", "customer_search_term",
        "dma_code", "postal_code", "device_type", "browser_family", "operating_system",
        "total_conversions_after_30d", "total_revenue_after_30d", "total_quantity_after_30d",
        "last_event_dt_before"
    ]
    return df.drop(columns=[col for col in features_to_drop if col in df.columns], errors='ignore')

@app.route('/ping', methods=['GET'])
def ping():
    """
    Health check endpoint: returns 200 if the model is loaded.

    """
    health = model is not None
    status = 200 if health else 503
    return jsonify(status="healthy" if health else "unhealthy"), status

@app.route('/execution-parameters', methods=['GET'])
def execution_parameters():
    return jsonify({"SupportsExecutionParameters": False}), 200

@app.route('/invocations', methods=['POST'])
def invocations():
    """
     Inference endpoint:
    - Receives CSV input via HTTP POST.
    - Prepares features in the same way as during training.
    - Uses the loaded LightGBM model to predict probabilities.
    - Returns the top percentage (default 10%) of records sorted by high potential probability.
    """
    try:
        logger.info("Received an inference request.")
        input_data = load_data_from_request(request)
        # Process input data to obtain prediction features.
        features_df = prepare_features(input_data)[["user_id","total_impressions_30d", "total_clicks_30d"]].reset_index(drop=True) # minimize data size to quick test
        
        # Predict probabilities using the LightGBM model.
        y_pred_proba = model.predict_proba(features_df[["total_impressions_30d", "total_clicks_30d"]].astype(float).fillna(0))
        # For binary classification, use the probability for class 1.

        # generate output data
        output_data = features_df[["user_id"]].copy()
        output_data['high_potential_probability'] = y_pred_proba[:,1]
        
        # Sort the data by predicted probability and select the top percentage.
        top_percent = int(os.environ.get("top_pct", 10))
        output_data.sort_values(by='high_potential_probability', ascending=False, inplace=True)
        output_data = output_data.head(int(len(output_data) * (top_percent / 100)))
        
        # Convert the output to CSV format.
        output_csv = io.StringIO()
        # If 'user_id' exists, include it along with the predicted probability.
        output_cols = ['user_id', 'high_potential_probability']
        existing_cols = [col for col in output_cols if col in output_data.columns]
        output_data[existing_cols].to_csv(output_csv, index=False)
        output_csv.seek(0)
        
        response = Response(output_csv.getvalue(), mimetype='text/csv')
        response.headers['Content-Disposition'] = 'attachment; filename=high_potential_predictions.csv'
        logger.info("Returning predictions as CSV file.")
        return response, 200

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8080)
