import os
import pandas as pd
import pickle
import logging
import sys
import io
from flask import Flask, request, jsonify, Response
from autogluon.tabular import TabularPredictor

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

# Flask app setup
app = Flask(__name__)

# Global variables
model = None
purchase_history = None

# Load the trained model
def load_model():
    global model
    # model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model/")
    model_dir = os.environ.get("SM_MODEL_DIR", "/home/ec2-user/sylvia/HighPotentialCustomers/ml/model/")
    logger.info(f"Loading model from directory: {model_dir}")
    try:
        model = TabularPredictor.load(model_dir)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise

# Load the inference dataset
def load_data(request):
    logger.info("Loading the input data.")
    try:
        csv_data = io.StringIO(request.data.decode('utf-8'))
        df = pd.read_csv(csv_data)
        logger.info(f"Parsed input data shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load inference data: {e}", exc_info=True)
        raise

# Health check endpoint
@app.route('/ping', methods=['GET'])
def ping():
    health = model is not None  # Check if the model is loaded
    status = 200 if health else 503
    return jsonify(status="healthy" if health else "unhealthy"), status

@app.route('/invocations', methods=['POST'])
def invocations():
    try:
        logger.info("Received an inference request.")
        input_data = load_data(request)
        top_percent = int(os.environ.get("top_pct", 10))
        
        # Generate predictions
        predictions = model.predict_proba(input_data)
        input_data['high_potential_probability'] = predictions[1]
        input_data.sort_values(by='high_potential_probability', ascending=False, inplace=True)
        
        output_data = input_data.head(int(len(input_data) * (top_percent / 100)))
        logger.info(f"Predictions generated successfully and top {top_percent}% is exported.")
        
        # Convert DataFrame to CSV
        output_csv = io.StringIO()
        output_data[['user_id', 'high_potential_probability']].to_csv(output_csv, index=False)
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
