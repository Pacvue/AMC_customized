# High Potential Prediction Using Tabular Neural Networks

This project aims to predict high potentials using a Tabular Neural Network model and Anomaly Detection techniques. The model is trained using historical user purchase and engagement data and leverages AutoGluon for efficient hyperparameter tuning. The project includes scripts for training the model (`train.py`) and performing inference (`infer.py`) to identify users likely to make high potentials.

## Project Structure

- **train.py**: Script to preprocess data, train an LSTM model & Anomaly Detection, tune hyperparameters, and save the best-performing model along with the scalers and label encoders.
- **infer.py**: Script to load the saved model, preprocess new data, and predict churn.
- **amc_sql.sql**: SQL query to pull historical data for training & inference from AMC.

## Prerequisites

Ensure you have the following installed:
- Python 3.7+
- Docker Desktop

## Setup Instructions

1. **Install Docker Desktop**:
   Download and install Docker Desktop from [here](https://www.docker.com/products/docker-desktop/). This application will allow you to build, share, and run containerized python code on you local environment. It provides a streamlined way to manage Docker containers and images, making it easier to develop, test, and deploy applications in a consistent environment.

2. **Directory Structure**:
```
User Churn Prediction/
│
├── ml/
│   ├── code/
│   │   ├── infer/
│   │   │   ├── Dockerfile
│   │   │   ├── infer.py
│   │   │   ├── push_docker.cmd
│   │   │   ├── requirements.txt
│   │   │   ├── test_docker.sh
│   │   │   ├── test_docker.cmd
│   │   │   ├── push_docker.sh
│   │   │   ├── run_docker.sh
│   │   │   └── run_docker.cmd
│   │   │
│   │   └── train/
│   │       ├── Dockerfile
│   │       ├── push_docker.cmd
│   │       ├── push_docker.sh
│   │       ├── requirements.txt
│   │       ├── run_docker.cmd
│   │       ├── run_docker.sh
│   │       └── train.py
│   │
│   ├── input/
│   │   └── data/
│   │       ├── infer/
│   │       │   └── {Training Data.csv}
│   │       └── train/
│   │           └── {Inference Data.csv}
│   │
│   ├── model/
│   └── output/
│       └── data/
│           ├── audiences/
│           └── log/
│
└── SQL/
    └── amc_sql.sql
```


## How to Run

### Generating Synthetic Data
1. Use the `training.sql` and `inference.sql` queries in the AMC Sandbox environment. Ensure that the `_for_audiences` suffix is removed during synthetic data generation. Apply different time windows for training and inference data.

2. Once the synthetic data is available please have them saved in the below paths in csv format
   ```bash
   Training = /ml/input/train/
   Inference = /ml/input/infer/
   ```

### Initiating Docker Desktop
1. Make sure the Docker Desktop is Started and Running properly

2. If there is any issues please restart the Docker Desktop before proceeding

### Configuring the model code and paths
1. {Optional} If needed please update the `train.py` and `infer.py` to fit your Use Case

2. Update the `run_docker.cmd` in both train and infer section of the code with appropriate path from your local directory.

3. Update the `push_docker.cmd` in both train and infer section of the code with appropriate AWS Account IDs

### Training the Model
1. Open CLI or Terminal and navigate to the Train section of the code
   ```bash
   cd /ml/code/train/
   ```

2. Run `run_docker.cmd` (or `run_docker.sh` if your are using Mac) to train the model and save it locally:
   ```bash
   run_docker.cmd
   ```
   ```bash
   ./run_docker.sh
   ```

3. Once satisfied with the execution of Training Container Run `push_docker.cmd` (or `push_docker.sh` if your are using Mac) to push the container to an ECR repository.
   ```bash
   push_docker.cmd
   ```
   ```bash
   ./push_docker.sh
   ```
   > **_NOTE:_** If you are getting errors while running the .sh file use chmod to change the file to an executable


### Performing Inference
1. Open CLI or Terminal and navigate to the Inference section of the code
   ```bash
   cd /ml/code/infer/
   ```

2. Run `run_docker.cmd` (or `run_docker.sh` if your are using Mac) to start the Webserver:
   ```bash
   run_docker.cmd
   ```
   ```bash
   ./run_docker.sh
   ```

3. Once the Webserver is live Run `test_docker.cmd` (or `test_docker.sh` if your are using Mac) to generate inference from the trained model and save it locally:
   ```bash
   test_docker.cmd
   ```
   ```bash
   ./test_docker.sh
   ```

4. Once satisfied with the execution of Inference Container Run `push_docker.cmd` (or `push_docker.sh` if your are using Mac) to push the container to an ECR repository.
   ```bash
   push_docker.cmd
   ```
   ```bash
   ./push_docker.sh
   ```

## Key Components

### train.py
- **Preprocesses Data**: Scales numerical features and handles missing data.
- **Trains a Tabular Neural Network**: Uses AutoGluon to optimize hyperparameters and improve performance.
- **Saves the Model**: Stores the best-performing model, scalers, and other preprocessing artifacts for use during inference.

### infer.py
- **Loads the Model**: Retrieves the trained model and preprocessing artifacts.
- **Preprocesses New Data**: Ensures the new data is consistent with the training data.
- **Generates Predictions**: Outputs probabilities indicating the likelihood of high potentials.
- **Sorts Results**: Ranks users based on their high potential probability.

## Notes
- **SQL Query Consistency**: Use the same SQL query for Training and Inference but apply different time windows.
- **Prediction Threshold**: Adjust the threshold for classifying high-probability high potential customers based on your business needs.

## Troubleshooting
- **Data Validation**: Ensure input data does not contain NaN or infinite values before running the scripts.
- **Performance Optimization**: Use a GPU for faster processing or adjust the batch size and number of epochs for large datasets.

## Future Improvements
- Experiment with additional models like Random Forests or Gradient Boosting for comparison.
- Introduce advanced feature engineering techniques to capture deeper insights.
