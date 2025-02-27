# ML Input Channel
# Train & Infer
        name="dry_run_train_input_dataset",
        model_arn=[configured_model_algorithm_associations],
        channel_name="TRAIN",
        sql_query="SELECT user_id from conversions_all_for_audiences LIMIT 3000",
        time_start="2024-11-20T00:00:00",
        time_end="2024-11-22T00:00:00"

# Trained Model
        name="dry_run_trained_model",
        model_arn=configured_model_algorithm_associations,
        input_channels=[ml_input_channel_id],
        instance_type="ml.m5.xlarge",
        instance_count=1,
        vol_size=10,
        timeout=123400

# Inference Data
         name="dry_run_inference",
         model_arn=configured_model_algorithm_associations,
         trained_model_id=trained_model_id,
         input_channel_id=ml_input_channel_id,
         instance_type="ml.m5.xlarge"
