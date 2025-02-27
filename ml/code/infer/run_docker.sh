docker build -t acm_repeat_purchaser_infer:latest .

# docker run -v "<path to the ml folder>":/opt/ml repeat_purchaser_infer:latest
docker run -v "/home/ec2-user/sylvia/RepeatPurchaserPrediction/ml":/opt/ml acm_repeat_purchaser_infer:latest

