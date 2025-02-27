docker build -t acm_repeat_purchaser_train:latest .

# docker run -v "<path to the ml folder>":/opt/ml acm_repeat_purchaser_train:latest
docker run -v "/home/ec2-user/sylvia/RepeatPurchaserPrediction/ml":/opt/ml acm_repeat_purchaser_train:latest