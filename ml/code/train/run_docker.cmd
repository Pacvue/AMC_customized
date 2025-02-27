docker build -t acm_repeat_purchaser_train:latest .

:: docker run -v "<path to the ml folder>":/opt/ml acm_repeat_purchaser_train:latest
docker run -v "C:\Users\bkumards\Desktop\ATS\ACM\Models\Repeat Purchaser Prediction\ml":/opt/ml acm_repeat_purchaser_train:latest


/home/ec2-user/sylvia/RepeatPurchaserPrediction/ml