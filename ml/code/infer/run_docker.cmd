docker build -t acm_repeat_purchaser_infer:latest .

:: docker run -v "<path to the ml folder>":/opt/ml repeat_purchaser_infer:latest
docker run -p 8080:8080 -e top_pct=15 -v "C:\Users\bkumards\Desktop\ATS\ACM\Models\Repeat Purchaser Prediction\ml":/opt/ml acm_repeat_purchaser_infer:latest

