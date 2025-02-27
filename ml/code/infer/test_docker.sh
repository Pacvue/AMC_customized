curl -X GET http://127.0.0.1:8080/ping

curl -X POST http://127.0.0.1:8080/invocations -H "Content-Type: text/csv" --data-binary @"C:\Users\bkumards\Desktop\ATS\ACM\Models\Repeat Purchaser Prediction\ml\input\data\infer\sandbox_query_results_bfa0f680-38f7-452c-8eff-aa1056bc92f1.csv" -o "C:\Users\bkumards\Desktop\ATS\ACM\Models\Repeat Purchaser Prediction\ml\output\data\audiences\output.csv"
