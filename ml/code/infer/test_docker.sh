curl -X GET http://172.17.0.8:8080/ping

curl -X POST http://172.17.0.8:8080/invocations -H "Content-Type: text/csv" --data-binary @"/home/ec2-user/sylvia/HighPotentialCustomers/ml/input/data/infer/sandbox_query_results_008da601-3002-45dd-867e-7c1c6fb01415_last90days.csv" -o "/home/ec2-user/sylvia/HighPotentialCustomers/ml/output/data/audiences/output.csv"

