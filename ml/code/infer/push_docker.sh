docker build -t acm_high_potential_customers_infer:latest .

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 405907896252.dkr.ecr.us-east-1.amazonaws.com


aws ecr create-repository --repository-name sylvia.wang/acm_high_potential_customers_infer --region us-east-1

# docker tag acm_repeat_purchaser_infer:latest <AWS Account ID>.dkr.ecr.us-east-1.amazonaws.com/acm_repeat_purchaser_infer:latest
docker tag acm_high_potential_customers_infer:latest 405907896252.dkr.ecr.us-east-1.amazonaws.com/sylvia.wang/acm_high_potential_customers_infer:latest

# docker push <AWS Account ID>.dkr.ecr.us-east-1.amazonaws.com/acm_repeat_purchaser_infer:latest
docker push 405907896252.dkr.ecr.us-east-1.amazonaws.com/sylvia.wang/acm_high_potential_customers_infer:latest
