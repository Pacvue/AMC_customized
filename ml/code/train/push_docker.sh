docker build -t acm_high_potential_customers_train:latest .

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 405907896252.dkr.ecr.us-east-1.amazonaws.com

aws ecr create-repository --repository-name sylvia.wang/acm_high_potential_customers_train --region us-east-1

docker tag acm_high_potential_customers_train:latest 405907896252.dkr.ecr.us-east-1.amazonaws.com/sylvia.wang/acm_high_potential_customers_train:latest

docker push 405907896252.dkr.ecr.us-east-1.amazonaws.com/sylvia.wang/acm_high_potential_customers_train:latest
