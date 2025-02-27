docker build -t acm_repeat_purchaser_infer:latest .

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 008971674832.dkr.ecr.us-east-1.amazonaws.com

aws ecr create-repository --repository-name acm_repeat_purchaser_infer

:: docker tag acm_repeat_purchaser_infer:latest <AWS Account ID>.dkr.ecr.us-east-1.amazonaws.com/acm_repeat_purchaser_infer:latest
docker tag acm_repeat_purchaser_infer:latest 008971674832.dkr.ecr.us-east-1.amazonaws.com/acm_repeat_purchaser_infer:latest

:: docker push <AWS Account ID>.dkr.ecr.us-east-1.amazonaws.com/acm_repeat_purchaser_infer:latest
docker push 008971674832.dkr.ecr.us-east-1.amazonaws.com/acm_repeat_purchaser_infer:latest
