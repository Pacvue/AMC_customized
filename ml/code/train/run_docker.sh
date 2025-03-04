docker build -t acm_high_potential_customers_train:latest .

docker run -v "/home/ec2-user/sylvia/AMC_customized/ml":/opt/ml \
        acm_high_potential_customers_train:latest