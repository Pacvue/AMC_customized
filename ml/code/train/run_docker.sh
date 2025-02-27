docker build -t acm_high_potential_customers_train:latest .

# docker run -v "<path to the ml folder>":/opt/ml acm_repeat_purchaser_train:latest
# docker run -v "/home/ec2-user/sylvia/RepeatPurchaserPrediction_Dj/ml/data":/opt/ml/input/data acm_repeat_purchaser_train:latest


docker run -v "/home/ec2-user/sylvia/HighPotentialCustomers/ml":/opt/ml \
        acm_high_potential_customers_train:latest
        #    -v "/home/ec2-user/sylvia/HighPotentialCustomers/ml/model":/opt/ml/model \
           

