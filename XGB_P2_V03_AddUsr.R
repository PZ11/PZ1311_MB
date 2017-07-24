
###########################################################################################################
#
# Kaggle Instacart competition
# Paul Zhang, July 2017
# Phase 2, on xgboost starter, start score as 0.3818, submit 42
# Try to add all the variables based on artical of "Repeat Buyer Prediction for E-Commerce" 
#
###########################################################################################################

options(warn=-1)
#options(warn=0)

library(data.table)
library(dplyr)
library(tidyr)
library(xgboost)
#library(matrix)
library(zoo)

##### ALL F1 Functions #####
# F Score Calculation 
F1Score <- function (fact, pred) {
  "%ni%" <- Negate("%in%")
  fact <- strsplit(fact, " ")
  fact <- fact[nzchar(fact)][[1]]
  
  pred <- strsplit(pred, " ")
  pred <- pred[nzchar(pred)][[1]]
  
  TP = sum(pred %in% fact)
  
  if(TP == 0) {
    return(0)
  }
  
  precision <- TP/length(pred)
  recall <- TP/length(fact)
  
  2 * precision * recall / (precision + recall)
}

applyF1Score <- function (df) {
  apply(df, 1, function (x) F1Score(x["fact"], x["predicted"]))
}

# Calcualte submit Average F Store 
calc_submit_f1 <- function (acc_th, t, m, xgbM) {
  
  t$pred_reordered <- predict(m, xgbM)
  
  t$pred_reordered <- (t$pred_reordered > acc_th) * 1
  
  submission <- t %>%
    filter(pred_reordered == 1) %>%
    group_by(order_id) %>%
    summarise(
      products = paste(product_id, collapse = " ")
    )
  
  missing <- data.frame(
    order_id = unique(t$order_id[!t$order_id %in% submission$order_id]),
    products = "None"
  )
  
  submission <- submission %>% bind_rows(missing) %>% arrange(order_id)
  #write.csv(submission, file = "C:/DEV/MarketBusket/Submit/submit_pz_xgb_01.csv", row.names = F)
  #str(submission)
  
  
  #-----------------------------------------------
  fact <- t %>%
    filter(reordered == 1) %>%
    group_by(order_id) %>%
    summarise(
      fact_products = paste(product_id, collapse = " ")
    )
  
  fact_missing <- data.frame(
    order_id = unique(t$order_id[!t$order_id %in% fact$order_id]),
    fact_products = "None"
  )
  fact <- fact %>% bind_rows(fact_missing) %>% arrange(order_id)
  
  submission_fact <- fact %>% 
    inner_join(submission, by = "order_id") 
  
  df <- data.frame(order_id = submission_fact$order_id, 
                   fact = submission_fact$fact_products,
                   predicted = submission_fact$products)
  
  df$f1 <- applyF1Score(df)
  calc_f1 = mean(df$f1)
  print(paste(" f1 Score is:", calc_f1, 
              "fact: ",nrow(fact),
              "fact_missing: ",nrow(missing),
              "missing: ",nrow(fact_missing),
              "sub_fact: ",nrow(submission_fact) ))  
  return (calc_f1)
  
}

# Windows Load Data ---------------------------------------------------------------
path <- "C:/DEV/MarketBusket/Data_ORG/"

#Mac Load Data
#path <- "/Users/yuepengzhang/Documents/Kaggle/MarketBasket/InputORG"

aisles <- fread(file.path(path, "aisles.csv"))
departments <- fread(file.path(path, "departments.csv"))
orderp <- fread(file.path(path, "order_products__prior.csv"))
ordert <- fread(file.path(path, "order_products__train.csv"))
orders <- fread(file.path(path, "orders.csv"))
products <- fread(file.path(path, "products.csv"))


#### Reshape data ------------------------------------------------------------

aisles$aisle <- as.factor(aisles$aisle)
departments$department <- as.factor(departments$department)
products$product_name <- as.factor(products$product_name)

products <- products %>% 
  inner_join(aisles) %>% inner_join(departments) %>% 
  select(-aisle_id, -department_id)

rm(aisles, departments)

# Debug, filter to user 1 only -----------------------------------------------
orders_org <- orders
#orders <- orders_org[user_id == 2 , ]
orders <- orders_org[user_id <= 10, ]


# Orders data 
orders$eval_set <- as.factor(orders$eval_set)
orders$days_since_prior_order[is.na(orders$days_since_prior_order)] <- 0

orders <-  orders %>% 
  arrange(user_id, order_number) %>%
  group_by (user_id) %>% 
  mutate( days_cumsum = cumsum(days_since_prior_order))

orders_prod_prior <- orders %>%   inner_join(orderp, by = "order_id") %>%
  arrange(user_id, product_id, order_number)

# ordert to validate result only, ! do not use for training ! 
orders_prod_test <- orders %>% 
  inner_join(ordert, by = "order_id")
ordert$user_id <- orders$user_id[match(ordert$order_id, orders$order_id)]




#### Features on Users -----------------------------------------------
users_prior <- orders %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise(
    u_order_cnt = max(order_number),
    u_days_cnt = sum(days_since_prior_order, na.rm = T)
  )



#### Features on User-Product -----------------------------------------
# order product diff from previous lag 
orders_prod_prior_diff <- orders_prod_prior %>%
  group_by(user_id, product_id) %>%
  mutate(up_order_lag1 = lag(x = order_number, n = 1)) %>%
  mutate(up_days_lag1 = lag(x = days_cumsum, n = 1)) %>%
  mutate(up_order_lag1_diff = order_number - up_order_lag1) %>%
  mutate(up_days_lag1_diff = days_cumsum - up_days_lag1)

data_op_lag1_diff <- orders_prod_prior_diff %>%
  filter(up_order_lag1_diff >= 0) %>%
  select (user_id,product_id,up_order_lag1_diff, up_days_lag1_diff ) %>%
  group_by(user_id, product_id) %>%
  summarise( 
    up_order_lag1_diff_max = max(up_order_lag1_diff,na.rm = T),
    up_order_lag1_diff_mean = mean(up_order_lag1_diff,na.rm = T),
    up_order_lag1_diff_median = median(up_order_lag1_diff,na.rm = T),
#    up_order_lag1_diff_sd = sd(up_order_lag1_diff,na.rm = T),
    
    up_days_lag1_diff_max = max(up_days_lag1_diff,na.rm = T),    
    up_days_lag1_diff_mean = mean(up_days_lag1_diff,na.rm = T),
    up_days_lag1_diff_median = median(up_days_lag1_diff,na.rm = T)
#    ,up_days_lag1_diff_sd = sd(up_days_lag1_diff,na.rm = T)
  )

data_up <- orders_prod_prior %>%
  group_by(user_id, product_id) %>%
  summarise( 
    up_order_max = max(order_number,na.rm = T),
    up_order_min = min(order_number,na.rm = T),
    up_reorder_cnt = n(),

    up_days_max = max(days_cumsum,na.rm = T),
    
    up_dow_mean = mean(order_dow,na.rm = T),
#    up_dow_sd = sd(order_dow,na.rm = T),
    up_hod_mean = mean(order_hour_of_day,na.rm = T),
#    up_hod_sd = sd(order_hour_of_day,na.rm = T),
    up_atco_mean = mean(add_to_cart_order,na.rm = T)
#    ,up_atco_sd = sd(add_to_cart_order,na.rm = T)   
             ) %>%
  inner_join(users_prior %>% select ( user_id,u_order_cnt,u_days_cnt ), by = ("user_id")) %>%
  left_join(data_op_lag1_diff, by = c("user_id", "product_id"))

data_up$up_order_rate <- data_up$up_reorder_cnt / data_up$u_order_cnt
data_up$up_order_rate_since_fist <- data_up$up_reorder_cnt / 
  ( data_up$u_order_cnt - data_up$up_order_min + 1)
 

data <- data_up %>% 
  inner_join(orders %>% 
               filter(eval_set != "prior") %>%
               select ( user_id, order_number, order_dow, order_hour_of_day, days_cumsum ), by=("user_id")) %>%
  mutate( up_order_cnt_after_last = order_number - up_order_max) %>%
  mutate( up_days_cnt_after_last = days_cumsum - up_days_max) %>%
  mutate( up_dow_ratio = order_dow / up_dow_mean) %>%
  mutate( up_hod_ratio = order_hour_of_day / up_hod_mean) %>%
  mutate( up_ordercount_ratio = up_order_cnt_after_last / up_order_lag1_diff_mean) %>%
  mutate( up_daycount_ratio = up_days_cnt_after_last / up_days_lag1_diff_mean)


data$order_number = NULL
data$order_dow = NULL
data$order_hour_of_day = NULL
data$days_cumsum = NULL




#### Load Data from Environment ############################################
############################################################################
############################################################################



# Add reorder from ordert for model validation. 
data <- data %>% 
  left_join(ordert %>% select(user_id, product_id, reordered), by = c("user_id", "product_id"))
# Add eval_set of test or train, add order_id from orders
data <- data %>% 
  inner_join(orders %>% filter(eval_set != "prior") %>%
               select ( user_id, eval_set, order_id), by=("user_id")) 
  

# rm(orders_prod_prior_diff, data_op_lag1_diff)
# rm(data_up, users_prior, orders_prod_prior, orders_prod_test)
# rm(orderp, ordert, orders)
# gc()

#### Features on recent activies ###########################################


# Products ----------------------------------------------------------------


# Users -------------------------------------------------------------------


# PZ, redefine train and test ----------------------------------
datanew <- data[data$eval_set == "train", ]

rm(orders_prod_prior_diff, data_op_lag1_diff)
rm(data_up, users_prior, orders_prod_prior, orders_prod_test)
rm(orderp, ordert, orders)
gc()


# Train / Test datasets ---------------------------------------------------
set.seed(101) 
userlist = unique(datanew$user_id)
sample <- sample.int(n = length(userlist), 
                     size = floor(.75*length(userlist)), 
                     replace = F)

train <- datanew[datanew$user_id %in% userlist[sample],]
test  <- datanew[!datanew$user_id %in% userlist[sample],]

train$eval_set <- NULL
train$user_id <- NULL
train$product_id <- NULL
train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0

test$eval_set <- NULL
#test$user_id <- NULL


## PZ, Model -------------------------------------------------------------------
set.seed(1)
# Somehow it doesn't work here 
#subtrain <- train %>% sample_frac(0.1)

subtrain <- train[sample(nrow(train)/10),]

# #### Original Parameter #########################
params <- list(
  "objective"           = "binary:logistic",
  "eval_metric"         = "logloss",
  "eta"                 = 0.1,
  "max_depth"           = 6,
  "min_child_weight"    = 10,
  "gamma"               = 0.70,
  "subsample"           = 0.76,
  "colsample_bytree"    = 0.95,
  "alpha"               = 2e-05,
  "lambda"              = 10,
  "silent"              = 1
)

X <- xgb.DMatrix(as.matrix(subtrain %>% select(-reordered)), label = subtrain$reordered)
model <- xgboost(data = X, params = params, nrounds = 80)

importance <- xgb.importance(colnames(X), model = model)
xgb.ggplot.importance(importance)

# Apply model -------------------------------------------------------------
best_f1 = 0 
best_threshold = 0

test[] <- lapply(test, as.numeric)

X <- xgb.DMatrix(as.matrix(test %>% select( -user_id, -product_id, -reordered)))

for(i in 17:40)
{
  acc_threshold = as.numeric(i/100)
  f1 =calc_submit_f1(acc_threshold, test, model, X)
  if (best_f1 < f1) {
    best_f1 = f1
    best_threshold = acc_threshold
  }
}
print(paste("best_threshold:", best_threshold, "best_f1 is:", best_f1 ))  




###########################################################################
######################### Submit ##########################################
###########################################################################
# Train / Test datasets ---------------------------------------------------
rm(orders_prod_prior_diff, data_op_lag1_diff)
rm(data_up, users_prior, orders_prod_prior, orders_prod_test)
rm(orderp, ordert, orders)
gc()

# PZ, redefine train and test ----------------------------------
datanew <- data[,]

acc_threshold <- 0.19

train <- as.data.frame(datanew[datanew$eval_set == "train",])
test <- as.data.frame(datanew[datanew$eval_set == "test",])



train$eval_set <- NULL
train$user_id <- NULL
train$product_id <- NULL
train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0


test$eval_set <- NULL
test$user_id <- NULL



# Model Origina Para -------------------------------------------------------------------

params <- list(
  "objective"           = "binary:logistic",
  "eval_metric"         = "logloss",
  "eta"                 = 0.1,
  "max_depth"           = 6,
  "min_child_weight"    = 10,
  "gamma"               = 0.70,
  "subsample"           = 0.76,
  "colsample_bytree"    = 0.95,
  "alpha"               = 2e-05,
  "lambda"              = 10,
  "silent"             = 1
)

# PZ, para 02  -------------------------------------------------------------------

set.seed(1)
subtrain <- train %>% sample_frac(0.1)
X <- xgb.DMatrix(as.matrix(subtrain %>% select(-reordered)), label = subtrain$reordered)
model <- xgboost(data = X, params = params, nrounds = 80)

importance <- xgb.importance(colnames(X), model = model)
xgb.ggplot.importance(importance)

#rm(X, importance, subtrain)
#gc()


# Apply model -------------------------------------------------------------
X <- xgb.DMatrix(as.matrix(test %>% select(-order_id, -product_id)))
test$pred_reordered <- predict(model, X)

test$pred_reordered <- (test$pred_reordered > acc_threshold) * 1

submission <- test %>%
  filter(pred_reordered == 1) %>%
  group_by(order_id) %>%
  summarise(
    products = paste(product_id, collapse = " ")
  )

missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]),
  products = "None"
)

submission <- submission %>% bind_rows(missing) %>% arrange(order_id)
write.csv(submission, file = "C:/DEV/MarketBusket/Submit/submit_ph2_try44.csv", row.names = F)



