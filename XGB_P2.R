
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
  
  # acc_th = 10
  # t = test
  # m = model
  # xgbM = X
  
  
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
  print(paste(
    " acc_th is:", acc_th, 
    " f1 Score is:", calc_f1, 
    "fact: ",nrow(fact),
    "fact_missing: ",nrow(fact_missing),
    "missing: ",nrow(missing),
    "sub_fact: ",nrow(submission_fact) ))  
  return (calc_f1)
  
}

# Calcualte submit Average F Store 
calc_submit_f1_noPred <- function (t) {
  
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
  print(paste(
    " f1 Score is:", calc_f1, 
    "fact: ",nrow(fact),
    "missing: ",nrow(missing),
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
#orders <- orders_org[user_id == 2 , ]
#orders_org <- orders
#orders <- orders_org[user_id <= 10, ]


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

# Get the Users order cout and day count to prepare up features 
users_prior <- orders %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise(
    u_order_cnt = max(order_number),
    u_days_cnt = sum(days_since_prior_order, na.rm = T)
  )





############# !!! run up here before load data environment #################


#### User-Product Features -----------------------------------------
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


#### User Features-----------------------------------------------
user_dow_hod <- orders %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise( u_dow_mean = mean(order_dow),
             u_dow_sd = sd(order_dow),
             u_hod_mean = mean(order_hour_of_day),
             u_hod_sd = sd(order_hour_of_day)                   
             )

user_dspo <- orders %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise(
    u_dspo_mean = mean(days_since_prior_order),
    u_dspo_sd = sd(days_since_prior_order)
  )

user_order_f <- user_dow_hod %>%
  inner_join(user_dspo, by = "user_id") %>%
  inner_join(orders %>% filter(eval_set != "prior") %>%
               select(user_id, order_dow, order_hour_of_day, days_since_prior_order), by="user_id") %>%
  mutate(u_dow_ratio = order_dow / u_dow_mean, 
         u_hod_ratio = order_hour_of_day / u_hod_mean,
         u_dspo_ratio = days_since_prior_order / u_dspo_mean)
               
user_order_f$order_dow <- NULL
user_order_f$order_hour_of_day <- NULL
user_order_f$days_since_prior_order <- NULL

# Get  product count and reordered ratio 
users_prod_cnt <- orders_prod_prior %>%
  group_by(user_id) %>%
  summarise(
    u_prod_cnt = n(),
    u_reorder_prod_cnt = sum(reordered,na.rm = T)
  ) %>%
mutate(u_reorder_prod_ratio = u_reorder_prod_cnt / u_prod_cnt)

# Get the unique product count and reordered ratio 
users_unique_prod_cnt <- orders_prod_prior %>%
  group_by(user_id, product_id) %>%
  summarise(u_prod_cnt_tmp = 1) %>%
  group_by(user_id) %>%
  summarise(u_unique_prod_cnt = n())

users_unique_prod_reordered_cnt <- orders_prod_prior %>%
  filter(reordered == 1) %>%
  group_by(user_id, product_id) %>%
  summarise(u_prod_cnt_tmp = 1) %>%
  group_by(user_id) %>%
  summarise(u_unique_reorder_prod_cnt = n())

# Get busket Size 
users_basket <- orders_prod_prior %>%
  group_by(user_id, order_id) %>%
  summarise(
    user_order_basket = n(),
    user_basket_reorder_cnt = sum(reordered)
  ) %>%
  mutate(user_reorder_ratio = user_basket_reorder_cnt / user_order_basket)%>%
  group_by(user_id) %>%
  summarise(
    u_basket_mean = mean(user_order_basket),
    u_basket_median = median(user_order_basket), 
    u_basket_max = max(user_order_basket), 
    u_basket_sd = sd(user_order_basket), 
    u_basket_reorder_ratio_mean = mean(user_reorder_ratio),
    u_basket_reorder_ratio_sd = sd(user_reorder_ratio)
  )

# Combine all user features 
users <- users_prod_cnt %>%
  inner_join(user_order_f, by = "user_id") %>%
  inner_join(users_basket, by = "user_id") %>%
  inner_join(users_unique_prod_cnt, by = "user_id") %>%
  inner_join(users_unique_prod_reordered_cnt, by = "user_id") %>%
  mutate ( u_unique_reorder_prod_ratio = u_unique_reorder_prod_cnt / u_unique_prod_cnt ) 



rm(users_prod_cnt, users_unique_prod_cnt, users_unique_prod_reordered_cnt, users_basket)
rm(user_dow_hod, user_dspo, user_order_f)

# Add user Features 
data <- data %>% 
  left_join(users , by=("user_id")) 


#### Load Data from Environment ############################################



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

# Products ----------------------------------------------------------------


prod_user_cnt <- orders_prod_prior %>%
  group_by(product_id) %>%
  summarise( p_user_cnt = n(), 
             p_atco_mean = mean(add_to_cart_order),
             p_atco_median = median(add_to_cart_order))


prod_reorder_user_cnt <- orders_prod_prior %>%
  filter(reordered == 1) %>%
  group_by(product_id) %>%
summarise( p_reorder_user_cnt = n())
  
prod_user_cnt <- prod_user_cnt %>%
  left_join(prod_reorder_user_cnt, by = "product_id") %>%
  mutate ( p_reorder_user_rate = p_reorder_user_cnt / p_user_cnt ) 


# Get the unique user count 
prod_unique_user_cnt <- orders_prod_prior %>%
  group_by(user_id, product_id) %>%
  summarise(p_user_cnt_tmp = 1) %>%
  group_by(product_id) %>%
  summarise(p_unique_user_cnt = n())


# Get the unique reordered user count 
prod_unique_user_reordered_cnt <- orders_prod_prior %>%
  filter(reordered == 1) %>%
  group_by(user_id, product_id) %>%
  summarise(p_user_reorder_cnt_tmp = 1) %>%
  group_by(product_id) %>%
  summarise(p_unique_reorder_user_cnt = n())
  

# Combine all user features 
prod_features <- prod_user_cnt %>%
  inner_join( prod_unique_user_cnt , by = "product_id") %>%
  left_join(prod_unique_user_reordered_cnt, by = "product_id") %>%
  mutate ( p_unique_reorder_user_rate = p_unique_reorder_user_cnt / p_unique_user_cnt ) 

prod_features$p_reorder_user_cnt <- NULL
prod_features$p_unique_reorder_user_cnt <- NULL

rm(prod_unique_user_cnt, prod_unique_user_reordered_cnt, prod_user_cnt, prod_reorder_user_cnt)
  
data <- data %>%
  left_join(prod_features, by = "product_id") 



#### P90D User_Product Features  ###########################################

orders_90d <-  orders %>% 
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  mutate( user_period = sum(days_since_prior_order) ) %>%
  mutate( days_rev = user_period - days_cumsum ) %>%
  filter(days_rev <= 90) 


users_90d <- orders_90d %>%
  group_by(user_id) %>%
  summarise(
    p90d_max_orders = max(order_number),
    p90d_min_orders = min(order_number)
  ) %>%
  mutate( p90d_user_orders = p90d_max_orders - p90d_min_orders + 1 ) %>%
  select(user_id, p90d_user_orders)

data_90d <- orders_prod_prior %>%
  inner_join(orders_90d %>% select(user_id, order_id), by=c("user_id", "order_id")) %>%
  group_by(user_id, product_id) %>% 
  summarise(
    p90d_up_orders = n()
  )

data_90d <- data_90d %>% 
  inner_join(users_90d, by = "user_id")

data_90d$p90d_up_order_rate <- data_90d$p90d_up_orders / data_90d$p90d_user_orders

#data_90d$p90d_user_orders <- NULL

data <- data %>%
  left_join(data_90d, by = c("user_id", "product_id")) %>%
  mutate( p90d_up_order_cnt_ratio_inall = p90d_up_orders / user_order_cnt ) %>%
  mutate( p90d_u_order_cnt_ratio_inall = p90d_user_orders / user_order_cnt )
  
data_10 <- data[data$user_id <= 10, ]

rm(data_90d, orders_90d, users_90d, data_10)
gc()






############# !!! Place to load data environment #################


# Clean memory, Remove Tables ----------------------------------
rm(orders_prod_prior_diff, data_op_lag1_diff)
rm(data_up, users_prior, orders_prod_prior, orders_prod_test)
rm(orderp, ordert, orders)
rm(datanew, test, train, submission, subtrain, missing)
rm(products, prod_features)

gc()



#### Setup datanew with new variables --------------------------------

datanew <- data.table(
  user_id = data$user_id,
  product_id = data$product_id,
  order_id = data$order_id,

  # apply 9 up features  
  up_order_cnt_after_last= data$up_order_cnt_after_last,
  up_order_rate= data$up_order_rate,
  up_order_rate_since_fist= data$up_order_rate_since_fist,
  up_days_lag1_diff_median= data$up_days_lag1_diff_median,
  up_ordercount_ratio= data$up_ordercount_ratio,
  up_days_cnt_after_last= data$up_days_cnt_after_last,
  up_daycount_ratio= data$up_daycount_ratio,
  up_days_lag1_diff_mean= data$up_days_lag1_diff_mean,
  up_reorder_cnt= data$up_reorder_cnt,

  
  # apply 5 p90d features
  p90d_up_orders = data$p90d_up_orders,
  p90d_user_orders = data$p90d_user_orders,
  p90d_up_order_rate = data$p90d_up_order_rate,
  p90d_up_order_cnt_ratio_inall = data$p90d_up_order_cnt_ratio_inall,
  p90d_u_order_cnt_ratio_inall = data$p90d_u_order_cnt_ratio_inall , 

  # apply 6 prod features
  p_user_cnt = data$p_user_cnt, 
  p_atco_mean = data$p_atco_mean,
  p_atco_median  = data$p_atco_median, 
  p_unique_user_cnt = data$p_unique_user_cnt,
  p_unique_reorder_user_ratio = data$p_unique_reorder_user_ratio,
  p_reorder_user_rate = data$p_reorder_user_rate,


  eval_set = data$eval_set,
  reordered = data$reordered
  ) 

#rm(data)
gc()


# Train / Test datasets ---------------------------------------------------

datanew <- datanew[eval_set == "train", ]

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
test$user_id <- NULL




#### Original Parameter #########################

set.seed(1)
subtrain <- train %>% sample_frac(0.1)

#Somehow it doesn't work here 
#subtrain <- train %>% sample_frac(0.1)
#subtrain <- train[sample(nrow(train)/10),]

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

pred_threshold = 0.20

test[] <- lapply(test, as.numeric)
X <- xgb.DMatrix(as.matrix(test %>% select(  -order_id, -product_id, -reordered)))


# #### Get result by fixed threshold 
test$pred_reordered <- predict(model, X)
test$pred_reordered <- ( test$pred_reordered > pred_threshold) * 1
f1 =calc_submit_f1_noPred( test )

# 
# 
# #### Predict and Calcualte F1 Score 
# for(i in 17:30)
# {
#   
#   acc_threshold = as.numeric(i/100)
#   f1 =calc_submit_f1(acc_threshold, test, model, X)
#   
#   if (best_f1 < f1) {
#     best_f1 = f1
#     best_threshold = acc_threshold
#   }
# }
# print(paste("best_threshold:", best_threshold, "best_f1 is:", best_f1 ))  
# 






######################### !!! Submit TO LB##########################################

# Train / Test datasets ---------------------------------------------------
rm(orders_prod_prior_diff, data_op_lag1_diff)
rm(data_up, users_prior, orders_prod_prior, orders_prod_test)
rm(orderp, ordert, orders)
gc()


# redefine train and test ----------------------------------

train <- as.data.frame(datanew[datanew$eval_set == "train",])
test <- as.data.frame(datanew[datanew$eval_set == "test",])



train$eval_set <- NULL
train$user_id <- NULL
train$product_id <- NULL
train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0


test$eval_set <- NULL
test$user_id <- NULL



# Model Original Para -------------------------------------------------------------------

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

acc_threshold <- 0.20

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
write.csv(submission, file = "C:/DEV/MarketBusket/Submit/submit_ph2_try54_th20.csv", row.names = F)









