

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
  
  #acc_th = 0.2
  #t = test
  #m = model
  #xgbM = X
  Add_None_Threshold_1p = 0.5
  Add_None_Threshold_2p = 0.41
  Add_None_Threshold_3p = 0.32
  
  
  t$pred_reordered_org <- predict(m, xgbM)
  
  t$pred_reordered <- (t$pred_reordered_org > acc_th) * 1
  
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
  
  
  #### Add single produdct submission with None 
  t_single_prod <- t %>%
    group_by(order_id) %>%
    mutate(prodcnt = sum(pred_reordered)) %>%
    filter(prodcnt == 1 & pred_reordered == 1 & pred_reordered_org <= Add_None_Threshold_1p)
  
  submission_singlewithNone <- submission[submission$order_id %in% t_single_prod$order_id,] %>%
    group_by(order_id) %>%
    summarise(
      products = paste( products, "None", collapse = " ")
    )
  
  submission_new <- submission[! submission$order_id %in% submission_singlewithNone$order_id,] 
  submission <- submission_new %>% bind_rows(submission_singlewithNone) %>% arrange(order_id)
  
  
  #### Add None to 2 products submit 
  t_2_prod <- t %>%
    group_by(order_id) %>%
    mutate(prodcnt = sum(pred_reordered)) %>%
    filter(prodcnt == 2 & pred_reordered == 1 ) %>%
    mutate (pred_mean = mean(pred_reordered_org)) %>%
    filter(pred_mean <= Add_None_Threshold_2p )
  
  submission_2p_withNone <- submission[submission$order_id %in% t_2_prod$order_id,] %>%
    group_by(order_id) %>%
    summarise(
      products = paste( products, "None", collapse = " ")
    )
  
  
  submission_new <- submission[! submission$order_id %in% submission_2p_withNone$order_id,] 
  submission <- submission_new %>% bind_rows(submission_2p_withNone) %>% arrange(order_id)
  
  
  #### Add None to 3 products submit 
  t_3_prod <- t %>%
    group_by(order_id) %>%
    mutate(prodcnt = sum(pred_reordered)) %>%
    filter(prodcnt == 3 & pred_reordered == 1 ) %>%
    mutate (pred_mean = mean(pred_reordered_org)) %>%
    filter(pred_mean <= Add_None_Threshold_3p )
  
  submission_3p_withNone <- submission[submission$order_id %in% t_3_prod$order_id,] %>%
    group_by(order_id) %>%
    summarise(
      products = paste( products, "None", collapse = " ")
    )
  
  
  submission_new <- submission[! submission$order_id %in% submission_3p_withNone$order_id,] 
  submission <- submission_new %>% bind_rows(submission_3p_withNone) %>% arrange(order_id)
  
  
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
    #"fact: ",nrow(fact),
    "pred rows: ", nrow(t[pred_reordered == 1,]),
    #"fact_missing: ",nrow(fact_missing),
    "missing: ",nrow(missing),
    "sub_fact: ",nrow(submission_fact) ))  
  return (calc_f1)
  
}

# Calcualte submit Average F Store 
calc_submit_f1_noPred <- function (t) {
  
  Best_Add_None_Threshold = 0.46
  
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
  
  
  #### Add single produdct submission with None 
  t_single_prod <- t %>%
    group_by(order_id) %>%
    mutate(prodcnt = sum(pred_reordered)) %>%
    filter(prodcnt == 1 & pred_reordered == 1 & pred_reordered_org <= Best_Add_None_Threshold)
  
  submission_singlewithNone <- submission[submission$order_id %in% t_single_prod$order_id,] %>%
    group_by(order_id) %>%
    summarise(
      products = paste( products, "None", collapse = " ")
    )
  
  submission_new <- submission[! submission$order_id %in% submission_singlewithNone$order_id,] 
  submission <- submission_new %>% bind_rows(submission_singlewithNone) %>% arrange(order_id)
  
  
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

#aisles <- fread(file.path(path, "aisles.csv"))
#departments <- fread(file.path(path, "departments.csv"))
orderp <- fread(file.path(path, "order_products__prior.csv"))
ordert <- fread(file.path(path, "order_products__train.csv"))
orders <- fread(file.path(path, "orders.csv"))
#products <- fread(file.path(path, "products.csv"))


#### Reshape data ------------------------------------------------------------

#aisles$aisle <- as.factor(aisles$aisle)
#departments$department <- as.factor(departments$department)
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

# Get busket Size
users_basket <- orders_prod_prior %>%
  group_by(user_id, order_id) %>%
  summarise(
    user_order_basket = n()
  ) %>%
  group_by(user_id) %>%
  summarise(
    u_basket_mean = mean(user_order_basket)
  )

############### Find average basket Size #################

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

# Get busket Size
users_basket <- orders_prod_prior %>%
  group_by(user_id, order_id) %>%
  summarise(
    user_order_basket = n()
  ) %>%
  group_by(user_id) %>%
  summarise(
    u_basket_mean = mean(user_order_basket)
  )




############# !!! Place to load data environment #################

load("C:/DEV/GitHub/PZ1311_MB/XGB_Ph2_UP&P90D&P.RData")



############ Try to Add OrderSteak Feature ###############
OSLoad_path <- "C:/DEV/MarketBusket/OtherUsers/OrderSteakFeature"
order_steak <- fread(file.path(OSLoad_path, "order_streaks.csv"))

data <- data %>%
  left_join(order_steak, by = c("user_id", "product_id"))

rm(order_steak)
gc()


# Clean memory, Remove Tables ----------------------------------
rm(orders_prod_prior_diff, data_op_lag1_diff)
rm(data_up, users_prior, orders_prod_prior, orders_prod_test)
rm(orderp, ordert, orders)
rm(datanew, test, train, submission, subtrain, missing)
rm(products, prod_features)

rm(df, fact, fact_missing, submission, missing, 
   submission_singlewithNone, submission_fact, submission_new, 
   t, t_single_prod, test_single_prod, test_pred, 
   t_2_prod, t_3_prod, submission_2p_withNone, submission_3p_withNone,
   t_4_prod, submission_4p_withNone,)

rm(datanew, test,train,user_basket,subtrain, importance, users_basket)

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
  p_unique_reorder_user_rate = data$p_unique_reorder_user_rate,
  p_reorder_user_rate = data$p_reorder_user_rate,
  
  order_streak = data$order_streak,
  
  eval_set = data$eval_set,
  reordered = data$reordered
) 

#rm(data)
gc()



########################### !!! Submit TO LB   ##################################

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
#test$user_id <- NULL



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
#subtrain <- train
X <- xgb.DMatrix(as.matrix(subtrain %>% select(-reordered)), label = subtrain$reordered)
model <- xgboost(data = X, params = params, nrounds = 80)

importance <- xgb.importance(colnames(X), model = model)
xgb.ggplot.importance(importance)

#rm(X, importance, subtrain)
#gc()


# Apply model -------------------------------------------------------------
X <- xgb.DMatrix(as.matrix(test %>% select(-user_id, -order_id, -product_id)))



pred_submit_threshold = 0.16

test$pred_reordered_org <- predict(model, X)
test$pred_reordered <- NULL
test$pred_reordered <- (test$pred_reordered_org > pred_submit_threshold) * 1

test_pred <- test[,c("user_id","order_id","product_id", "pred_reordered", "pred_reordered_org", "reordered")]


############ Try to Add OrderSteak Feature ###############
basket_Load_path <- "C:/DEV/MarketBusket/Submit"
pred_busket <- fread(file.path(basket_Load_path, "estimate_basket_size_V01.csv"))


pred_busket$est_basketsize <- round(pred_busket$pred)

t <- test_pred %>%
  filter(pred_reordered == 1) %>%
  inner_join(pred_busket, by = "user_id") %>%
  arrange(user_id,  desc(pred_reordered_org)) %>%
  group_by(order_id) %>%
  mutate(pred_rank = row_number())  %>%
  filter(pred_rank < pred)

# #### By Average basket size 
# t <- test_pred %>%
#   filter(pred_reordered == 1) %>%
#   inner_join(users_basket, by = "user_id") %>%
#   arrange(user_id,  desc(pred_reordered_org)) %>%
#   group_by(order_id) %>%
#   mutate(pred_rank = row_number())  %>%
#   filter(pred_rank < (u_basket_mean + 1 ))


 test_pred <- test[,c("user_id","order_id","product_id", "pred_reordered", "pred_reordered_org", "reordered")]
 

submission <- t %>%
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

#### Add None to single produdct submission 

Add_None_Threshold_1p = 0.5
Add_None_Threshold_2p = 0.41
Add_None_Threshold_3p = 0.32


test_single_prod <- t %>%
  group_by(order_id) %>%
  mutate(prodcnt = sum(pred_reordered)) %>%
  filter(prodcnt == 1 & pred_reordered == 1 & pred_reordered_org <= Add_None_Threshold_1p)

submission_singlewithNone <- submission[submission$order_id %in% test_single_prod$order_id,] %>%
  group_by(order_id) %>%
  summarise(
    products = paste( products, "None", collapse = " ")
  )

submission_new <- submission[! submission$order_id %in% submission_singlewithNone$order_id,] 
submission <- submission_new %>% bind_rows(submission_singlewithNone) %>% arrange(order_id)

#### Add None to 2 products submit 
t_2_prod <- t %>%
  group_by(order_id) %>%
  mutate(prodcnt = sum(pred_reordered)) %>%
  filter(prodcnt == 2 & pred_reordered == 1 ) %>%
  mutate (pred_mean = mean(pred_reordered_org)) %>%
  filter(pred_mean <= Add_None_Threshold_2p )

submission_2p_withNone <- submission[submission$order_id %in% t_2_prod$order_id,] %>%
  group_by(order_id) %>%
  summarise(
    products = paste( products, "None", collapse = " ")
  )

submission_new <- submission[! submission$order_id %in% submission_2p_withNone$order_id,] 
submission <- submission_new %>% bind_rows(submission_2p_withNone) %>% arrange(order_id)


#### Add None to 3 products submit 
t_3_prod <- t %>%
  group_by(order_id) %>%
  mutate(prodcnt = sum(pred_reordered)) %>%
  filter(prodcnt == 3 & pred_reordered == 1 ) %>%
  mutate (pred_mean = mean(pred_reordered_org)) %>%
  filter(pred_mean <= Add_None_Threshold_3p )

submission_3p_withNone <- submission[submission$order_id %in% t_3_prod$order_id,] %>%
  group_by(order_id) %>%
  summarise(
    products = paste( products, "None", collapse = " ")
  )


submission_new <- submission[! submission$order_id %in% submission_3p_withNone$order_id,] 
submission <- submission_new %>% bind_rows(submission_3p_withNone) %>% arrange(order_id)


## When prediction is less than 0.25, submit to None Only, no help
# submission[submission$order_id %in% test_single_prod[test_single_prod$pred_reordered_org <= 0.25, ]$order_id,]$products = "None"

write.csv(submission, file = "C:/DEV/MarketBusket/Submit/submit_ph2_try69_th17_Basket_estsizeroundup_None1s2s3s.csv", row.names = F)


