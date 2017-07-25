
###########################################################################################################
#
# Kaggle Instacart competition
# Paul Zhang, July 2017
# Phase 2, on xgboost starter, start score as 0.3818, submit 42
# Try to add all the variables based on artical of "Repeat Buyer Prediction for E-Commerce" 
#
###########################################################################################################

#options(warn=-1)
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
  print(paste(
    " acc_th is:", acc_th, 
    " f1 Score is:", calc_f1, 
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
#orders_org <- orders
#orders <- orders_org[user_id == 2 , ]
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








############# !!! Place to load data environment #################

#### Features on Users -----------------------------------------------
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
  
# Add user Features 
data <- data %>% 
  left_join(users , by=("user_id")) 

# rm(orders_prod_prior_diff, data_op_lag1_diff)
# rm(data_up, users_prior, orders_prod_prior, orders_prod_test)
# rm(orderp, ordert, orders)
# gc()

#### Features on recent activies ###########################################


# Products ----------------------------------------------------------------


# Users -------------------------------------------------------------------


# PZ, redefine train and test ----------------------------------
rm(orders_prod_prior_diff, data_op_lag1_diff)
rm(data_up, users_prior, orders_prod_prior, orders_prod_test)
rm(orderp, ordert, orders)
rm(datanew, test, train, submission, users,subtrain)
gc()


#### t03 ####################################################################
# Need specify the column in datanew, otherwise subset fail on sample_flac
datanew <- data.table(
  user_id	 = data$user_id	,
  product_id	 = data$product_id	,
#  up_order_max	 = data$up_order_max	,
  up_order_min	 = data$up_order_min	,
  up_reorder_cnt	 = data$up_reorder_cnt	,
  up_days_max	 = data$up_days_max	,
#  up_dow_mean	 = data$up_dow_mean	,
#  up_hod_mean	 = data$up_hod_mean	,
  up_atco_mean	 = data$up_atco_mean	,
  u_order_cnt	 = data$u_order_cnt	,
  u_days_cnt	 = data$u_days_cnt	,
#  up_order_lag1_diff_max	 = data$up_order_lag1_diff_max	,
  up_order_lag1_diff_mean	 = data$up_order_lag1_diff_mean	,
  up_order_lag1_diff_median	 = data$up_order_lag1_diff_median	,
#  up_days_lag1_diff_max	 = data$up_days_lag1_diff_max	,
#  up_days_lag1_diff_mean	 = data$up_days_lag1_diff_mean	,
#  up_days_lag1_diff_median	 = data$up_days_lag1_diff_median	,
  up_order_rate	 = data$up_order_rate	,
  up_order_rate_since_fist	 = data$up_order_rate_since_fist	,
  up_order_cnt_after_last	 = data$up_order_cnt_after_last	,
  up_days_cnt_after_last	 = data$up_days_cnt_after_last	,
#  up_dow_ratio	 = data$up_dow_ratio	,
  up_hod_ratio	 = data$up_hod_ratio	,
  up_ordercount_ratio	 = data$up_ordercount_ratio	,
  up_daycount_ratio	 = data$up_daycount_ratio	,
  reordered	 = data$reordered	,
  eval_set	 = data$eval_set	,
  order_id	 = data$order_id	,
#  u_prod_cnt	 = data$u_prod_cnt	,
#  u_reorder_prod_cnt	 = data$u_reorder_prod_cnt	,
  u_reorder_prod_ratio	 = data$u_reorder_prod_ratio	,
  u_dow_mean	 = data$u_dow_mean	,
#  u_dow_sd	 = data$u_dow_sd	,
  u_hod_mean	 = data$u_hod_mean	,
#  u_hod_sd	 = data$u_hod_sd	,
  u_dspo_mean	 = data$u_dspo_mean	,
  u_dspo_sd	 = data$u_dspo_sd	,
  u_dow_ratio	 = data$u_dow_ratio	,
  u_hod_ratio	 = data$u_hod_ratio	,
  u_dspo_ratio	 = data$u_dspo_ratio	,
  u_basket_mean	 = data$u_basket_mean	,
#  u_basket_median	 = data$u_basket_median	,
#  u_basket_max	 = data$u_basket_max	,
  u_basket_sd	 = data$u_basket_sd	,
  u_basket_reorder_ratio_mean	 = data$u_basket_reorder_ratio_mean	,
  u_basket_reorder_ratio_sd	 = data$u_basket_reorder_ratio_sd	,
  u_unique_prod_cnt	 = data$u_unique_prod_cnt	,
#  u_unique_reorder_prod_cnt	 = data$u_unique_reorder_prod_cnt	,
  u_unique_reorder_prod_ratio	 = data$u_unique_reorder_prod_ratio
)


#### t04 ####################################################################
# Need specify the column in datanew, otherwise subset fail on sample_flac
datanew <- data.table(
  user_id	 = data$user_id	,
  product_id	 = data$product_id	,
  #  up_order_max	 = data$up_order_max	,
  up_order_min	 = data$up_order_min	,
  up_reorder_cnt	 = data$up_reorder_cnt	,
  up_days_max	 = data$up_days_max	,
  #up_dow_mean	 = data$up_dow_mean	,
  #up_hod_mean	 = data$up_hod_mean	,
  up_atco_mean	 = data$up_atco_mean	,
  u_order_cnt	 = data$u_order_cnt	,
  u_days_cnt	 = data$u_days_cnt	,
  #  up_order_lag1_diff_max	 = data$up_order_lag1_diff_max	,
  up_order_lag1_diff_mean	 = data$up_order_lag1_diff_mean	,
  # up_order_lag1_diff_median	 = data$up_order_lag1_diff_median	,
  #  up_days_lag1_diff_max	 = data$up_days_lag1_diff_max	,
  up_days_lag1_diff_mean	 = data$up_days_lag1_diff_mean	,
  up_days_lag1_diff_median	 = data$up_days_lag1_diff_median	,
  up_order_rate	 = data$up_order_rate	,
  up_order_rate_since_fist	 = data$up_order_rate_since_fist	,
  up_order_cnt_after_last	 = data$up_order_cnt_after_last	,
  up_days_cnt_after_last	 = data$up_days_cnt_after_last	,
  up_dow_ratio	 = data$up_dow_ratio	,
  up_hod_ratio	 = data$up_hod_ratio	,
  up_ordercount_ratio	 = data$up_ordercount_ratio	,
  up_daycount_ratio	 = data$up_daycount_ratio	,
  reordered	 = data$reordered	,
  eval_set	 = data$eval_set	,
  order_id	 = data$order_id	,
  #  u_prod_cnt	 = data$u_prod_cnt	,
  #  u_reorder_prod_cnt	 = data$u_reorder_prod_cnt	,
  u_reorder_prod_ratio	 = data$u_reorder_prod_ratio	,
  u_dow_mean	 = data$u_dow_mean	,
  u_dow_sd	 = data$u_dow_sd	,
  u_hod_mean	 = data$u_hod_mean	,
  u_hod_sd	 = data$u_hod_sd	,
  u_dspo_mean	 = data$u_dspo_mean	,
  u_dspo_sd	 = data$u_dspo_sd	,
  u_dow_ratio	 = data$u_dow_ratio	,
  u_hod_ratio	 = data$u_hod_ratio	,
  u_dspo_ratio	 = data$u_dspo_ratio	,
  u_basket_mean	 = data$u_basket_mean	,
  #  u_basket_median	 = data$u_basket_median	,
  #  u_basket_max	 = data$u_basket_max	,
  u_basket_sd	 = data$u_basket_sd	,
  u_basket_reorder_ratio_mean	 = data$u_basket_reorder_ratio_mean	,
  u_basket_reorder_ratio_sd	 = data$u_basket_reorder_ratio_sd	,
  u_unique_prod_cnt	 = data$u_unique_prod_cnt	,
  #  u_unique_reorder_prod_cnt	 = data$u_unique_reorder_prod_cnt	,
  u_unique_reorder_prod_ratio	 = data$u_unique_reorder_prod_ratio
)




#### t05 ####################################################################
# UP Var only 
datanew <- data.table(
  user_id	 = data$user_id	,
  product_id	 = data$product_id	,
  up_order_max	 = data$up_order_max	,
  up_order_min	 = data$up_order_min	,
  up_reorder_cnt	 = data$up_reorder_cnt	,
  up_days_max	 = data$up_days_max	,
  up_dow_mean	 = data$up_dow_mean	,
  up_hod_mean	 = data$up_hod_mean	,
  up_atco_mean	 = data$up_atco_mean	,
  u_order_cnt	 = data$u_order_cnt	,
  u_days_cnt	 = data$u_days_cnt	,
  up_order_lag1_diff_max	 = data$up_order_lag1_diff_max	,
  up_order_lag1_diff_mean	 = data$up_order_lag1_diff_mean	,
  up_order_lag1_diff_median	 = data$up_order_lag1_diff_median	,
  up_days_lag1_diff_max	 = data$up_days_lag1_diff_max	,
  up_days_lag1_diff_mean	 = data$up_days_lag1_diff_mean	,
  up_days_lag1_diff_median	 = data$up_days_lag1_diff_median	,
  up_order_rate	 = data$up_order_rate	,
  up_order_rate_since_fist	 = data$up_order_rate_since_fist	,
  up_order_cnt_after_last	 = data$up_order_cnt_after_last	,
  up_days_cnt_after_last	 = data$up_days_cnt_after_last	,
  up_dow_ratio	 = data$up_dow_ratio	,
  up_hod_ratio	 = data$up_hod_ratio	,
  up_ordercount_ratio	 = data$up_ordercount_ratio	,
  up_daycount_ratio	 = data$up_daycount_ratio	,
  reordered	 = data$reordered	,
  eval_set	 = data$eval_set	,
  order_id	 = data$order_id	
  #  u_prod_cnt	 = data$u_prod_cnt	,
  #  u_reorder_prod_cnt	 = data$u_reorder_prod_cnt	,
  #u_reorder_prod_ratio	 = data$u_reorder_prod_ratio	,
  #u_dow_mean	 = data$u_dow_mean	,
  #u_dow_sd	 = data$u_dow_sd	,
  #u_hod_mean	 = data$u_hod_mean	,
  #u_hod_sd	 = data$u_hod_sd	,
  #u_dspo_mean	 = data$u_dspo_mean	,
  #u_dspo_sd	 = data$u_dspo_sd	,
  #u_dow_ratio	 = data$u_dow_ratio	,
  #u_hod_ratio	 = data$u_hod_ratio	,
  #u_dspo_ratio	 = data$u_dspo_ratio	,
  #u_basket_mean	 = data$u_basket_mean	,
  #  u_basket_median	 = data$u_basket_median	,
  #  u_basket_max	 = data$u_basket_max	,
  #u_basket_sd	 = data$u_basket_sd	,
  #u_basket_reorder_ratio_mean	 = data$u_basket_reorder_ratio_mean	,
  #u_basket_reorder_ratio_sd	 = data$u_basket_reorder_ratio_sd	,
  #u_unique_prod_cnt	 = data$u_unique_prod_cnt	,
  #  u_unique_reorder_prod_cnt	 = data$u_unique_reorder_prod_cnt	,
  #u_unique_reorder_prod_ratio	 = data$u_unique_reorder_prod_ratio
)

data_10 <- data[data$user_id <= 10, ]
rm(data, data_10)
gc()

datanew <- datanew[datanew$eval_set == "train", ]



# Train / Test datasets ---------------------------------------------------
set.seed(101) 
userlist = unique(datanew$user_id)
sample <- sample.int(n = length(userlist), 
                     size = floor(.75*length(userlist)), 
                     replace = F)

train <- datanew[datanew$user_id %in% userlist[sample],]
test  <- datanew[!datanew$user_id %in% userlist[sample],]

rm(datanew)
gc()


train$eval_set <- NULL
train$user_id <- NULL
train$product_id <- NULL
train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0

test$eval_set <- NULL
#test$user_id <- NULL


## PZ, Model -------------------------------------------------------------------
set.seed(1)
subtrain <- train %>% sample_frac(0.1)
# Somehow it doesn't work here, before redefine datanew
#subtrain <- train[sample(nrow(train)/10),]

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




### PZ, Tune the XGB Parameter ############


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


set.seed(1)
subtrain <- train %>% sample_frac(0.05)

X <- xgb.DMatrix(as.matrix(subtrain %>% select(-reordered)), label = subtrain$reordered)

best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0

for (iter in 1:50) {

    iter = 1
    param <- list(objective = "binary:logistic",
                  eval_metric = "logloss",
                  max_depth = sample(6:10, 1),
                  eta = runif(1, .01, .3),
                  gamma = runif(1, 0.0, 0.2),
                  subsample = runif(1, .6, .9),
                  colsample_bytree = runif(1, .5, .8),
                  min_child_weight = sample(1:40, 1),
                  max_delta_step = sample(1:10, 1)
    )


    cv.nround = 500
    cv.nfold = 5
    seed.number = sample.int(10000, 1)[[1]]
    set.seed(seed.number)
    mdcv <- xgb.cv(data=X, params = param, nthread=6,
                   nfold=cv.nfold, nrounds=cv.nround,
                   verbose = T, early.stop.round=8, maximize=FALSE)


  #min_logloss = min(mdcv[, test.mlogloss.mean])
  #min_logloss_index = which.min(mdcv[, test.mlogloss.mean])

  min_logloss = min(mdcv$evaluation_log$test_logloss_mean)
  min_logloss_index = which.min(mdcv$evaluation_log$test_logloss_mean)

  if (min_logloss < best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }
}


# > best_logloss
  # [1] 0.2472278
# > best_logloss_index
  # [1] 223
# > best_seednumber
  # [1] 4910
# > best_param
  # $objective
  # [1] "binary:logistic"
  # $eval_metric
  # [1] "logloss"
  # $max_depth
  # [1] 6
  # $eta
  # [1] 0.04289444
  # $gamma
  # [1] 0.1860173
  # $subsample
  # [1] 0.7954664
  # $colsample_bytree
  # [1] 0.7825535
  # $min_child_weight
  # [1] 16
  # $max_delta_step
  # [1] 4
# > min_logloss_index
  # [1] 195



