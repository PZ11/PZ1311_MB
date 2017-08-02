options(warn=-1)
#options(warn=0)

library(data.table)
library(dplyr)
library(tidyr)
library(xgboost)
#library(matrix)
library(zoo)



# Windows Load Data ---------------------------------------------------------------
path <- "C:/DEV/MarketBusket/Data_ORG/"

#Mac Load Data
#path <- "/Users/yuepengzhang/Documents/Kaggle/MarketBasket/InputORG"

# aisles <- fread(file.path(path, "aisles.csv"))
# departments <- fread(file.path(path, "departments.csv"))
orderp <- fread(file.path(path, "order_products__prior.csv"))
ordert <- fread(file.path(path, "order_products__train.csv"))
orders <- fread(file.path(path, "orders.csv"))

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


opp_10 <- orders_prod_prior[orders_prod_prior$user_id==10,]
o_10 <- orders[orders$user_id==10,]
opt_10 <- orders_prod_test[orders_prod_test$user_id==10,]


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
























