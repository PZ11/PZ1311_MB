

####
#
# Kaggle Instacart competition
# Paul Zhang, July 2017
# Phase 2, on xgboost starter, start score as 0.3818, submit 42
# Try to add all the variables based on artical of "Repeat Buyer Prediction for E-Commerce" 
#
####

options(warn=-1)
#options(warn=0)

library(data.table)
library(dplyr)
library(tidyr)
#library(xgboost)
#library(matrix)
library(zoo)

library(lightgbm)
library(Matrix)

# Author: Faron, Lukasz Grad
#
# Quite fast implementation of Faron's expected F1 maximization using Rcpp and R
library(inline)
library(Rcpp)
Sys.setenv("PKG_CXXFLAGS"="-std=c++11")


##### CPP Function ###################################### 
# Input: p: item reorder probabilities (sorted), p_none: none probability (0 if not specified)
# Output: matrix[2][n + 1] out: out[0][j] - F1 score with top j products and None
#                               out[1][j] - F1 score with top j products
cppFunction(
  'NumericMatrix get_expectations(NumericVector p, double p_none) {
  // Assuming p is sorted, p_none == 0 if not specified
  int n = p.size();
  NumericMatrix expectations = NumericMatrix(2, n + 1);
  double DP_C[n + 2][n + 1];
  std::fill(DP_C[0], DP_C[0] + (n + 2) * (n + 1), 0);
  if (p_none == 0.0) {
  p_none = std::accumulate(p.begin(), p.end(), 1.0, [](double &a, double &b) {return a * (1.0 - b);});
  }
  DP_C[0][0] = 1.0;
  for (int j = 1; j < n; ++j)
  DP_C[0][j] = (1.0 - p[j - 1]) * DP_C[0][j - 1];
  for (int i = 1; i < n + 1; ++i) {
  DP_C[i][i] = DP_C[i - 1][i - 1] * p[i - 1];
  for (int j = i + 1; j < n + 1; ++j)
  DP_C[i][j] = p[j - 1] * DP_C[i - 1][j - 1] + (1.0 - p[j - 1]) * DP_C[i][j - 1];
  }
  double DP_S[2 * n + 1];
  double DP_SNone[2 * n + 1];
  for (int i = 1; i < (2 * n + 1); ++i) {
  DP_S[i] = 1.0 / (1.0 * i);
  DP_SNone[i] = 1.0 / (1.0 * i + 1);
  }
  for (int k = n; k >= 0; --k) {
  double f1 = 0.0;
  double f1None = 0.0;
  for (int k1 = 0; k1 < (n + 1); ++k1) {
  f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1];
  f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1];
  }
  for (int i = 1; i < (2 * k - 1); ++i) {
  DP_S[i] = (1 - p[k - 1]) * DP_S[i] + p[k - 1] * DP_S[i + 1];
  DP_SNone[i] = (1 - p[k - 1]) * DP_SNone[i] + p[k - 1] * DP_SNone[i + 1];
  }
  expectations(0, k) = f1None + 2 * p_none / (2.0 + k);
  expectations(1, k) = f1;
  }
  return expectations;
  }'
)

# Input: ps - item reorder probabilities, prods - item ids
# Output: reordered items string (as required in submission)
exact_F1_max_none <- function(ps, prods) {
  prods <- sapply(prods, as.character)
  perm <- order(ps, decreasing = T)
  ps <- ps[perm]
  prods <- prods[perm]
  expectations <-  get_expectations(ps, 0.0)
  max_idx <-  which.max(expectations)
  add_none <- max_idx %% 2 == 1
  size <- as.integer(max(0, max_idx - 1) / 2)
  if (size == 0) {
    return("None")
  }
  else {
#    if (add_none)
#      return(paste(c(prods[1:size], "None"), collapse = " "))
#    else 
      return(paste(prods[1:size], collapse = " "))
  }
}


# How to use it with dplyr:
#
# submission <- data %>%
#        group_by(order_id) %>%
#        summarise(products = exact_F1_max_none(reordered_prob, product_id))

# Quick example
# exact_F1_max_none(c(0.5, 0.9, 0.8, 0.1, 0.2, 0.3), c(129832, 1024, 32, 432, 1421, 1032))


exact_F1_max_none_size <- function(ps, prods) {
  prods <- sapply(prods, as.character)
  perm <- order(ps, decreasing = T)
  ps <- ps[perm]
  prods <- prods[perm]
  expectations <-  get_expectations(ps, 0.0)
  max_idx <-  which.max(expectations)
  add_none <- max_idx %% 2 == 1
  size <- as.integer(max(0, max_idx - 1) / 2)
  return(size)
}

# R port of Faron's F1-Score Expectation Maximization
## F1 Expectation over increasing list of items
get_expectation <- function(P, pNone=NULL){
  
  expectations <- NULL
  
  P <- sort(P, decreasing = T)
  n <- length(P)
  
  DP_C = matrix(0, nrow = n+2, ncol=n+1)
  
  if (is.null(pNone)) pNone <- prod(1.0 - P)
  
  DP_C[1, 1] = 1.0
  
  for (j in 1:(n-1)) {
    DP_C[1, j+1] = (1.0 - P[j]) * DP_C[1, j]
  }
  
  for (i in 1:n) {
    DP_C[i+1, i+1] = DP_C[i, i] * P[i]
    if (i<n) {
      for (j in (i+1):n) {
        DP_C[i+1, j+1] = P[j] * DP_C[i, j] + (1 - P[j]) * DP_C[i+1, j]
      }
    }
  }
  
  DP_S = numeric(2 * n + 1)
  DP_SNone = numeric(2 * n + 1)
  
  for (i in 1:(2*n)){
    DP_S[i+1] = 1/i
    DP_SNone[i+1] = 1/(i+1)
  }
  
  for (k in (n+1):1) {
    f1 = 0
    f1None = 0
    for (k1 in 1:(n+1)) {
      f1 <- f1 + 2 * (k1-1) * DP_C[k1, k] * DP_S[k + k1 -1]
      f1None <- f1None + 2 * (k1-1) * DP_C[k1, k] * DP_SNone[k + k1 -1]
    }
    if (2*(k-1) - 2 > 1){
      for (i in 1:(2*(k-1) - 2)) {
        DP_S[i+1] = (1 - P[k-1]) * DP_S[i+1] + P[k-1] * DP_S[i + 2]
        DP_SNone[i+1] = (1 - P[k-1]) * DP_SNone[i+1] + P[k-1] * DP_SNone[i + 2]
      }
    }
    expectations <- rbind(expectations, c(f1None + 2 * pNone / (1 + k), f1))
  }
  
  return(t(expectations[nrow(expectations):1, ]))
  
}

## Return best scenario
maximize_expectation <- function(P, pNone=NULL){
  
  expectations <- get_expectation(P, pNone)
  opt_coords <- which(expectations == max(expectations), arr.ind = TRUE)
  
  max_f1 <- expectations[opt_coords]
  predNone <- as.logical(opt_coords[1,1] == 1)
  best_k <- as.integer(opt_coords[1,2])-1
  
  return(list(best_k=best_k, predNone=predNone, max_f1=max_f1))
}



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
  
  # acc_th = 0.2
  # t = test
  # m = model
  # xgbM = X
  
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
    # "pred rows: ", nrow(t[pred_reordered == 1,]),
    #"fact_missing: ",nrow(fact_missing),
    "missing: ",nrow(missing),
    "sub_fact: ",nrow(submission_fact) ))  
  return (calc_f1)
  
}

# Fixed Threshold, Calcualte submit Average F Store 
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

# Use Max F1, Calcualte submit Average F Store 
calc_submit_MaxF1 <- function (n_th,test_pred, submission_maxf1, submission_maxf1_size) {
  
  #n_th = none_threshold 
  #t = test

  Add_None_Threshold_1p = 0.50
  Add_None_Threshold_2p = 0.41
  Add_None_Threshold_3p = 0.32
  

  
  # Try gregexpr to find size, did not use it. 
  #  length(as.vector(gregexpr(" ", as.character(submission[submission$order_id == 3251761,  "products"]))[[1]]))
  #  length((gregexpr(" ", (submission[45,  "products"]))[[1]]))


  # tmp <-  submission_maxf1_size %>%
  #   group_by(submit_products_size) %>%
  #   summarise(prodcnt = n())
  
  submission <- submission_maxf1
  
  #### Add single product submission with None 
  t_1_prod <- test_pred %>%
    inner_join(submission_maxf1_size, by = "order_id" ) %>%
    filter (submit_products_size == 1) %>%
    filter (pred_rank == 1) %>%
    group_by(order_id) %>%
    filter( pred_reordered_org <= Add_None_Threshold_1p)

  
  submission_singlewithNone <- submission[submission$order_id %in% t_1_prod$order_id,] %>%
    group_by(order_id) %>%
    summarise(
      products = paste( products, "None", collapse = " ")
    )
  
  submission_new <- submission[! submission$order_id %in% submission_singlewithNone$order_id,] 
  submission <- submission_new %>% bind_rows(submission_singlewithNone) %>% arrange(order_id)
  
  
  #### Add None to 2 products submit
  t_2_prod <- test_pred %>%
    inner_join(submission_maxf1_size, by = "order_id" ) %>%
    filter (submit_products_size == 2) %>%
    filter (pred_rank <= 2) %>%
    group_by(order_id) %>%
    mutate (pred_mean = mean(pred_reordered_org)) %>%
    filter(pred_mean <= Add_None_Threshold_2p )

  submission_2p_withNone <- submission[submission$order_id %in% t_2_prod$order_id,] %>%
    group_by(order_id) %>%
    summarise(
      products = paste( products, "None", collapse = " ")
    )


  submission_new <- submission[! submission$order_id %in% submission_2p_withNone$order_id,]
  submission <- submission_new %>% bind_rows(submission_2p_withNone) %>% arrange(order_id)


  # #### Add None to 3 products submit
  t_3_prod <- test_pred %>%
    inner_join(submission_maxf1_size, by = "order_id" ) %>%
    filter (submit_products_size == 3) %>%
    filter (pred_rank <= 3) %>%
    group_by(order_id) %>%
    mutate (pred_mean = mean(pred_reordered_org)) %>%
    filter(pred_mean <= Add_None_Threshold_3p )

  submission_3p_withNone <- submission[submission$order_id %in% t_3_prod$order_id,] %>%
    group_by(order_id) %>%
    summarise(
      products = paste( products, "None", collapse = " ")
    )


  submission_new <- submission[! submission$order_id %in% submission_3p_withNone$order_id,]
  submission <- submission_new %>% bind_rows(submission_3p_withNone) %>% arrange(order_id)

  
  
  fact <- test_pred %>%
    filter(reordered == 1) %>%
    group_by(order_id) %>%
    summarise(
      fact_products = paste(product_id, collapse = " ")
    )
  
  fact_missing <- data.frame(
    order_id = unique(test_pred$order_id[!test_pred$order_id %in% fact$order_id]),
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
#    "n_th: ",n_th,
#    "missing: ",nrow(missing),
    "sub_fact: ",nrow(submission_fact) ))  
  return (calc_f1)
  
}

# Use Max F1, Calcualte submit Average F Store 
calc_MaxF1_Combined_Fixed <- function (t_pred, submission_maxf1) {
  
  # submission_maxf1 <- submaxf1
  # t_pred <- test_pred
  
  Add_None_Threshold_1p = 0.50
  Add_None_Threshold_2p = 0.41
  Add_None_Threshold_3p = 0.32
  
  
  pred_submit_threshold = 0.20
  t_pred$pred_reordered <- (t_pred$pred_reordered_org > pred_submit_threshold) * 1
  
  
  submission <- t_pred %>%
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
  
  
  
  ### Add None to single produdct submission 
  
  Add_None_Threshold_1p = 0.5
  Add_None_Threshold_2p = 0.41
  Add_None_Threshold_3p = 0.32
  
  
  test_single_prod <- t_pred %>%
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
  t_2_prod <- t_pred %>%
    group_by(order_id) %>%
    mutate(prodcnt = sum(pred_reordered)) %>%
    filter(prodcnt == 2 & pred_reordered == 1 ) %>%
    mutate (pred_mean = mean(pred_reordered_org)) %>%
    mutate (pred_max = max(pred_reordered_org)) %>%
    filter(pred_mean <= Add_None_Threshold_2p )
  
  submission_2p_withNone <- submission[submission$order_id %in% t_2_prod$order_id,] %>%
    group_by(order_id) %>%
    summarise(
      products = paste( products, "None", collapse = " ")
    )
  
  submission_new <- submission[! submission$order_id %in% submission_2p_withNone$order_id,] 
  submission <- submission_new %>% bind_rows(submission_2p_withNone) %>% arrange(order_id)
  
  
  #### Add None to 3 products submit 
  t_3_prod <- t_pred %>%
    group_by(order_id) %>%
    mutate(prodcnt = sum(pred_reordered)) %>%
    filter(prodcnt == 3 & pred_reordered == 1 ) %>%
    mutate (pred_mean = mean(pred_reordered_org)) %>%
    mutate (pred_max = max(pred_reordered_org)) %>%
    filter(pred_mean <= Add_None_Threshold_3p  )
  
  submission_3p_withNone <- submission[submission$order_id %in% t_3_prod$order_id,] %>%
    group_by(order_id) %>%
    summarise(
      products = paste( products, "None", collapse = " ")
    )
  
  
  submission_new <- submission[! submission$order_id %in% submission_3p_withNone$order_id,] 
  submission <- submission_new %>% bind_rows(submission_3p_withNone) %>% arrange(order_id)
  
  
  
  #### Add None to 4 products submit
  t_4_prod <- t_pred %>%
    group_by(order_id) %>%
    mutate(prodcnt = sum(pred_reordered)) %>%
    filter(prodcnt == 4 & pred_reordered == 1 ) %>%
    mutate (pred_mean = mean(pred_reordered_org)) %>%
    mutate (pred_max = max(pred_reordered_org)) %>%
    filter(pred_mean <= 0.25 & pred_max < 0.5 )
  
  submission_4p_withNone <- submission[submission$order_id %in% t_4_prod$order_id,] %>%
    group_by(order_id) %>%
    summarise(
      products = paste( products, "None", collapse = " ")
    )
  
  
  submission_new <- submission[! submission$order_id %in% submission_4p_withNone$order_id,]
  submission <- submission_new %>% bind_rows(submission_4p_withNone) %>% arrange(order_id)
  
  submission_FixRate_WithNone <- submission
  
  
  ####merge 1/2/3/4 result with MaxF, Best on Aug 4rd, 0.3962
  
  
  submission <- submission_maxf1
  
  submission_1p_withNone <- submission_singlewithNone
  submission_1p <- submission[! submission$order_id %in% submission_1p_withNone$order_id,]
  submission <- submission_1p %>% bind_rows(submission_1p_withNone) %>% arrange(order_id)
  
  submission_2p <- submission[! submission$order_id %in% submission_2p_withNone$order_id,]
  submission <- submission_2p %>% bind_rows(submission_2p_withNone) %>% arrange(order_id)
  
  submission_3p <- submission[! submission$order_id %in% submission_3p_withNone$order_id,]
  submission <- submission_3p %>% bind_rows(submission_3p_withNone) %>% arrange(order_id)
  
  submission_4p <- submission[! submission$order_id %in% submission_4p_withNone$order_id,]
  submission <- submission_4p %>% bind_rows(submission_4p_withNone) %>% arrange(order_id)
  
  
  
  fact <- t_pred %>%
    filter(reordered == 1) %>%
    group_by(order_id) %>%
    summarise(
      fact_products = paste(product_id, collapse = " ")
    )
  
  fact_missing <- data.frame(
    order_id = unique(t_pred$order_id[!t_pred$order_id %in% fact$order_id]),
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
    #    "n_th: ",n_th,
    #    "missing: ",nrow(missing),
    "sub_fact: ",nrow(submission_fact) ))  
  return (calc_f1)
  
}

###################################################################33
# opp_10 <- orders_prod_prior[orders_prod_prior$user_id==10,]
# o_10 <- orders[orders$user_id==10,]
# opt_10 <- orders_prod_test[orders_prod_test$user_id==10,]
# data_10 <- data[data$user_id <= 10, ]


# !!! Windows Load Data ---------------------------------------------------------------
path <- "C:/DEV/MarketBusket/Data_ORG/"
products <- fread(file.path(path, "products.csv"))

aisles <- fread(file.path(path, "aisles.csv"))
departments <- fread(file.path(path, "departments.csv"))


aisles$aisle <- as.factor(aisles$aisle)
departments$department <- as.factor(departments$department)
products$product_name <- as.factor(products$product_name)

products <- products %>% 
  inner_join(aisles) %>% inner_join(departments) %>% 
  select(-aisle_id, -department_id)

rm(aisles, departments)


############# !!! Place to load data environment #################

load("C:/DEV/GitHub/PZ1311_MB/XGB_Ph2_UP&P90D&P&ProdStreak&da&dow_20170802.RData")

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
  

#  p_reorder_cnt = data$p_reorder_cnt,
#  p_reorder_rate = data$p_reorder_rate,
#  p_dow_reorder_cnt = data$p_dow_reorder_cnt, 
#  p_dow_reorder_rate = data$ p_dow_reorder_rate, 

#  p_hod_reorder_rate = data$p_hod_reorder_rate,

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

  ud_reorder_ratio_in_u = data$ud_reorder_ratio_in_u,  
  ui_reorder_ratio_in_u = data$ui_reorder_ratio_in_u,  

  
  eval_set = data$eval_set,
  reordered = data$reordered
  ) 

#rm(data)
gc()


datanew <- datanew[eval_set == "train", ]



############ load LGB data  #################

#path <- "C:/DEV/MarketBusket/R_script_pz/Ph3_LGB/"
#data_lgb <- fread(file.path(path, "data_Submit.txt"))
#data_val <- fread(file.path(path, "data_val_Submit.txt"))

load("C:/DEV/GitHub/PZ1311_MB/LGB_PH3_DataLoad.RData")

#head_data_lgb <- data_lgb[1:100,]
#View(head_data_lgb)


column_names <-c( 'user_id',
                  'product_id',
                  'user_product_reordered_ratio',
                  'reordered_sum',
                  'add_to_cart_order_inverted_mean',
                  'add_to_cart_order_relative_mean',
                  'reorder_prob',
                  'last',
                  'prev1',
                  'prev2',
                  'median',
                  'mean',
                  'dep_reordered_ratio',
                  'aisle_reordered_ratio',
                  'aisle_products',
                  'aisle_reordered',
                  'dep_products',
                  'dep_reordered',
                  'prod_users_unq',
                  'prod_users_unq_reordered',
                  'order_number',
                  'prod_add_to_card_mean',
                  'days_since_prior_order',
                  'order_dow',
                  'order_hour_of_day',
                  'reorder_ration',
                  'user_orders',
                  'user_order_starts_at',
                  'user_mean_days_since_prior',
                  'user_average_basket',
                  'user_distinct_products',
                  'user_reorder_ratio',
                  'user_total_products',
                  'prod_orders',
                  'prod_reorders',
                  'up_order_rate',
                  'up_orders_since_last_order',
                  'up_order_rate_since_first_order',
                  'up_orders',
                  'up_first_order',
                  'up_last_order',
                  'up_mean_cart_position',
                  'days_since_prior_order_mean',
                  'order_dow_mean',
                  'order_hour_of_day_mean',
                  'V1',
                  'V2',
                  'V3',
                  'V4',
                  'V5',
                  'V6',
                  'V7',
                  'V8',
                  'V9',
                  'V10',
                  'V11',
                  'V12',
                  'V13',
                  'V14',
                  'V15',
                  'V16',
                  'V17',
                  'V18',
                  'V19',
                  'V20',
                  'V21',
                  'V22',
                  'V23',
                  'V24',
                  'V25',
                  'V26',
                  'V27',
                  'V28',
                  'V29',
                  'V30',
                  'V31',
                  'V32'

)
setnames(data_lgb, column_names)
setnames(data_val, column_names)


load("C:/DEV/GitHub/PZ1311_MB/XGB_Ph2_UP&P90D&P&ProdStreak&da&dow_20170802.RData")


# data_lgb_train <- data_lgb %>%
#   inner_join(data %>% select( user_id, product_id, order_id, reordered ), by = c('user_id', 'product_id'))
# 
# data_lgb_test <- data_val %>%
#   inner_join(data %>% select( user_id, product_id, order_id ), by = c('user_id', 'product_id'))


####### !!! Load LGB Environment ######## 

rm(data, data_lgb, data_val)
rm(test, train, importance)
gc()



load("C:/DEV/GitHub/PZ1311_MB/XGB_Ph2_UP&P90D&P&ProdStreak&da&dow_20170802.RData")
datanew <- data.table(
  user_id = data$user_id,
  product_id = data$product_id,
  
  # apply 9 up features  
  z_up_order_cnt_after_last= data$up_order_cnt_after_last,
  z_up_order_rate= data$up_order_rate,
  z_up_order_rate_since_fist= data$up_order_rate_since_fist,
  z_up_days_lag1_diff_median= data$up_days_lag1_diff_median,
  z_up_ordercount_ratio= data$up_ordercount_ratio,
  z_up_days_cnt_after_last= data$up_days_cnt_after_last,
  z_up_daycount_ratio= data$up_daycount_ratio,
  z_up_days_lag1_diff_mean= data$up_days_lag1_diff_mean,
  z_up_reorder_cnt= data$up_reorder_cnt,
  
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
  
  z_order_streak = data$order_streak,
  
  z_ud_reorder_ratio_in_u = data$ud_reorder_ratio_in_u,  
  z_ui_reorder_ratio_in_u = data$ui_reorder_ratio_in_u

) 

rm(data)
gc()




load("C:/DEV/GitHub/PZ1311_MB/LGB_Ph3_data_lgb_train_org.R.RData")

data_lgb_train_new <- data_lgb_train %>%
  inner_join(datanew, by = c('user_id', 'product_id'))

rm( data_lgb_train)
gc()





# Train / Test datasets ---------------------------------------------------

datanew <- data_lgb_train
datanew <- data_lgb_train_new

datanew <- data_lgb_train %>% 
  inner_join(products %>%  select( product_id, aisle_id, department_id), by="product_id")

#rm(data_lgb_train_new)
gc()


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

rm(datanew)
rm(data)
rm(data_lgb_train_new)
rm(data_lgb_train)
gc()



########################### Light GBM  ##################################

set.seed(1)
subtrain <- train %>% sample_frac(0.1)

#subtrain <- train

params <-  list(
  "task" = "train",
  "boosting_type" = "gbdt",
  "objective" =  "binary",
  "metric" = c("binary_logloss", "auc"),
  "num_leaves"= 256,
  "min_sum_hessian_in_leaf" = 20,
  "max_depth"= 12,
  "learning_rate"= 0.05,
  "feature_fraction"= 0.6,
  "verbose" = 1
)


dtrain <- subtrain %>% select(-reordered, -department_id, -aisle_id)
#X <- sparse.model.matrix(~.-1, data = dtrain)

X<- as.matrix(data.frame(dtrain))
dlable <- subtrain$reordered

#categories = c('aisle_id', 'department_id')

cat_features = subtrain[, 'department_id']

dtrain_set <- lgb.Dataset(data = X,
                      label = dlable,
                      categorical_feature=cat_features )

dtrain_set <- lgb.Dataset(data = X,
                          label = dlable )

lgbmodel <- lightgbm(data = dtrain_set,
                  params = params,
                  nrounds = 380, 
                  early_stopping_rounds=10
)

rm(X_TEST)
gc()



# Apply model -------------------------------------------------------------

X_TEST <- as.matrix(test %>% select( -user_id, -order_id, -product_id, -reordered))
pred <- predict(lgbmodel, X_TEST)

test$pred_reordered_org <- pred
#test_pred <- test[,c("order_id", "product_id", "pred_reordered_org" , "reordered")]


#Get result by fixed threshold
pred_threshold = 0.18

test$pred_reordered <- ( test$pred_reordered_org > pred_threshold) * 1
f1 =calc_submit_f1_noPred( test )




##### Original Parameter on XGB #########################

set.seed(1)
subtrain <- train %>% sample_frac(0.1)

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
xgbmodel <- xgboost(data = X, params = params, nrounds = 80)

importance <- xgb.importance(colnames(X), model = xgbmodel)
xgb.ggplot.importance(importance)


# Apply model -------------------------------------------------------------

test[] <- lapply(test, as.numeric)
X <- xgb.DMatrix(as.matrix(test %>% select( -user_id, -order_id, -product_id, -reordered)))


head_test <- test[1:100,]
View(head_test)

##### Find Best Fixed Threshold on F1 Score ###############


test$pred_reordered_org <- predict(xgbmodel, X)
test_pred <- test[,c("order_id", "product_id", "pred_reordered_org" , "reordered")]

#Get result by fixed threshold
pred_threshold = 0.20

test$pred_reordered <- ( test$pred_reordered_org > pred_threshold) * 1
f1 =calc_submit_f1_noPred( test )



#################################################################################


# Add pred_rank by probability
t_pred <- test_pred %>%
  arrange(order_id,  desc(pred_reordered_org)) %>%
  group_by(order_id) %>%
  mutate(pred_rank = row_number()) 

submaxf1 <- test_pred %>%
  group_by(order_id) %>%
  summarise(products = exact_F1_max_none(pred_reordered_org, product_id))


sub_maxf1_size <- test_pred %>%
  group_by(order_id) %>%
  summarise(submit_products_size = exact_F1_max_none_size(pred_reordered_org, product_id))

calc_submit_MaxF1(1,t_pred, submaxf1, sub_maxf1_size)


# # Combine MaxF1 with P1/2/3/4None on threshold 0.2
# test_pred <- test[,c("order_id", "product_id", "pred_reordered_org" , "reordered")]
# submaxf1 <- test_pred %>%
#   group_by(order_id) %>%
#   summarise(products = exact_F1_max_none(pred_reordered_org, product_id))
# calc_MaxF1_Combined_Fixed(test_pred, submaxf1)





# best_f1 = 0
# best_none_threshold = 0
# 
# for(i in 17:21)
# {
#   acc_threshold = as.numeric(i/100)
#   f1 =calc_submit_f1(acc_threshold, test, model, X)
# 
#   if (best_f1 < f1) {
#     best_f1 = f1
#     best_threshold = acc_threshold
#   }
# }
# print(paste("best_threshold:", best_threshold, "best_f1 is:", best_f1 ))


head_test <- test[test$user_id <= 10,]



########################### !!! Submit TO LB   ##################################

load("C:/DEV/GitHub/PZ1311_MB/LGB_Ph3_data_lgb_train_org.R.RData")


load("C:/DEV/GitHub/PZ1311_MB/LGB_Ph3_data_lgb_test_org.R.RData")
data_lgb_test_new <- data_lgb_test %>%
  inner_join(datanew, by = c('user_id', 'product_id'))
rm( data_lgb_test, datanew)
gc()



train <- data_lgb_train_new


train <- data_lgb_train
rm(data_lgb_test_new, data_lgb_train)
gc()



########################### Light GBM  ##################################

set.seed(1)
subtrain <- train %>% sample_frac(0.1)

#subtrain <- train

params <-  list(
  "task" = "train",
  "boosting_type" = "gbdt",
  "objective" =  "binary",
  "metric" = c("binary_logloss", "auc"),
  "num_leaves"= 256,
  "min_sum_hessian_in_leaf" = 20,
  "max_depth"= 12,
  "learning_rate"= 0.05,
  "feature_fraction"= 0.6,
  "verbose" = 1
)


dtrain <- subtrain %>% select(-reordered, -department_id, -aisle_id)
dtrain <- subtrain %>% select(-reordered)
#X <- sparse.model.matrix(~.-1, data = dtrain)

X<- as.matrix(data.frame(dtrain))
dlable <- subtrain$reordered

#categories = c('aisle_id', 'department_id')

cat_features = subtrain[, 'department_id']

dtrain_set <- lgb.Dataset(data = X,
                          label = dlable,
                          categorical_feature=cat_features )

dtrain_set <- lgb.Dataset(data = X,
                          label = dlable )

lgbmodel <- lightgbm(data = dtrain_set,
                     params = params,
                     nrounds = 380, 
                     early_stopping_rounds=10
)

rm(X, X_TEST)
gc()



# Apply model -------------------------------------------------------------

load("C:/DEV/GitHub/PZ1311_MB/LGB_Ph3_data_lgb_test_org.R.RData")

test <- data_lgb_test
rm(data_lgb_test)
gc()


test$eval_set <- NULL

X_TEST <- as.matrix(test %>% select( -user_id, -order_id, -product_id))
pred <- predict(lgbmodel, X_TEST)

test$pred_reordered_org <- pred
test_pred <- test[,c("order_id", "product_id", "pred_reordered_org" )]


test_pred <- test_pred %>%
  arrange(order_id,  desc(pred_reordered_org)) %>%
  group_by(order_id) %>%
  mutate(pred_rank = row_number()) 


write.csv(test_pred, file = "C:/DEV/MarketBusket/Submit/test_pred_try89.csv", row.names = F)


#Get result by fixed threshold
pred_threshold = 0.18

test$pred_reordered <- ( test$pred_reordered_org > pred_threshold) * 1
f1 =calc_submit_f1_noPred( test )















############!!! Submit by XGB ###########
# redefine train and test ----------------------------------

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


test$pred_reordered_org <- predict(model, X)

test_pred <- test[,c("order_id", "product_id", "pred_reordered_org" )]


submaxf1 <- test_pred %>%
  group_by(order_id) %>%
  summarise(products = exact_F1_max_none(pred_reordered_org, product_id))

submission <- submaxf1

# write.csv(submission, file = "C:/DEV/MarketBusket/Submit/submit_ph3_try86_MaxF1_ORG.csv", row.names = F)


test_pred <- test_pred %>%
  arrange(order_id,  desc(pred_reordered_org)) %>%
  group_by(order_id) %>%
  mutate(pred_rank = row_number()) 

submission_maxf1_size <- test_pred %>%
  group_by(order_id) %>%
  summarise(submit_products_size = exact_F1_max_none_size(pred_reordered_org, product_id))



############# !!! Submit by MaxF1 ################


#### Add single product submission with None 
t_1_prod <- test_pred %>%
  inner_join(submission_maxf1_size, by = "order_id" ) %>%
  filter (submit_products_size == 1) %>%
  filter (pred_rank == 1) %>%
  group_by(order_id) %>%
  filter( pred_reordered_org <= Add_None_Threshold_1p)


submission_singlewithNone <- submission[submission$order_id %in% t_1_prod$order_id,] %>%
  group_by(order_id) %>%
  summarise(
    products = paste( products, "None", collapse = " ")
  )

submission_new <- submission[! submission$order_id %in% submission_singlewithNone$order_id,] 
submission <- submission_new %>% bind_rows(submission_singlewithNone) %>% arrange(order_id)


#### Add None to 2 products submit
t_2_prod <- test_pred %>%
  inner_join(submission_maxf1_size, by = "order_id" ) %>%
  filter (submit_products_size == 2) %>%
  filter (pred_rank <= 2) %>%
  group_by(order_id) %>%
  mutate (pred_mean = mean(pred_reordered_org)) %>%
  filter(pred_mean <= Add_None_Threshold_2p )

submission_2p_withNone <- submission[submission$order_id %in% t_2_prod$order_id,] %>%
  group_by(order_id) %>%
  summarise(
    products = paste( products, "None", collapse = " ")
  )


submission_new <- submission[! submission$order_id %in% submission_2p_withNone$order_id,]
submission <- submission_new %>% bind_rows(submission_2p_withNone) %>% arrange(order_id)


# #### Add None to 3 products submit
t_3_prod <- test_pred %>%
  inner_join(submission_maxf1_size, by = "order_id" ) %>%
  filter (submit_products_size == 3) %>%
  filter (pred_rank <= 3) %>%
  group_by(order_id) %>%
  mutate (pred_mean = mean(pred_reordered_org)) %>%
  filter(pred_mean <= Add_None_Threshold_3p )

submission_3p_withNone <- submission[submission$order_id %in% t_3_prod$order_id,] %>%
  group_by(order_id) %>%
  summarise(
    products = paste( products, "None", collapse = " ")
  )


submission_new <- submission[! submission$order_id %in% submission_3p_withNone$order_id,]
submission <- submission_new %>% bind_rows(submission_3p_withNone) %>% arrange(order_id)


write.csv(submission, file = "C:/DEV/MarketBusket/Submit/submit_ph3_try88_100Var_MaxF1_WithP123None.csv", row.names = F)










rm(submaxf1, submission, sub_maxf1_size, subtrain, t_1_prod, t_2_prod, t_3_prod)
rm( submission_2p_withNone, submission_3p_withNone, submission_new, submission_maxf1_size, submission_singlewithNone,t_pred, test_pred)
gc()

######### Apply Fix threshold of 0.20 ##############################
acc_threshold <- 0.20
test$pred_reordered_org <- predict(model, X)

test$pred_reordered <- (test$pred_reordered_org > acc_threshold) * 1

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


######### Original way to Apply None to 1/2/3 product submitssion ##############################

test$pred_reordered_org <- predict(model, X)
test_pred <- test[,c("order_id", "product_id", "pred_reordered_org" )]

pred_submit_threshold = 0.20
test_pred$pred_reordered <- (test_pred$pred_reordered_org > pred_submit_threshold) * 1


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

### Add None to single produdct submission 

Add_None_Threshold_1p = 0.5
Add_None_Threshold_2p = 0.41
Add_None_Threshold_3p = 0.32


test_single_prod <- test_pred %>%
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
t_2_prod <- test_pred %>%
  group_by(order_id) %>%
  mutate(prodcnt = sum(pred_reordered)) %>%
  filter(prodcnt == 2 & pred_reordered == 1 ) %>%
  mutate (pred_mean = mean(pred_reordered_org)) %>%
  mutate (pred_max = max(pred_reordered_org)) %>%
  filter(pred_mean <= Add_None_Threshold_2p )

submission_2p_withNone <- submission[submission$order_id %in% t_2_prod$order_id,] %>%
  group_by(order_id) %>%
  summarise(
    products = paste( products, "None", collapse = " ")
  )

submission_new <- submission[! submission$order_id %in% submission_2p_withNone$order_id,] 
submission <- submission_new %>% bind_rows(submission_2p_withNone) %>% arrange(order_id)


#### Add None to 3 products submit 
t_3_prod <- test_pred %>%
  group_by(order_id) %>%
  mutate(prodcnt = sum(pred_reordered)) %>%
  filter(prodcnt == 3 & pred_reordered == 1 ) %>%
  mutate (pred_mean = mean(pred_reordered_org)) %>%
  mutate (pred_max = max(pred_reordered_org)) %>%
  filter(pred_mean <= Add_None_Threshold_3p  )

submission_3p_withNone <- submission[submission$order_id %in% t_3_prod$order_id,] %>%
  group_by(order_id) %>%
  summarise(
    products = paste( products, "None", collapse = " ")
  )


submission_new <- submission[! submission$order_id %in% submission_3p_withNone$order_id,] 
submission <- submission_new %>% bind_rows(submission_3p_withNone) %>% arrange(order_id)



#### Add None to 4 products submit
t_4_prod <- test_pred %>%
  group_by(order_id) %>%
  mutate(prodcnt = sum(pred_reordered)) %>%
  filter(prodcnt == 4 & pred_reordered == 1 ) %>%
  mutate (pred_mean = mean(pred_reordered_org)) %>%
  mutate (pred_max = max(pred_reordered_org)) %>%
  filter(pred_mean <= 0.25 & pred_max < 0.5 )

submission_4p_withNone <- submission[submission$order_id %in% t_4_prod$order_id,] %>%
  group_by(order_id) %>%
  summarise(
    products = paste( products, "None", collapse = " ")
  )


submission_new <- submission[! submission$order_id %in% submission_4p_withNone$order_id,]
submission <- submission_new %>% bind_rows(submission_4p_withNone) %>% arrange(order_id)

submission_FixRate_WithNone <- submission


####merge 1/2/3/4 result with MaxF, Best on Aug 4rd, 0.3962
test$pred_reordered_org <- predict(model, X)
test_pred <- test[,c("order_id", "product_id", "pred_reordered_org" )]

test_pred$pred_reordered_org[is.na(test_pred$pred_reordered_org)] <- 0
test_pred$product_id[is.na(test_pred$product_id)] <- 0
test_pred$order_id[is.na(test_pred$order_id)] <- 0

submaxf1 <- test_pred %>%
  group_by(order_id) %>%
  summarise(products = exact_F1_max_none(pred_reordered_org, product_id))

submission <- submaxf1

submission_1p_withNone <- submission_singlewithNone
submission_1p <- submission[! submission$order_id %in% submission_1p_withNone$order_id,]
submission <- submission_1p %>% bind_rows(submission_1p_withNone) %>% arrange(order_id)

submission_2p <- submission[! submission$order_id %in% submission_2p_withNone$order_id,]
submission <- submission_2p %>% bind_rows(submission_2p_withNone) %>% arrange(order_id)

submission_3p <- submission[! submission$order_id %in% submission_3p_withNone$order_id,]
submission <- submission_3p %>% bind_rows(submission_3p_withNone) %>% arrange(order_id)

submission_4p <- submission[! submission$order_id %in% submission_4p_withNone$order_id,]
submission <- submission_4p %>% bind_rows(submission_4p_withNone) %>% arrange(order_id)

write.csv(submission, file = "C:/DEV/MarketBusket/Submit/submit_ph3_try85_76Vars_org.csv", row.names = F)







