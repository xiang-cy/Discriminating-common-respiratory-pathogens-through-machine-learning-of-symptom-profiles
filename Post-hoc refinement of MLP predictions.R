setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(dplyr)
library(tidyverse)
library(pROC)
library(caret)
library(psych)
library(ggplot2)
library(ggsci)
library(scales)
library(tibble)
library(binom)     

disease_mapping <- list(
  Class_0 = c(4, 5, 6),  
  Class_1 = c(0, 8),                
  Class_2 = c(1, 2, 3, 7)               
)

all.positive <- read.csv("epi_prior_all_popu.csv", stringsAsFactors = FALSE)
all.positive <- all.positive %>% filter(Age_Group=="6-17y")

pathogen_mapping <- tibble::tibble(
  Pathogen_9class = 0:8,
  Pathogen = c("HAdV", "HCoV", "HPIV", "HRV", "InfA", "InfB", "M.pneumoniae", "RSV", "SARS_CoV_2")
)

all.positive <- all.positive %>%
  left_join(pathogen_mapping, by = "Pathogen") %>%  
  mutate(
    Pathogen_9class = if_else(is.na(Pathogen_9class.x), Pathogen_9class.y, Pathogen_9class.x),
    n = if_else(is.na(n), 0, n)
  ) %>%
  select(-Pathogen_9class.x, -Pathogen_9class.y) 

data <- read.csv("XXX.csv", stringsAsFactors = FALSE)

likelihoods <- data %>%
  select(Index, Class_0_Probability, Class_1_Probability, Class_2_Probability, Date,Pathogen_9class ) %>%
  mutate(
    MLP_HAdV = Class_1_Probability / length(disease_mapping$Class_1),
    MLP_HCoV = Class_2_Probability / length(disease_mapping$Class_2),
    MLP_HPIV = Class_2_Probability / length(disease_mapping$Class_2),
    MLP_InfA = Class_0_Probability / length(disease_mapping$Class_0),
    MLP_InfB = Class_0_Probability / length(disease_mapping$Class_0),
    MLP_SARS_CoV_2 = Class_1_Probability / length(disease_mapping$Class_1),
    MLP_M.pneumoniae =  Class_0_Probability / length(disease_mapping$Class_0),
    MLP_HRV = Class_2_Probability / length(disease_mapping$Class_2),
    MLP_RSV = Class_2_Probability / length(disease_mapping$Class_2)
  ) %>%
  select(-Class_0_Probability, -Class_1_Probability, -Class_2_Probability)%>%
  rename(True_Pathogen = Pathogen_9class)

posterior_probs_part1 <- likelihoods %>%
  pivot_longer(
    cols = starts_with("MLP_"),         
    names_to = "Pathogen_predict",             
    values_to = "Likelihood"           
  )


safe_date_convert <- function(df) {
  df %>%
    mutate(
      Date = coalesce(
        as.Date(Date, format = "%Y-%m-%d"), 
        parse_date_time(Date, orders = c("ymd", "dmy", "mdy")) %>% as.Date() 
      )
    ) %>%
    filter(!is.na(Date)) %>% 
    arrange(Date) 
}
posterior_probs_part1 <- safe_date_convert(posterior_probs_part1) 

start_date = as.Date("2024-01-01")
posterior_probs_part1 <- posterior_probs_part1 %>%
  mutate(
    day_diff = as.numeric(Date - start_date),
    week_group = day_diff %/% 14 
  )

pathogen_mapping <- tibble::tibble(
  Pathogen_predict = c("MLP_HAdV", "MLP_HCoV", "MLP_HPIV", "MLP_HRV", "MLP_InfA", "MLP_InfB", "MLP_M.pneumoniae", "MLP_RSV", "MLP_SARS_CoV_2"),
  Pathogen_9class= 0:8
)


posterior_probs_part2 <- posterior_probs_part1 %>%
  left_join(pathogen_mapping, by = "Pathogen_predict") %>%  
  left_join(
    all.positive, 
    by = c("week_group", "Pathogen_9class") 
  )

posterior_probs_part3 <- posterior_probs_part2 %>%
  group_by(Index) %>%
  mutate(
    Numerator = Likelihood * prior, 
    Denominator = sum(Numerator, na.rm = TRUE),  
    Posterior = Numerator / Denominator  
  )


df <- posterior_probs_part3

df_sorted <- df %>%
  arrange(Index, desc(Posterior)) %>%
  group_by(Index) %>%
  mutate(rank = row_number())  

df_top1 <- df_sorted %>%
  filter(rank == 1) 


all_classes <- sort(unique(c(df_top1$Pathogen_9class, df_top1$True_Pathogen)))
conf_matrix1 <- table(
  Predicted = factor(df_top1$Pathogen_9class, levels = all_classes),
  Actual = factor(df_top1$True_Pathogen, levels = all_classes)
)

Pathogen_name <- c("HAdV", "HCoV", "HPIV", "HRV", "InfA", "InfB", "M.pneumoniae", "RSV", "SARS_CoV_2")
Pathogen <- 0:8

df_heatmap <- as.data.frame(as.table(conf_matrix1)) %>%
  setNames(c("Predicted", "Actual", "Freq")) %>%
  mutate(
    Predicted = factor(Predicted, levels = 0:8, labels = Pathogen_name),
    Actual    = factor(Actual,    levels = 0:8, labels = Pathogen_name)
  ) %>%
  group_by(Actual) %>%
  mutate(Percent = Freq / sum(Freq) * 100) %>%
  ungroup()

all_classes <- 0:8
class_names <- c("HAdV","HCoV","HPIV","HRV","InfA","InfB",
                 "M.pneumoniae","RSV","SARS_CoV_2")
B <- 1000  


calc_overall_str <- function(conf_mat) {
  res   <- confusionMatrix(conf_mat)
  acc   <- res$overall["Accuracy"];    acc_l <- res$overall["AccuracyLower"];    acc_u <- res$overall["AccuracyUpper"]
  acc_s <- sprintf("%.2f (%.2f–%.2f)", acc, acc_l, acc_u)
  
  kk    <- cohen.kappa(conf_mat)
  k     <- kk$kappa;                   k_l   <- kk$confid[1,1];                  k_u   <- kk$confid[1,2]
  k_s   <- sprintf("%.2f (%.2f–%.2f)", k, k_l, k_u)
  
  tibble(
    Metric = c("Accuracy","Kappa"),
    Value  = c(acc_s, k_s)
  )
}


calc_per_class_str <- function(df, class_id, B = 1000) {
  true_lbl <- df$True_Pathogen == class_id
  pred_lbl <- df$Pathogen_9class == class_id
  
  TP <- sum( true_lbl & pred_lbl)
  FN <- sum( true_lbl & !pred_lbl)
  FP <- sum(!true_lbl & pred_lbl)
  TN <- sum(!true_lbl & !pred_lbl)
  
  ci_s <- binom.confint(TP, TP+FN, method="wilson")
  sens  <- ci_s$mean; sens_l <- ci_s$lower; sens_u <- ci_s$upper
  sens_s<- sprintf("%.2f (%.2f–%.2f)", sens, sens_l, sens_u)
  
  ci_t <- binom.confint(TN, TN+FP, method="wilson")
  spec  <- ci_t$mean; spec_l <- ci_t$lower; spec_u <- ci_t$upper
  spec_s<- sprintf("%.2f (%.2f–%.2f)", spec, spec_l, spec_u)
  
  if ((TP+FP)>0) {
    ci_p <- binom.confint(TP, TP+FP, method="wilson")
    prec  <- ci_p$mean; prec_l <- ci_p$lower; prec_u <- ci_p$upper
    prec_s<- sprintf("%.2f (%.2f–%.2f)", prec, prec_l, prec_u)
  } else {
    prec  <- NA; prec_s <- "NA"
  }
  
  if (!is.na(prec) && !is.na(sens) && (prec + sens) > 0) {
    f1 <- 2 * prec * sens / (prec + sens)
  } else {
    f1 <- NA
  }
  
  f1_bs <- replicate(B, {
    idx <- sample(nrow(df), replace=TRUE)
    t   <- df$True_Pathogen[idx]   == class_id
    p   <- df$Pathogen_9class[idx] == class_id
    TPb <- sum(t & p); FNb <- sum(t & !p); FPb <- sum(!t & p)
    if ((TPb+FPb)==0 || (TPb+FNb)==0) return(NA)
    prec_b <- TPb/(TPb+FPb)
    rec_b  <- TPb/(TPb+FNb)
    if ((prec_b+rec_b)==0) return(NA)
    2 * prec_b * rec_b / (prec_b + rec_b)
  })
  f1_ci <- quantile(f1_bs, c(0.025,0.975), na.rm=TRUE)
  if (!is.na(f1)) {
    f1_s <- sprintf("%.2f (%.2f–%.2f)", f1, f1_ci[1], f1_ci[2])
  } else {
    f1_s <- "NA"
  }
  
  tibble(
    Pathogen    = class_id,
    Sensitivity = sens_s,
    Specificity = spec_s,
    Precision   = prec_s,
    F1          = f1_s
  )
}

conf1 <- table(
  Actual    = factor(df_top1$True_Pathogen,   levels=all_classes),
  Predicted = factor(df_top1$Pathogen_9class, levels=all_classes)
)
top1_overall  <- calc_overall_str(conf1)
top1_perclass <- bind_rows(lapply(all_classes, function(cl) calc_per_class_str(df_top1, cl, B))) %>%
  mutate(Pathogen = class_names[Pathogen + 1]) %>%
  select(Pathogen, Sensitivity, Specificity, Precision, F1)


df_top3 <- df_sorted %>%
  filter(rank <= 3) %>%
  group_by(Index) %>%
  summarise(Predicted_Top3 = paste(Pathogen_9class, collapse = ","), .groups = "drop")  

df_top3 <- df_top3 %>%
  left_join(df_sorted %>% filter(rank == 1) %>% select(Index, True_Pathogen), by = "Index") %>%
  mutate(Predicted_Top3_Original = Predicted_Top3) %>%  
  separate_rows(Predicted_Top3, sep = ",") %>% 
  mutate(Correct = (Predicted_Top3 == True_Pathogen)) %>% 
  group_by(Index, True_Pathogen, Predicted_Top3_Original) %>% 
  summarise(Correct = any(Correct), .groups = "drop")  

df_top3 <- df_top3 %>%
  mutate(Predicted_Positive = as.integer(Correct)) %>%  
  select(-Correct)  
df_top3$Predicted_Positive <- as.numeric(df_top3$Predicted_Positive)

accuracy <- mean(df_top3$Predicted_Positive) 


n <- nrow(df_top3)
p_hat <- accuracy
se <- sqrt(p_hat * (1 - p_hat) / n)  
z <- qnorm(0.975) 
ci_lower <- p_hat - z * se
ci_upper <- p_hat + z * se





