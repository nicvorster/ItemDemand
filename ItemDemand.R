library(vroom)
library(timetk)
library(tidyverse)
library(patchwork)
library(tidymodels)
library(forecast)

ID_train <- vroom("./train.csv") 
ID_test <- vroom("./test.csv") 



store1_item3 <- ID_train %>% 
  filter(store ==1 & item ==3)

store5_item11 <- ID_train %>% 
  filter(store ==5 & item ==11)

store7_item13 <- ID_train %>% 
  filter(store ==7 & item ==13)

store10_item25 <- ID_train %>% 
  filter(store ==10 & item ==25)

plot1 <- store1_item3 %>%
  pull(sales) %>% 
  forecast::ggAcf(.,lag.max= 2*365)

plot2 <- store5_item11 %>%
  pull(sales) %>% 
  forecast::ggAcf(.,lag.max= 2*365)

plot3 <- store7_item13 %>%
  pull(sales) %>% 
  forecast::ggAcf(.,lag.max= 2*365)

plot4 <- store10_item25 %>%
  pull(sales) %>% 
  forecast::ggAcf(.,lag.max= 2*365)

(plot1 + plot2)/(plot3+plot4)

##############


storeItem <- ID_train %>% 
  filter(store == 3, item == 7)

my_recipe <- recipe(sales~., data = storeItem) %>% 
  step_date(date, features =c("dow","month","year"))
  
  

my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 600) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")


rf_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(my_mod)

tuning_grid <- grid_regular(mtry(range = c(1,6)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(storeItem, v= 5, repeats = 1)


CV_results <- rf_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(smape))

bestTune <- CV_results %>% 
  select_best()


collect_metrics(CV_results) %>% 
  filter(bestTune) %>% 
  pull(mean)

final_wf <-
  RF_amazon_workflow %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = ID_train)


#########
#ES


install.packages("modeltime")
library(modeltime)


train <- ID_train %>% filter(store == 3, item == 10)
cv_split1 <- time_series_split(train, assess= "3 months", cumulative = TRUE)
cv_split %>% tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)

es_model <- exp_smoothing() %>% 
  set_engine("ets") %>% 
  fit(sales~date, data =training(cv_split))

cv_results <- modeltime_calibrate(es_model, 
                                  new_data = testing(cv_split))
p1 <-cv_results %>% 
  modeltime_forecast(new_data = testing(cv_split),
                     actual_data = train) %>% 
  plot_modeltime_forecast(.interactive = TRUE)

cv_results %>% modeltime_accuracy() %>% 
  table_modeltime_accuracy(
    .interactive = FALSE
  )

es_fullfit <- cv_results %>% 
  modeltime_refit(data = train)

es_preds <- es_fullfit %>% 
  modeltime_forecast(h = "3 months") %>% 
  rename(date= .index, sales = .value) %>% 
  select(date, sales) %>% 
  full_join(., y= ID_test, by= "date") %>% 
  select(id, sales)

p2 <-es_fullfit %>% 
  modeltime_forecast(h = "3 months", actual_data = train) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

##

train2 <- ID_train %>% filter(store == 5, item == 7)
cv_split2 <- time_series_split(train2, assess= "3 months", cumulative = TRUE)
cv_split %>% tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)

es_model <- exp_smoothing() %>% 
  set_engine("ets") %>% 
  fit(sales~date, data =training(cv_split))

cv_results <- modeltime_calibrate(es_model, 
                                  new_data = testing(cv_split))
p3 <-cv_results %>% 
  modeltime_forecast(new_data = testing(cv_split),
                     actual_data = train2) %>% 
  plot_modeltime_forecast(.interactive = TRUE)

cv_results %>% modeltime_accuracy() %>% 
  table_modeltime_accuracy(
    .interactive = FALSE
  )

es_fullfit <- cv_results %>% 
  modeltime_refit(data = train)

es_preds <- es_fullfit %>% 
  modeltime_forecast(h = "3 months") %>% 
  rename(date= .index, sales = .value) %>% 
  select(date, sales) %>% 
  full_join(., y= ID_test, by= "date") %>% 
  select(id, sales)

p4 <-es_fullfit %>% 
  modeltime_forecast(h = "3 months", actual_data = train2) %>% 
  plot_modeltime_forecast(.interactive = FALSE)
plotly::subplot(p1,p3, p2,p4, nrows = 2)

##########
#ARIMA
###########

install.packages("forecast")

arima_recipe <- recipe(sales~., data = train) %>% 
  step_date(date, features =c("dow","month","year"))

arima_model <- arima_reg(seasonal_period = 365,
                         non_seasonal_ar = 5,
                         non_seasonal_ma = 5, 
                         seasonal_ar = 2,
                         non_seasonal_differences = 2,
                         seasonal_differences = 2) %>% 
  set_engine("auto_arima")
######## PROFESSOR CODE



arima_recipe <- recipe(sales~., data=train) %>%
  step_rm(item, store) %>%
  step_date(date, features=c("doy", "decimal")) %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
  step_rm(date_doy)
bake(prep(arima_recipe), new_data=train)
arima_model <- arima_reg() %>%
  set_engine("auto_arima")
arima_wf <- workflow() %>%
  add_recipe(arima_recipe) %>%
  add_model(arima_model) %>%
  fit(data=training(split))
cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(split))
## Visualize results
cv_results %>%
  modeltime_forecast(
    new_data    = testing(split),
    actual_data = train
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)
## Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )
## Refit to whole data
fullfit <- cv_results %>%
  modeltime_refit(data = train)
fullfit %>%
  modeltime_forecast(
    new_data    = test,
    actual_data = train
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)





#######
cv_split1 <- time_series_split(train, assess= "3 months", cumulative = TRUE)

train <- ID_train %>% filter(store == 3, item == 10)
cv_split1 <- time_series_split(train, assess= "3 months", cumulative = TRUE)
cv_split1 %>% tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)

test1 <- ID_test %>% filter(store == 3, item == 10)
cv_split1 <- time_series_split(test1, assess= "3 months", cumulative = TRUE)

arima_wf <- workflow() %>% 
  add_recipe(arima_recipe) %>% 
  add_model(arima_model) %>% 
  fit(data=training(cv_split1))
  
  
cv_results <- modeltime_calibrate(arima_wf, 
                                  new_data = testing(cv_split1))
p1<- cv_results %>% 
  modeltime_forecast(new_data = testing(cv_split1),
                     actual_data = train) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

cv_results %>% modeltime_accuracy() %>% 
  table_modeltime_accuracy(.interactive = FALSE)

  
arima_fullfit <- cv_results %>% 
  modeltime_refit(data = train)

arima_preds <- arima_fullfit %>% 
  modeltime_forecast(new_data=train) %>% 
  rename(date= .index, sales = .value) %>% 
  select(date, sales) %>% 
  full_join(., y= ID_test, by= "date") %>% 
  select(id, sales)

p2<- arima_fullfit %>% 
  modeltime_forecast(new_data= train, actual_data = train) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

###########

arima_recipe <- recipe(sales~., data = train2) %>% 
  step_date(date, features =c("dow","month","year"))

arima_model <- arima_reg(seasonal_period = 365,
                         non_seasonal_ar = 5,
                         non_seasonal_ma = 5, 
                         seasonal_ar = 2,
                         non_seasonal_differences = 2,
                         seasonal_differences = 2) %>% 
  set_engine("auto_arima")

arima_wf <- workflow() %>% 
  add_recipe(arima_recipe) %>% 
  add_model(arima_model) %>% 
  fit(data=training(cv_split2))


cv_results <- modeltime_calibrate(arima_wf, 
                                  new_data = testing(cv_split2))
p3 <- cv_results %>% 
  modeltime_forecast(new_data = testing(cv_split2),
                     actual_data = train2) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

cv_results %>% modeltime_accuracy() %>% 
  table_modeltime_accuracy(.interactive = FALSE)

arima_fullfit <- cv_results %>% 
  modeltime_refit(data = train2)

arima_preds <- arima_fullfit %>% 
  modeltime_forecast(new_data=train2) %>% 
  rename(date= .index, sales = .value) %>% 
  select(date, sales) %>% 
  full_join(., y= ID_test, by= "date") %>% 
  select(id, sales)

p4 <- arima_fullfit %>% 
  modeltime_forecast(new_data= train2, actual_data = train2) %>% 
  plot_modeltime_forecast(.interactive = FALSE, .legend_show = FALSE)

plotly::subplot(p1,p3, p2,p4, nrows = 2)  
  
  
################
#Prophet
###############

train <- ID_train %>% filter(store == 3, item == 10)
cv_split1 <- time_series_split(train, assess= "3 months", cumulative = TRUE)

prophet_model <- prophet_reg() %>% 
  set_engine(engine= "prophet") %>% 
  fit(sales~date, data = training(cv_split1))

cv_results <- modeltime_calibrate(prophet_model,
                                  new_data = testing(cv_split1))
cv_split1 %>% tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)

p1 <- cv_results %>% 
  modeltime_forecast(new_data = testing(cv_split1),
                     actual_data = train) %>% 
  plot_modeltime_forecast(.interactive = TRUE, .legend_show = FALSE)

cv_results %>% modeltime_accuracy() %>% 
  table_modeltime_accuracy(
    .interactive = FALSE
  )

es_fullfit <- cv_results %>% 
  modeltime_refit(data = train)

es_preds <- es_fullfit %>% 
  modeltime_forecast(h = "3 months") %>% 
  rename(date= .index, sales = .value) %>% 
  select(date, sales) %>% 
  full_join(., y= ID_test, by= "date") %>% 
  select(id, sales)

p2 <- es_fullfit %>% 
  modeltime_forecast(h = "3 months", actual_data = train) %>% 
  plot_modeltime_forecast(.interactive = FALSE, .legend_show = FALSE)

####

train2 <- ID_train %>% filter(store == 5, item == 7)
cv_split2 <- time_series_split(train2, assess= "3 months", cumulative = TRUE)

prophet_model <- prophet_reg() %>% 
  set_engine(engine= "prophet") %>% 
  fit(sales~date, data = training(cv_split2))

cv_results <- modeltime_calibrate(prophet_model,
                                  new_data = testing(cv_split2))
cv_split2 %>% tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)

p3 <- cv_results %>% 
  modeltime_forecast(new_data = testing(cv_split2),
                     actual_data = train2) %>% 
  plot_modeltime_forecast(.interactive = TRUE, .legend_show = FALSE)

cv_results %>% modeltime_accuracy() %>% 
  table_modeltime_accuracy(
    .interactive = FALSE)

es_fullfit <- cv_results %>% 
  modeltime_refit(data = train2)

es_preds <- es_fullfit %>% 
  modeltime_forecast(h = "3 months") %>% 
  rename(date= .index, sales = .value) %>% 
  select(date, sales) %>% 
  full_join(., y= ID_test, by= "date") %>% 
  select(id, sales)

p4 <- es_fullfit %>% 
  modeltime_forecast(h = "3 months", actual_data = train2) %>% 
  plot_modeltime_forecast(.interactive = FALSE, .legend_show = FALSE)

plotly::subplot(p1,p3, p2,p4, nrows = 2) 

#######

### FINAL MODEL ###

#######
library(vroom)
library(timetk)
library(tidyverse)
library(patchwork)
library(tidymodels)
library(forecast)

ID_train <- vroom("./train.csv") 
ID_test <- vroom("./test.csv") 



store1_item3 <- ID_train %>% 
  filter(store ==1 & item ==3)

store5_item11 <- ID_train %>% 
  filter(store ==5 & item ==11)

store7_item13 <- ID_train %>% 
  filter(store ==7 & item ==13)

store10_item25 <- ID_train %>% 
  filter(store ==10 & item ==25)

storeItem <- ID_train %>% 
  filter(store == 3, item == 7)

install.packages("modeltime")

library(modeltime)
ID_train <- ID_train %>% filter(store == 3, item == 10)
#cv_split1 <- time_series_split(train, assess= "3 months", cumulative = TRUE)

nStores <- max(ID_train$store)
nItems <- max(ID_train$item)

it <- 0
for(s in 1:nStores){
  for(i in 1:nItems){
    it <- it + 1
    storeItemTrain <- ID_train %>% 
      filter(store ==s, item ==i) %>% 
      select(date, sales)
    storeItemTest <- ID_test %>% 
      filter(store==s, item ==1) %>% 
      select(id, date)
    
    cv_split <- time_series_split(ID_train, assess= "3 months",
                                   cumulative = TRUE)
    prophet_model <- prophet_reg() %>% 
      set_engine(engine = "prophet") %>% 
      fit(sales ~ date, data = training(cv_split))
    cv_results <- modeltime_calibrate(prophet_model, 
                                      new_data= testing(cv_split))
    
    preds <- cv_results %>% 
      modeltime_refit(data= ID_train) %>% 
      modeltime_forecast(h= "3 months") %>% 
      rename(date = .index, sales = .value) %>% 
      select(date, sales) %>% 
      full_join(.,y = ID_test, by= "date") %>% 
      select(id, sales)
    
    if(it==1){
      all_preds <- preds
    } else{
      all_preds <- bind_rows(all_preds, preds)
    }
  }
}

all_preds <- all_preds %>% 
  arrange(id)

