
############
seasonality_flags <- function(prophet_df, intervalType, seasonality_switch) {
  tryCatch({
    if (!is.data.frame(prophet_df)) stop("prophet_df 应为数据框")
    if (!"ds" %in% names(prophet_df)) stop("prophet_df 缺少时间列 'ds'")
    if (!is.character(intervalType) || !(intervalType %in% c("day", "week", "month", "year"))) {
      stop("intervalType 参数不合法，应为 'day', 'week', 'month', 或 'year'")
    }
    if (!is.logical(seasonality_switch)) stop("seasonality_switch 应为逻辑值True/False")
    
    n_points <- nrow(prophet_df)
    yearly <- FALSE
    weekly <- FALSE
    daily <- FALSE
    
    if (intervalType == 'month' && n_points >= 6) {
      yearly <- TRUE
    }
    if (intervalType == 'week' && n_points >= 53) {
      yearly <- TRUE
    }
    if (intervalType == 'day') {
      if (n_points >= 365) {
        yearly <- TRUE
      }
      weekly <- TRUE
    }
    
    if (!seasonality_switch) {
      yearly <- FALSE
      weekly <- FALSE
      daily <- FALSE
    }
    
    return(list(
      yearly = yearly,
      weekly = weekly,
      daily = daily
    ))
  }, error = function(e) {
    warning(paste0("seasonality_flags() 执行出错:", e$message))
    return(list(
      yearly = FALSE,
      weekly = FALSE,
      daily = FALSE
    ))
  })
}


#####
split_train_test <- function(prophet_df, is_trained, split_ratio = 0.85){
  if(is_trained){
    train_size <- floor(split_ratio*nrow(prophet_df))
    train_df <- prophet_df[1:train_size,]
    test_df <- prophet_df[(train_size + 1):nrow(prophet_df),]
  } else{
    train_df <- prophet_df
    test_df <- NULL
  }
  return(list(train_df = train_df, test_df = test_df))
}

#######
evaluate_prophet <- function(changepoint.prior.scale, seasonality.prior.scale, 
                             n.changepoints, changepoint.range, holidays.prior.scale, 
                             train_df, test_df, holidays_df, intervalType, is_trained, seasonality_switch,
                             current_iter = NULL, total_iters = NULL) {
  tryCatch({
    if (!is.null(current_iter) && !is.null(total_iters)) {
      cat("\r\033[K")  # 清除当前行
      cat(sprintf("[进度] %3d/%3d (%3.0f%%) | changepoint=%5.2f seasonality=%5.2f n_chg=%3d range=%4.2f holidays=%4.2f",
                  current_iter, total_iters, current_iter / total_iters * 100,
                  changepoint.prior.scale, seasonality.prior.scale,
                  n.changepoints, changepoint.range, holidays.prior.scale))
      flush.console()
    }
    if (!is.data.frame(train_df)) stop("train_df 不是 data.frame")
    if (!is.data.frame(holidays_df)) stop("holidays_df 不是 data.frame")
    if (!is.logical(is_trained)) stop("is_trained 不是逻辑值")
    if (!is.character(intervalType)) stop("intervalType 不是字符串")
    
    sflags <- tryCatch({
      seasonality_flags(train_df, intervalType, seasonality_switch)
    }, error = function(e) {
      warning(paste0("seasonality_flags() 运行失败: ", e$message))
      list(yearly = FALSE, weekly = FALSE, daily = FALSE)
    })
    
    m <- tryCatch({
      prophet(
        changepoint.prior.scale = changepoint.prior.scale,
        seasonality.prior.scale = seasonality.prior.scale,
        interval.width = 0.95,
        n.changepoints = n.changepoints,
        changepoint.range = changepoint.range,
        holidays.prior.scale = holidays.prior.scale,
        holidays = holidays_df,
        yearly.seasonality = sflags$yearly,
        weekly.seasonality = sflags$weekly,
        daily.seasonality = sflags$daily
      )
    }, error = function(e) {
      warning(paste0("prophet 初始化失败: ", e$message))
      return(NA)
    })
    
    if (is.null(m) || all(is.na(m))) return(c(test_mape = NA, train_mape = NA))
    if (intervalType != 'month' & intervalType != 'quarter' & seasonality_switch) {
      m <- tryCatch({
        add_seasonality(m, name = 'monthly', period = 30.5, fourier.order = 4)
      }, error = function(e) {
        warning(paste0("add_seasonality() 失败: ", e$message))
        m
      })
    }
    warnings <- list()
    withCallingHandlers({
      m <- fit.prophet(m, train_df)
    }, warning = function(w) {
      warnings <<- c(warnings, list(w$message))
      invokeRestart("muffleWarning")
    })
    future_periods <- ifelse(is_trained, nrow(test_df), 1)
    future_df <- tryCatch({
      make_future_dataframe(m, periods = future_periods, freq = intervalType)
    }, error = function(e) {
      warning(paste0("make_future_dataframe() 失败: ", e$message))
      return(NULL)
    })
    
    if (is.null(future_df)) return(c(test_mape = NA, train_mape = NA))
    
    forecast <- tryCatch({
      predict(m, future_df)
    }, error = function(e) {
      warning(paste0("predict() 失败: ", e$message))
      return(NULL)
    })
    
    if (is.null(forecast)) return(c(test_mape = NA, train_mape = NA))
    if (!is_trained) forecast <- forecast[1:nrow(train_df), ]
    train_mape <- tryCatch({
      mean(abs((train_df$y - head(forecast$yhat, nrow(train_df))) / train_df$y), na.rm = TRUE)
    }, error = function(e) {
      warning(paste0("train_mape 计算失败: ", e$message))
      NA
    })
    
    test_mape <- tryCatch({
      if (is_trained && !is.null(test_df)) {
        preds <- tail(forecast$yhat, nrow(test_df))
        actuals <- test_df$y
        mean(abs((actuals - preds) / actuals), na.rm = TRUE)
      } else {
        NA
      }
    }, error = function(e) {
      warning(paste0("test_mape 计算失败: ", e$message))
      NA
    })
    
    return(c(test_mape = test_mape, train_mape = train_mape))
  }, error = function(e) {
    warning(paste0("evaluate_prophet() 整体执行失败: ", e$message))
    return(c(test_mape = NA, train_mape = NA))
  })
}


#######
run_grid_search <- function(param_grid, prophet_df, holidays_df, intervalType, is_trained, seasonality_switch) {

  plan(multisession, workers = parallel::detectCores() - 1) 
  try_wrapper <- function(...) {
    result <- tryCatch({
      evaluate_prophet(...)
    }, error = function(e) {
      warning(paste0("evaluate_prophet() 出错: ", e$message))
      return(c(test_mape = NA, train_mape = NA))
    })
    return(result)
  }
  
  split_data <- split_train_test(prophet_df, is_trained)
  train_df <- split_data$train_df
  test_df <- split_data$test_df
  
  total_iters <- nrow(param_grid)
  message(sprintf("=== 网格搜索开始（共 %d 组参数） ===", total_iters))
  param_grid <- param_grid %>% mutate(iter = row_number())
  
  handlers(global = TRUE)
  handlers("txtprogressbar")  # 可选 "progress" 或 "cli"
  results <- with_progress({
    p <- progressor(steps = total_iters)
    
    future_pmap(
      list(
        changepoint.prior.scale = param_grid$changepoint.prior.scale,
        seasonality.prior.scale = param_grid$seasonality.prior.scale,
        n.changepoints = param_grid$n.changepoints,
        changepoint.range = param_grid$changepoint.range,
        holidays.prior.scale = param_grid$holidays.prior.scale,
        current_iter = param_grid$iter,
        total_iters = total_iters
      ),
      function(changepoint.prior.scale, seasonality.prior.scale, 
               n.changepoints, changepoint.range, holidays.prior.scale, 
               current_iter, total_iters) {
        p(sprintf("正在处理第 %d/%d 个参数组合", current_iter, total_iters))
        
        try_wrapper(
          changepoint.prior.scale = changepoint.prior.scale,
          seasonality.prior.scale = seasonality.prior.scale,
          n.changepoints = n.changepoints,
          changepoint.range = changepoint.range,
          holidays.prior.scale = holidays.prior.scale,
          train_df = train_df,
          test_df = test_df,
          holidays_df = holidays_df,
          intervalType = intervalType,
          is_trained = is_trained,
          seasonality_switch = seasonality_switch,
          current_iter = current_iter,
          total_iters = total_iters
        )
      },
      .options = furrr_options(seed = TRUE)
    )
  })
  param_grid <- param_grid %>%
    mutate(
      test_mape = map_dbl(results, ~ .x["test_mape"][[1]]),
      train_mape = map_dbl(results, ~ .x["train_mape"][[1]])
    ) %>%
    select(-iter)
  
  message("=== 网格搜索完成 ===")
  return(param_grid)
}


#######
prophet_pipeline <- function(
    best_params,
    prophet_df,
    holidays_df,      
    intervalType,
    is_trained,     
    seasonality_switch,
    further_forecast
) {
  tryCatch({
    stopifnot(is.data.frame(prophet_df), nrow(prophet_df) > 0)
    stopifnot(is.data.frame(holidays_df))
    stopifnot(all(c("changepoint.prior.scale", "seasonality.prior.scale", "n.changepoints", 
                    "changepoint.range", "holidays.prior.scale") %in% names(best_params)))
    stopifnot(is.character(intervalType), intervalType %in% c("day", "week", "month", "quarter", "year"))
    stopifnot(is.logical(is_trained))
    stopifnot(is.logical(seasonality_switch))
    stopifnot(is.numeric(further_forecast))
  }, error = function(e) {
    stop(paste0("网格搜索参数验证失败:", e$message))
  })
  
  split_data <- tryCatch({
    split_train_test(prophet_df, is_trained = is_trained)
  }, error = function(e) {
    stop(paste0("训练测试数据集拆分失败: ", e$message))
  })
  
  train_df <- split_data$train_df
  test_df <- split_data$test_df
  
  sflags <- tryCatch({
    seasonality_flags(train_df, intervalType, seasonality_switch)
  }, error = function(e) {
    stop(paste0("季节性Seasonality生成失败: ", e$message))
  })
  
  m <- tryCatch({
    prophet(
      changepoint.prior.scale = best_params$changepoint.prior.scale,
      seasonality.prior.scale = best_params$seasonality.prior.scale,
      interval.width = 0.95,
      n.changepoints = best_params$n.changepoints,
      changepoint.range = best_params$changepoint.range,
      holidays.prior.scale = best_params$holidays.prior.scale,
      holidays = holidays_df,
      yearly.seasonality = sflags$yearly,                          
      weekly.seasonality = sflags$weekly,                         
      daily.seasonality = sflags$daily
    )
  }, error = function(e) {
    stop(paste0("Prophet模型初始化失败: ", e$message))
  })
  
  if (intervalType != 'month' & intervalType != 'quarter' & seasonality_switch) {
    m <- tryCatch({
      add_seasonality(m, name = 'monthly', period = 30.5, fourier.order = 4)
    }, error = function(e) {
      warning(paste0("添加自定义季节性失败: ", e$message))
      m
    })
  }

  m <- tryCatch({
    fit.prophet(m, train_df)
  }, error = function(e) {
    stop(paste0("Prophet模型训练失败: ", e$message))
  })
  
  forecast_full <- tryCatch({
    future_time_steps <- ifelse(is_trained, nrow(test_df), 1)
    future_full <- make_future_dataframe(m, periods = future_time_steps, freq = intervalType)
    forecast <- predict(m, future_full)
    if (!is_trained) forecast <- forecast[1:nrow(train_df),]
    forecast
  }, error = function(e) {
    stop(paste0("Prophet预测生成失败: ", e$message))
  })
  
  forecast_full_further <- tryCatch({
    future_full_further <- make_future_dataframe(m, periods = ifelse(is_trained, nrow(test_df) + further_forecast, further_forecast), freq = intervalType)
    predict(m, future_full_further)
  }, error = function(e) {
    warning(paste0("未来点预测生成失败: ", e$message))
    NULL
  })
  
  trend_plot <- tryCatch({
    plot(m, forecast_full)
  }, error = function(e) {
    warning(paste0("趋势图绘制失败: ", e$message))
    NULL
  })
  
  components_plot <- tryCatch({
    prophet_plot_components(m, forecast_full)
  }, error = function(e) {
    warning(paste0("分量图绘制失败: ", e$message))
    NULL
  })
  
  ### 6. MAPE Plot ###
  mape_plot <- tryCatch({
    mape_data <- prophet_df %>%
      full_join(forecast_full, by = "ds") %>%
      filter(!is.na(y)) %>%
      mutate(
        residual = y - yhat,
        mape = abs(y - yhat) / y
      )
    
    ggplot(mape_data, aes(x = ds, y = mape)) +
      geom_line(color = 'steelblue', size = 1) +
      geom_point(color = 'red', size = 2) +
      labs(
        title = "预测绝对百分比误差(MAPE)随时间变化",
        x = "日期",
        y = "MAPE(%)"
      ) +
      scale_y_continuous(labels = scales::percent) + 
      theme_minimal()
  }, error = function(e) {
    warning(paste0("MAPE图绘制失败: ", e$message))
    NULL
  })
  
  final_mape <- tryCatch({
    if (is_trained && !is.null(test_df) && nrow(test_df) > 0) {
      actuals <- test_df$y
      preds <- tail(forecast_full$yhat, nrow(test_df)) 
      mape <- mean(abs((actuals - preds) / actuals), na.rm = TRUE)
      cat("优化后模型测试集 MAPE:", round(mape * 100, 4), "%\n")
      mape
    } else if (!is_trained && !is.null(prophet_df) && nrow(prophet_df) > 0) {
      actuals <- prophet_df$y
      preds <- tail(forecast_full$yhat, nrow(prophet_df)) 
      mape <- mean(abs((actuals - preds) / actuals), na.rm = TRUE)
      cat("全量训练模型在测试集上的 MAPE:", round(mape * 100, 4), "%\n")
      mape
    } else {
      warning(paste0("测试集为空或结构不匹配，无法计算MAPE。"))
      NA
    }
  }, error = function(e) {
    warning(paste0("MAPE计算失败: ", e$message))
    NA
  })
  
  return(list(
    model = m,
    forecast = forecast_full,
    forecast_further = forecast_full_further,
    trend_plot = trend_plot,
    components_plot = components_plot,
    mape_plot = mape_plot,
    final_mape = final_mape
  ))
}


##############
set_prophet_holidays <- function(dt_transform, dt_holidays, intervalType) {
  opts <- c("day", "week", "month", "quarter")
  if (!intervalType %in% opts) {
    stop("Pass a valid 'intervalType'. Any of: ", paste(opts, collapse = ", "))
  }
  
  if (intervalType == "day") {
    holidays <- dt_holidays
  }
  
  if (intervalType == "week") {
    weekStartInput <- lubridate::wday(dt_transform$ds[1], week_start = 1)
    holidays <- dt_holidays %>%
      mutate(ds = floor_date(as.Date(ds, origin = "1970-01-01"), unit = "week", week_start = weekStartInput)) %>%
      select(ds, holiday, country, year) %>%
      arrange(ds, country, year)
  }
  
  if (intervalType == "month") {
    if (!all(day(dt_transform$ds) == 1)) {
      stop("Monthly data should have first day of month as datestampe, e.g.'2020-01-01'")
    }
    holidays <- dt_holidays %>%
      mutate(ds = cut(as.Date(ds, origin = "1970-01-01"), intervalType)) %>%
      select(ds, holiday, country, year) %>%
      select(ds, holiday, country, year) %>%
      arrange(ds, country, year)
  }
  
  if (intervalType == "quarter") {
    if (!all(day(dt_transform$ds) == 1)) {
      stop("Quarterly data should have first day of month as datestampe, e.g.'2020-01-01'")
    }
    holidays <- dt_holidays %>%
      mutate(ds = cut(as.Date(ds, origin = "1970-01-01"), intervalType)) %>%
      select(ds, holiday, country, year) %>%
      select(ds, holiday, country, year) %>%
      arrange(ds, country, year)
  }
  
  
  return(holidays)
}










