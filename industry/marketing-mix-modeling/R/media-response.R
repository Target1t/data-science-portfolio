adstock_geometric <- function(seq_x, theta){
  seq_x_decayed <- c(seq_x[1], rep(0, length(seq_x) - 1))
  for (xi in 2:length(seq_x_decayed)) {
    seq_x_decayed[xi] <- seq_x[xi] + theta * seq_x_decayed[xi - 1]
  }
  return(seq_x_decayed)
}

## Saturation处理
f_sat <- function(seq_x_decayed, alpha, gammaTran, coef){
  seq_y <- coef * seq_x_decayed ^ alpha / (seq_x_decayed ^ alpha + gammaTran ^ alpha)
  return(seq_y)
}

## 近似求导（差分）处理
f_prime <- function(seq_x, seq_y){
  f_prime_values <- c(0, (seq_y[2:length(seq_y)] - seq_y[1:length(seq_y)-1]) / (seq_x[2:length(seq_x)] - seq_x[1:length(seq_x)-1]))
  return(f_prime_values)
}
