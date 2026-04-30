# Caricamento librerie necessarie (installale con install.packages() se mancanti)
library(ggplot2)
library(splines)

# ====================== DATI SINTETICI ======================
set.seed(42)
x <- seq(0, 1, length.out = 200)
y_true <- sin(2 * pi * x) + 0.5 * x^2 + sin(10 * x)
noise <- rnorm(length(x), mean = 0, sd = 0.15)
y <- y_true + noise

# Creiamo un data frame per i plot successivi
df_data <- data.frame(x = x, y = y, y_true = y_true)

# ====================== 1. SMOOTHING SPLINE (Wahba) ======================
# In R, smooth.spline() è l'implementazione classica della Smoothing Spline
# spar=0.6 offre un grado di regolarizzazione visivamente simile all's=0.8 di SciPy
spl <- smooth.spline(x, y, spar = 0.6) 
y_pred_smooth <- predict(spl, x)$y

df_smooth <- data.frame(x = x, y = y, y_true = y_true, y_pred = y_pred_smooth)

p1 <- ggplot(df_smooth, aes(x = x)) +
  geom_point(aes(y = y), color = "gray", alpha = 0.5, size = 1.5) +
  geom_line(aes(y = y_true), linetype = "dashed", color = "black", linewidth = 1) +
  # Sostituito "crimson" con il suo HEX code #DC143C
  geom_ribbon(aes(ymin = y_pred - 0.3, ymax = y_pred + 0.3), fill = "#DC143C", alpha = 0.15) +
  geom_line(aes(y = y_pred), color = "#DC143C", linewidth = 1.2) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.major = element_line(color = "grey85"),
    panel.background = element_rect(fill = "#EAEAF2", color = NA) # Simula darkgrid di seaborn
  ) +
  labs(title = "Smoothing Spline Classica (Wahba)", x = "x", y = "y")

ggsave("slide2_smoothing_spline.png", plot = p1, width = 10, height = 6, dpi = 350, bg = "white")

# ====================== 2. P-SPLINES (Eilers & Marx) ======================
# Generazione dei nodi (25 equispaziati come in Python)
knots <- seq(0, 1, length.out = 25)
# Rimuoviamo i bordi per passarli a bs() come nodi interni (aggiunge lui i boundaries)
internal_knots <- knots[-c(1, length(knots))]

# Genera matrice B-spline (equivalente a scipy.interpolate.BSpline)
B <- bs(x, knots = internal_knots, degree = 3, intercept = TRUE)

# Matrice di penalizzazione (differenze finite di ordine 2)
D <- diff(diag(ncol(B)), differences = 2)

lambdas <- c(0.001, 0.5, 50)
colors <- c("limegreen", "royalblue", "darkorange")
labels_base <- c("λ=0.001 (sotto-smussata)", "λ=0.5 (bilanciata)", "λ=50 (sovra-smussata)")

# Inizializziamo una lista per raccogliere i dati del loop
results_list <- list()

for (i in seq_along(lambdas)) {
  lam <- lambdas[i]
  
  # Costruzione matrici A e t(B)y
  pen_matrix <- lam * t(D) %*% D
  A <- t(B) %*% B + pen_matrix
  
  # Fitting (Risoluzione del sistema lineare)
  theta <- solve(A, t(B) %*% y)
  y_p_spline <- as.vector(B %*% theta)
  
  # Effective Degrees of Freedom (EDF)
  # tr(S) dove S = B %*% A^-1 %*% B^T. Sfruttiamo tr(AB) = tr(BA) per efficienza computazionale.
  edf <- sum(diag(solve(A, t(B) %*% B)))
  
  # MSE
  mse <- mean((y_true - y_p_spline)^2)
  
  # Etichetta formattata per la legenda
  label_full <- sprintf("%s\nEDF=%.1f | MSE=%.4f", labels_base[i], edf, mse)
  
  results_list[[i]] <- data.frame(
    x = x,
    y_pred = y_p_spline,
    Lambda_Label = label_full,
    Color = colors[i]
  )
}

df_results <- do.call(rbind, results_list)
# Fissiamo l'ordine della legenda come nel loop tramite i "factor"
df_results$Lambda_Label <- factor(df_results$Lambda_Label, levels = unique(df_results$Lambda_Label))

p2 <- ggplot() +
  geom_point(data = df_data, aes(x = x, y = y), color = "gray", alpha = 0.4, size = 1.5) +
  geom_line(data = df_data, aes(x = x, y = y_true), color = "black", linetype = "dashed", linewidth = 1, alpha = 0.8) +
  geom_line(data = df_results, aes(x = x, y = y_pred, color = Lambda_Label), linewidth = 1.2) +
  scale_color_manual(values = setNames(colors, unique(df_results$Lambda_Label))) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    legend.key.height = unit(1.5, "cm"), # Spazio extra per il testo su due righe (\n)
    panel.grid.major = element_line(color = "grey85"),
    panel.background = element_rect(fill = "#EAEAF2", color = NA) # Simula darkgrid
  ) +
  labs(title = "P-Splines: Effetto della Ridge Penalty λ sul Bias-Variance Tradeoff",
       x = "x", y = "y")

ggsave("slide2_p_spline_lambda_tradeoff.png", plot = p2, width = 11, height = 7, dpi = 350, bg = "white")

cat("✅ File salvati:\n")
cat("   → slide2_smoothing_spline.png\n")
cat("   → slide2_p_spline_lambda_tradeoff.png\n")
cat("   EDF e MSE calcolati per ogni λ!\n")
# ========================================================================
# AGGIUNTA DIDATTICA 1: LE FUNZIONI DI BASE (B-SPLINES)
# ========================================================================
# Usiamo il lambda "bilanciato" (0.5) per calcolare i coefficienti theta ottimali
lam_opt <- 0.5
pen_matrix_opt <- lam_opt * t(D) %*% D
A_opt <- t(B) %*% B + pen_matrix_opt
theta_opt <- solve(A_opt, t(B) %*% y)
y_fit_opt <- as.vector(B %*% theta_opt)

# Estraiamo le singole basi moltiplicate per il loro coefficiente
basis_list <- list()
for(j in 1:ncol(B)) {
  basis_list[[j]] <- data.frame(
    x = x,
    y_val = B[, j] * theta_opt[j],
    basis_id = as.factor(j),
    # Alterniamo lo stile della linea per distinguerle meglio
    line_type = ifelse(j %% 2 == 0, "even", "odd") 
  )
}
df_basis <- do.call(rbind, basis_list)
df_fit <- data.frame(x = x, y_fit = y_fit_opt)

p3 <- ggplot() +
  # Dati grezzi in sottofondo
  geom_point(data = df_data, aes(x = x, y = y), color = "black", alpha = 0.3, size = 1) +
  # Basi ponderate (grigie)
  geom_line(data = df_basis, aes(x = x, y = y_val, group = basis_id, linetype = line_type),
            color = "gray40", alpha = 0.6, linewidth = 0.8) +
  # Curva finale (somma delle basi)
  geom_line(data = df_fit, aes(x = x, y = y_fit), color = "royalblue", linewidth = 1.5) +
  scale_linetype_manual(values = c("even" = "dashed", "odd" = "solid"), guide = "none") +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.major = element_line(color = "grey85"),
    panel.background = element_rect(fill = "#EAEAF2", color = NA)
  ) +
  labs(title = "Costruzione P-Spline: Basi pesate dai coefficienti θ",
       subtitle = "La curva blu è la somma delle curve grigie sottostanti",
       x = "x", y = "y / B(x)*θ")

ggsave("slide1_b_spline_basis.png", plot = p3, width = 10, height = 5, dpi = 350, bg = "white")

# ========================================================================
# AGGIUNTA DIDATTICA 2: LA CURVA GCV (Scelta Automatica di Lambda)
# ========================================================================
# Creiamo una griglia logaritmica da 10^-4 a 10^4 (come logspace in numpy)
lambdas_grid <- 10^seq(-4, 4, length.out = 100)
n_obs <- length(y)

gcv_scores <- numeric(length(lambdas_grid))
edfs_grid <- numeric(length(lambdas_grid))

BtB <- t(B) %*% B
Bty <- t(B) %*% y

for (k in seq_along(lambdas_grid)) {
  lam_curr <- lambdas_grid[k]
  P_curr <- lam_curr * t(D) %*% D
  A_curr <- BtB + P_curr
  
  # Risoluzione del sistema e previsione
  theta_curr <- solve(A_curr, Bty)
  y_pred_curr <- as.vector(B %*% theta_curr)
  
  # EDF: Traccia della Hat Matrix H. 
  # H = B(B'B + P)^-1 B'. Traccia(H) = Traccia((B'B + P)^-1 B'B)
  edf_curr <- sum(diag(solve(A_curr, BtB)))
  edfs_grid[k] <- edf_curr
  
  # Errore Quadratico Medio
  mse_curr <- mean((y - y_pred_curr)^2)
  
  # Punteggio GCV
  gcv_scores[k] <- mse_curr / (1 - edf_curr / n_obs)^2
}

df_gcv <- data.frame(lambda = lambdas_grid, gcv = gcv_scores, edf = edfs_grid)

# Trova il punto di minimo del GCV
opt_idx <- which.min(gcv_scores)
best_lam <- lambdas_grid[opt_idx]
best_edf <- edfs_grid[opt_idx]

# Posizione per l'etichetta del testo nel plot
y_pos_text <- min(gcv_scores) + (max(gcv_scores) - min(gcv_scores)) * 0.4

p4 <- ggplot(df_gcv, aes(x = lambda, y = gcv)) +
  geom_line(color = "darkorange", linewidth = 1.2) +
  geom_vline(xintercept = best_lam, color = "#DC143C", linetype = "dashed", linewidth = 1) +
  # Etichetta del punto di minimo
  annotate("text", x = best_lam * 1.5, y = y_pos_text, 
           label = sprintf("λ Ottimo = %.4f\n(EDF = %.1f)", best_lam, best_edf),
           color = "#DC143C", hjust = 0, size = 5, fontface = "bold") +
  scale_x_log10(labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.major = element_line(color = "grey85"),
    panel.background = element_rect(fill = "#EAEAF2", color = NA)
  ) +
  labs(title = "Generalized Cross-Validation (GCV)",
       subtitle = "Ricerca del parametro di penalizzazione λ che minimizza l'errore di generalizzazione",
       x = "Parametro λ (Scala Logaritmica)", y = "Punteggio GCV")

ggsave("slide2_gcv_curve.png", plot = p4, width = 9, height = 5, dpi = 350, bg = "white")

cat("✅ Aggiunte didattiche completate:\n")
cat("   → slide1_b_spline_basis.png\n")
cat("   → slide2_gcv_curve.png\n")