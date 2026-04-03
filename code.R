# =============================================================================
# SYSTEM FOR AUTOMATIC LEARNING OF ELECTRICAL LOAD PATTERNS
# AND POWER SYSTEM SIZING — FULL ANALYSIS 
# Data Science  | Reg No.: Puneeth 23BCE0191, Sanjay 23BCE0211,Sai Krishna 23BCE2005,Harshil 23BCE0900
# =============================================================================
# Dataset : UCI Individual Household Electric Power Consumption
# Source  : https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
# Direct  : https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip
# Citation: Hebrail, G., & Berard, A. (2012). UCI Machine Learning Repository.
# =============================================================================

# ── 0. Install & Load Libraries ───────────────────────────────────────────────
packages <- c(
  "tidyverse",    # Data wrangling & ggplot2
  "lubridate",    # Date-time parsing
  "forecast",     # ARIMA / ETS (load BEFORE Metrics to avoid accuracy() clash)
  "tseries",      # ADF / KPSS stationarity tests
  "cluster",      # K-means + silhouette
  "factoextra",   # Cluster & PCA visualisation
  "corrplot",     # Correlation matrix heat-map
  "ggcorrplot",   # ggplot2 correlation with p-values
  "GGally",       # Pairwise scatter matrix
  "scales",       # Axis formatting helpers
  "gridExtra",    # Multi-panel plot layout
  "ggthemes",     # Extra ggplot2 themes
  "zoo",          # Rolling window functions
  "moments",      # Skewness & kurtosis
  "changepoint",  # Change-point detection (PELT)
  "viridis",      # Perceptually uniform palettes
  "randomForest", # Random Forest regression
  "Metrics"       # RMSE / MAE (load AFTER forecast)
)

installed <- rownames(installed.packages())
to_install <- packages[!packages %in% installed]
if (length(to_install) > 0)
  install.packages(to_install, repos = "https://cran.r-project.org", quiet = TRUE)

# Load in order; keep forecast::accuracy accessible via namespace
invisible(lapply(packages, library, character.only = TRUE))

# Fix accuracy() namespace clash: Metrics masks forecast's accuracy()
# We'll call forecast::accuracy() explicitly throughout
cat("✅ All libraries loaded.\n\n")

# ── 1. Download & Load Dataset ────────────────────────────────────────────────
cat("📥 Downloading UCI Household Power Consumption dataset...\n")

url     <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
zipfile <- tempfile(fileext = ".zip")
datadir <- tempdir()

tryCatch({
  download.file(url, destfile = zipfile, mode = "wb", quiet = TRUE)
  unzip(zipfile, exdir = datadir)
  cat("✅ Download successful.\n\n")
}, error = function(e) stop("❌ Download failed: ", e$message))

raw <- read.table(
  file.path(datadir, "household_power_consumption.txt"),
  sep = ";", header = TRUE,
  na.strings = "?", stringsAsFactors = FALSE
)
cat(sprintf("✅ Raw data: %d rows × %d columns\n\n", nrow(raw), ncol(raw)))

# ── 2. Pre-processing & Feature Engineering ───────────────────────────────────
cat("🔧 Pre-processing & engineering features...\n")

df <- raw %>%
  mutate(
    datetime = dmy_hms(paste(Date, Time)),
    Date     = as.Date(Date, "%d/%m/%Y"),
    across(c(Global_active_power, Global_reactive_power,
             Voltage, Global_intensity,
             Sub_metering_1, Sub_metering_2, Sub_metering_3), as.numeric)
  ) %>%
  filter(!is.na(Global_active_power)) %>%
  mutate(
    year        = year(datetime),
    month       = month(datetime, label = TRUE),
    month_num   = month(datetime),
    week        = week(datetime),
    day         = day(datetime),
    hour        = hour(datetime),
    minute      = minute(datetime),
    weekday     = wday(datetime, label = TRUE, week_start = 1),
    weekday_num = wday(datetime, week_start = 1),
    is_weekend  = ifelse(wday(datetime) %in% c(1, 7), "Weekend", "Weekday"),
    season      = case_when(
      month_num %in% c(12, 1, 2) ~ "Winter",
      month_num %in% c(3, 4, 5)  ~ "Spring",
      month_num %in% c(6, 7, 8)  ~ "Summer",
      TRUE                        ~ "Autumn"
    ),
    time_of_day = case_when(
      hour >= 6  & hour < 12 ~ "Morning",
      hour >= 12 & hour < 17 ~ "Afternoon",
      hour >= 17 & hour < 21 ~ "Evening",
      TRUE                    ~ "Night"
    ),
    energy_kwh        = Global_active_power / 60,
    apparent_power    = sqrt(Global_active_power^2 + Global_reactive_power^2),
    power_factor      = ifelse(apparent_power > 0,
                               Global_active_power / apparent_power, NA),
    reactive_ratio    = Global_reactive_power / (Global_active_power + 0.001),
    other_submetering = pmax(
      (Global_active_power * 1000 / 60) -
        Sub_metering_1 - Sub_metering_2 - Sub_metering_3, 0)
  )

cat(sprintf("✅ Clean dataset: %d rows × %d columns\n\n", nrow(df), ncol(df)))

# ── 3. EDA – Summary Statistics ───────────────────────────────────────────────
cat("📊 Section 3: Descriptive Statistics & Normality Tests\n\n")

key_vars <- c("Global_active_power", "Global_reactive_power",
              "Voltage", "Global_intensity", "energy_kwh", "power_factor")

desc_stats <- df %>%
  select(all_of(key_vars)) %>%
  summarise(across(everything(), list(
    n      = ~sum(!is.na(.)),
    min    = ~min(.,    na.rm = TRUE),
    Q1     = ~quantile(., 0.25, na.rm = TRUE),
    mean   = ~mean(.,   na.rm = TRUE),
    median = ~median(., na.rm = TRUE),
    Q3     = ~quantile(., 0.75, na.rm = TRUE),
    max    = ~max(.,    na.rm = TRUE),
    sd     = ~sd(.,     na.rm = TRUE),
    cv_pct = ~sd(., na.rm = TRUE) / mean(., na.rm = TRUE) * 100,
    skew   = ~moments::skewness(., na.rm = TRUE),
    kurt   = ~moments::kurtosis(., na.rm = TRUE)
  )))

cat("── Descriptive Statistics (transposed) ─────────────────────────────────────\n")
print(as.data.frame(t(desc_stats)), digits = 4)

cat("\n── Missing Value Report ────────────────────────────────────────────────────\n")
miss_df <- data.frame(
  Variable = names(df),
  Missing  = colSums(is.na(df)),
  Pct      = round(colSums(is.na(df)) / nrow(df) * 100, 2)
) %>% filter(Missing > 0)
if (nrow(miss_df) == 0) cat("  No missing values remaining.\n") else print(miss_df)

cat("\n── Shapiro-Wilk Normality Test (n = 5000 sample) ───────────────────────────\n")
for (v in key_vars[1:4]) {
  s   <- sample(na.omit(df[[v]]), 5000)
  res <- shapiro.test(s)
  cat(sprintf("  %-28s W=%.4f  p=%.2e  → %s\n", v, res$statistic, res$p.value,
              ifelse(res$p.value < 0.05, "NON-NORMAL ❌", "NORMAL ✅")))
}

cat("\n── Kruskal-Wallis Test: Power across Seasons ───────────────────────────────\n")
kw <- kruskal.test(Global_active_power ~ season, data = df %>% sample_n(20000))
cat(sprintf("  χ²=%.2f, df=%d, p=%.2e → %s\n", kw$statistic, kw$parameter, kw$p.value,
            ifelse(kw$p.value < 0.05,
                   "Significant seasonal difference ✅",
                   "No significant difference")))

cat("\n── Kruskal-Wallis Test: Power across Time-of-Day ───────────────────────────\n")
kw2 <- kruskal.test(Global_active_power ~ time_of_day, data = df %>% sample_n(20000))
cat(sprintf("  χ²=%.2f, df=%d, p=%.2e → %s\n", kw2$statistic, kw2$parameter, kw2$p.value,
            ifelse(kw2$p.value < 0.05, "Significant ✅", "Not significant")))

# ── 4. Aggregations ───────────────────────────────────────────────────────────
cat("\n🔢 Section 4: Aggregations\n")

daily <- df %>%
  group_by(Date, is_weekend, season) %>%
  summarise(
    total_kwh    = sum(energy_kwh,              na.rm = TRUE),
    mean_power   = mean(Global_active_power,    na.rm = TRUE),
    peak_power   = max(Global_active_power,     na.rm = TRUE),
    mean_voltage = mean(Voltage,                na.rm = TRUE),
    sd_voltage   = sd(Voltage,                  na.rm = TRUE),
    mean_pf      = mean(power_factor,           na.rm = TRUE),
    mean_react   = mean(Global_reactive_power,  na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(Date) %>%
  mutate(
    rolling_avg_7  = rollmean(total_kwh, 7,  fill = NA, align = "right"),
    rolling_avg_30 = rollmean(total_kwh, 30, fill = NA, align = "right"),
    rolling_sd_7   = rollapply(total_kwh, 7, sd, fill = NA, align = "right"),
    day_of_year    = yday(Date),
    week_of_year   = week(Date),
    year           = year(Date),
    month          = month(Date, label = TRUE),
    log_kwh        = log1p(total_kwh)
  )

monthly <- df %>%
  group_by(year, month, month_num, season) %>%
  summarise(
    total_kwh  = sum(energy_kwh,           na.rm = TRUE),
    mean_power = mean(Global_active_power, na.rm = TRUE),
    peak_power = max(Global_active_power,  na.rm = TRUE),
    min_power  = min(Global_active_power,  na.rm = TRUE),
    mean_pf    = mean(power_factor,        na.rm = TRUE),
    .groups = "drop"
  )

hourly_profile <- df %>%
  group_by(hour, is_weekend, season) %>%
  summarise(
    mean_power = mean(Global_active_power, na.rm = TRUE),
    sd_power   = sd(Global_active_power,   na.rm = TRUE),
    p90_power  = quantile(Global_active_power, 0.90, na.rm = TRUE),
    .groups = "drop"
  )

seasonal_hour <- df %>%
  group_by(season, hour) %>%
  summarise(mean_power = mean(Global_active_power, na.rm = TRUE), .groups = "drop")

tod_summary <- df %>%
  group_by(time_of_day, season) %>%
  summarise(
    mean_power = mean(Global_active_power, na.rm = TRUE),
    total_kwh  = sum(energy_kwh,           na.rm = TRUE),
    .groups = "drop"
  )

cat("✅ Aggregations complete.\n")

# ── 5. Visualisations ─────────────────────────────────────────────────────────
cat("\n🎨 Section 5: Visualisations (12 plots)\n")

theme_load <- theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", size = 13),
        plot.subtitle = element_text(size = 9,  colour = "grey40"),
        axis.title    = element_text(size = 11),
        legend.position = "bottom")

# P1 – Daily energy with 7/30-day rolling averages
p1 <- ggplot(daily, aes(x = Date, y = total_kwh)) +
  geom_line(colour = "steelblue", alpha = 0.35, linewidth = 0.25) +
  geom_line(aes(y = rolling_avg_7),  colour = "darkorange", linewidth = 0.9) +
  geom_line(aes(y = rolling_avg_30), colour = "darkred",    linewidth = 1.1) +
  labs(title    = "Daily Household Energy Consumption",
       subtitle = "Blue=raw | Orange=7-day MA | Red=30-day MA",
       x = NULL, y = "Energy (kWh)") + theme_load

p1

# P2 – Hourly winter load profile: weekday vs weekend
p2 <- ggplot(hourly_profile %>% filter(season == "Winter"),
             aes(x = hour, y = mean_power, colour = is_weekend, fill = is_weekend)) +
  geom_ribbon(aes(ymin = mean_power - sd_power, ymax = mean_power + sd_power),
              alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1.2) +
  scale_x_continuous(breaks = 0:23) +
  scale_colour_manual(values = c("Weekday" = "steelblue", "Weekend" = "tomato")) +
  scale_fill_manual(values   = c("Weekday" = "steelblue", "Weekend" = "tomato")) +
  labs(title = "Hourly Load Profile – Winter", subtitle = "Ribbon = ±1 SD",
       x = "Hour", y = "Active Power (kW)", colour = NULL, fill = NULL) + theme_load

p2

# P3 – Monthly energy heatmap
p3 <- ggplot(monthly, aes(x = month, y = factor(year), fill = total_kwh)) +
  geom_tile(colour = "white") +
  scale_fill_viridis_c(name = "kWh", option = "C") +
  labs(title = "Monthly Energy Consumption Heatmap",
       x = "Month", y = "Year") + theme_load

p3

# P4 – Active power distribution with mean/median lines
p4 <- ggplot(df %>% sample_n(60000), aes(x = Global_active_power)) +
  geom_histogram(bins = 100, fill = "steelblue", colour = "white", alpha = 0.8) +
  geom_vline(xintercept = mean(df$Global_active_power, na.rm = TRUE),
             colour = "red",       linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = median(df$Global_active_power, na.rm = TRUE),
             colour = "darkgreen", linetype = "dashed", linewidth = 1) +
  labs(title    = "Active Power Distribution",
       subtitle = "Red=mean | Green=median",
       x = "Active Power (kW)", y = "Count") + theme_load

p4

# P5 – Sub-metering stacked bar (2008)
sub_long <- df %>%
  filter(year == 2008) %>%
  group_by(month) %>%
  summarise(
    Kitchen = sum(Sub_metering_1,     na.rm = TRUE) / 1000,
    Laundry = sum(Sub_metering_2,     na.rm = TRUE) / 1000,
    HVAC    = sum(Sub_metering_3,     na.rm = TRUE) / 1000,
    Other   = sum(other_submetering,  na.rm = TRUE) / 1000,
    .groups = "drop"
  ) %>%
  pivot_longer(-month, names_to = "Category", values_to = "kWh")

p5 <- ggplot(sub_long, aes(x = month, y = kWh, fill = Category)) +
  geom_col(position = "stack") +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Monthly Sub-Metering Breakdown (2008)",
       x = "Month", y = "Energy (kWh)", fill = NULL) + theme_load

p5

# P6 – Power factor density
p6 <- ggplot(df %>% filter(!is.na(power_factor)), aes(x = power_factor)) +
  geom_density(fill = "mediumorchid", alpha = 0.6) +
  geom_vline(xintercept = 0.9, colour = "red", linetype = "dashed") +
  labs(title    = "Power Factor Distribution",
       subtitle = "Red dashed = 0.9 utility threshold",
       x = "Power Factor", y = "Density") + theme_load
p6

# P7 – Seasonal hourly heatmap
p7 <- ggplot(seasonal_hour, aes(x = hour, y = season, fill = mean_power)) +
  geom_tile() +
  scale_fill_viridis_c(name = "kW", option = "B") +
  labs(title = "Seasonal Hourly Load Heatmap",
       x = "Hour of Day", y = "Season") + theme_load

p7

# P8 – Time-of-day mean power by season
p8 <- ggplot(tod_summary, aes(x = time_of_day, y = mean_power, fill = season)) +
  geom_col(position = "dodge") +
  scale_fill_brewer(palette = "Spectral") +
  labs(title = "Average Power by Time-of-Day & Season",
       x = "Period", y = "Mean Power (kW)", fill = "Season") + theme_load
p8
# P9 – 7-day rolling volatility (SD) coloured by season
p9 <- ggplot(daily %>% filter(!is.na(rolling_sd_7)),
             aes(x = Date, y = rolling_sd_7, colour = season)) +
  geom_line(linewidth = 0.7) +
  scale_colour_brewer(palette = "Dark2") +
  labs(title = "7-Day Rolling Volatility (SD of Daily kWh)",
       x = NULL, y = "SD (kWh)", colour = "Season") + theme_load

p9

# P10 – Boxplot by day of week
p10 <- ggplot(df %>% sample_n(80000),
              aes(x = weekday, y = Global_active_power, fill = weekday)) +
  geom_boxplot(outlier.size = 0.3, outlier.alpha = 0.2) +
  scale_fill_viridis_d() +
  labs(title = "Active Power Distribution by Day of Week",
       x = "Day", y = "Active Power (kW)") +
  theme_load + theme(legend.position = "none")

p10

# P11 – Violin + boxplot by season
p11 <- ggplot(df %>% sample_n(80000),
              aes(x = season, y = Global_active_power, fill = season)) +
  geom_violin(trim = FALSE, alpha = 0.7) +
  geom_boxplot(width = 0.08, fill = "white", outlier.size = 0.2) +
  scale_fill_brewer(palette = "Pastel1") +
  labs(title = "Power Distribution by Season (Violin + Box)",
       x = "Season", y = "Active Power (kW)") +
  theme_load + theme(legend.position = "none")

p11

# P12 – Voltage vs active power with per-season regression lines
p12 <- ggplot(df %>% sample_n(20000),
              aes(x = Voltage, y = Global_active_power, colour = season)) +
  geom_point(alpha = 0.2, size = 0.5) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 1.2) +
  scale_colour_brewer(palette = "Set1") +
  labs(title = "Voltage vs Active Power (by Season)",
       x = "Voltage (V)", y = "Active Power (kW)", colour = "Season") + theme_load

p12

# Render all plots
print(p1); print(p2)
grid.arrange(p3, p4, ncol = 2)
grid.arrange(p5, p6, ncol = 2)
grid.arrange(p7, p8, ncol = 2)
grid.arrange(p9, p10, ncol = 2)
grid.arrange(p11, p12, ncol = 2)
cat("✅ 12 visualisations rendered.\n")

# ── 6. Correlation Analysis ───────────────────────────────────────────────────
cat("\n🔗 Section 6: Correlation & Pair Analysis\n")

corr_df <- df %>%
  select(Global_active_power, Global_reactive_power,
         Voltage, Global_intensity,
         Sub_metering_1, Sub_metering_2, Sub_metering_3, power_factor) %>%
  drop_na()

corr_mat <- cor(corr_df)
cat("Full correlation matrix:\n")
print(round(corr_mat, 3))

# Base corrplot
corrplot(corr_mat, method = "color", type = "upper",
         tl.cex = 0.8, addCoef.col = "black", number.cex = 0.65,
         title  = "Electrical Variable Correlation", mar = c(0, 0, 2, 0))

# ggcorrplot with p-value significance blanking
p_mat <- cor_pmat(corr_df)
ggcorrplot(corr_mat, hc.order = TRUE, type = "lower",
           lab = TRUE, lab_size = 3,
           p.mat = p_mat, sig.level = 0.05, insig = "blank",
           title = "Correlation Matrix (blanked if p > 0.05)")

# GGpairs pairwise scatter/density matrix
ggpairs(
  df %>% sample_n(3000) %>%
    select(Global_active_power, Global_reactive_power,
           Voltage, Global_intensity, power_factor),
  title = "Pairwise Scatter & Density Matrix",
  lower = list(continuous = wrap("points", alpha = 0.15, size = 0.4)),
  diag  = list(continuous = wrap("densityDiag", fill = "steelblue", alpha = 0.6))
)

# ── 7. K-Means Clustering – FIXED silhouette ──────────────────────────────────
cat("\n🔍 Section 7: K-Means Load Pattern Clustering\n")

hourly_wide <- df %>%
  group_by(Date, hour) %>%
  summarise(mean_power = mean(Global_active_power, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = hour, values_from = mean_power, names_prefix = "h") %>%
  drop_na()

cm <- scale(as.matrix(hourly_wide[ , -1]))   # matrix, not data frame
n_rows <- nrow(cm)

# Elbow method
set.seed(42)
wss <- sapply(1:10, function(k) {
  kmeans(cm, centers = k, nstart = 25, iter.max = 50)$tot.withinss
})

# FIXED silhouette: sample SAME rows for both cluster vector & dist matrix
SIL_N <- min(500, n_rows)

set.seed(42)
sil_scores <- sapply(2:8, function(k) {
  km_full   <- kmeans(cm, centers = k, nstart = 25, iter.max = 50)
  samp_idx  <- sample(n_rows, SIL_N)
  sil_obj   <- silhouette(
    x    = km_full$cluster[samp_idx],
    dist = dist(cm[samp_idx, ])
  )
  mean(sil_obj[, 3])
})

# Plot elbow + silhouette
elbow_df <- data.frame(k = 1:10, wss = wss)
sil_df   <- data.frame(k = 2:8,  sil = sil_scores)

pe <- ggplot(elbow_df, aes(k, wss)) +
  geom_line(colour = "steelblue", linewidth = 1.2) +
  geom_point(size = 3, colour = "steelblue") +
  labs(title = "Elbow Method", x = "k", y = "Within-cluster SS") + theme_load

ps <- ggplot(sil_df, aes(k, sil)) +
  geom_line(colour = "tomato", linewidth = 1.2) +
  geom_point(size = 3, colour = "tomato") +
  labs(title = "Silhouette Score", x = "k", y = "Avg Silhouette Width") + theme_load

grid.arrange(pe, ps, ncol = 2)

best_k <- sil_df$k[which.max(sil_df$sil)]
cat(sprintf("Elbow suggests k ≈ 3. Silhouette best k = %d.\n", best_k))
cat("Proceeding with k = 3 (Low / Medium / High load days).\n")

# Final k=3 model
set.seed(42)
km3 <- kmeans(cm, centers = 3, nstart = 50, iter.max = 100)
hourly_wide$cluster <- factor(km3$cluster)
cat(sprintf("Cluster sizes: %s\n",
            paste(names(table(km3$cluster)),
                  table(km3$cluster), sep = "=", collapse = " | ")))

# Cluster daily-profile plot
cluster_long <- hourly_wide %>%
  pivot_longer(starts_with("h"), names_to = "hour_str", values_to = "power") %>%
  mutate(hour = as.integer(sub("h", "", hour_str))) %>%
  group_by(cluster, hour) %>%
  summarise(mean_power = mean(power, na.rm = TRUE), .groups = "drop")

ggplot(cluster_long, aes(x = hour, y = mean_power,
                         colour = cluster, group = cluster)) +
  geom_line(linewidth = 1.4) +
  scale_x_continuous(breaks = 0:23) +
  scale_colour_brewer(palette = "Set1") +
  labs(title    = "Daily Load Profiles by Cluster",
       subtitle = "3 clusters: Low / Medium / High consumption",
       x = "Hour of Day", y = "Mean Active Power (kW)", colour = "Cluster") +
  theme_load

# PCA cluster visualisation
pca_res <- prcomp(cm, scale. = FALSE)
pca_df  <- data.frame(pca_res$x[, 1:2], cluster = hourly_wide$cluster)
pve     <- summary(pca_res)$importance[2, 1:2] * 100

ggplot(pca_df, aes(PC1, PC2, colour = cluster)) +
  geom_point(alpha = 0.4, size = 0.9) +
  stat_ellipse(linewidth = 1.2) +
  scale_colour_brewer(palette = "Set1") +
  labs(title    = "PCA Cluster Visualisation",
       subtitle = sprintf("PC1=%.1f%% | PC2=%.1f%% variance explained",
                          pve[1], pve[2]),
       colour = "Cluster") + theme_load

# ── 8. Hierarchical Clustering (dendrogram) ───────────────────────────────────
cat("\n🌳 Section 8: Hierarchical Clustering\n")

set.seed(42)
hc_idx    <- sample(n_rows, 200)
hc_sample <- cm[hc_idx, ]
hc_dist   <- dist(hc_sample, method = "euclidean")
hc_model  <- hclust(hc_dist, method = "ward.D2")

plot(hc_model,
     labels = FALSE,
     main   = "Hierarchical Clustering Dendrogram (Ward D2, n=200)",
     xlab   = "Day Sample", ylab = "Height")
rect.hclust(hc_model, k = 3, border = c("blue", "tomato", "seagreen"))
cat("Dendrogram rendered (3 groups highlighted).\n")

# ── 9. Time Series: Stationarity & Decomposition ─────────────────────────────
cat("\n📈 Section 9: Time Series Analysis\n")

daily_ts <- ts(daily$total_kwh,
               start     = c(year(min(daily$Date)), yday(min(daily$Date))),
               frequency = 365)

# ADF Test
adf_r <- adf.test(na.omit(daily_ts))
cat(sprintf("ADF:  stat=%.4f, p=%.4f → %s\n",
            adf_r$statistic, adf_r$p.value,
            ifelse(adf_r$p.value < 0.05, "STATIONARY ✅", "NON-STATIONARY ⚠️")))

# KPSS Test
kpss_r <- kpss.test(na.omit(daily_ts))
cat(sprintf("KPSS: stat=%.4f, p=%.4f → %s\n",
            kpss_r$statistic, kpss_r$p.value,
            ifelse(kpss_r$p.value > 0.05, "STATIONARY ✅", "NON-STATIONARY ⚠️")))

# STL decomposition
stl_d <- stl(na.approx(daily_ts), s.window = "periodic")
plot(stl_d, main = "STL Decomposition – Daily Energy Consumption")

# Differenced series
diff_ts  <- diff(na.approx(daily_ts))
adf_diff <- adf.test(diff_ts)
cat(sprintf("ADF (differenced): p=%.4f → %s\n",
            adf_diff$p.value,
            ifelse(adf_diff$p.value < 0.05, "STATIONARY ✅", "NON-STATIONARY ⚠️")))

# ACF / PACF – 2×2 panel
par(mfrow = c(2, 2))
acf(na.omit(daily_ts),  lag.max = 60, main = "ACF – Daily kWh")
pacf(na.omit(daily_ts), lag.max = 60, main = "PACF – Daily kWh")
acf(diff_ts,            lag.max = 60, main = "ACF – Differenced")
pacf(diff_ts,           lag.max = 60, main = "PACF – Differenced")
par(mfrow = c(1, 1))

# ── 10. Change-Point Detection ────────────────────────────────────────────────
# FIX: switched penalty from "BIC" to "MBIC" to prevent near-every-day
#      false detections on a high-frequency ts (frequency = 365).
cat("\n🔎 Section 10: Change-Point Detection (PELT / MBIC)\n")

ts_clean <- na.approx(daily_ts)

cpt_m <- cpt.mean(ts_clean, method = "PELT", penalty = "MBIC")
cpt_v <- cpt.var(ts_clean,  method = "PELT", penalty = "MBIC")

cat(sprintf("Mean change-points detected   : %d\n", length(cpts(cpt_m))))
cat(sprintf("Mean change-points at index   : %s\n",
            paste(cpts(cpt_m), collapse = ", ")))
cat(sprintf("Variance change-points at index: %s\n",
            paste(cpts(cpt_v), collapse = ", ")))

plot(cpt_m,
     main = "Change-Point Detection – Mean (PELT/MBIC)",
     xlab = "Day Index", ylab = "Daily kWh", cpt.col = "red")

plot(cpt_v,
     main = "Change-Point Detection – Variance (PELT/MBIC)",
     xlab = "Day Index", ylab = "Daily kWh", cpt.col = "darkorange")

# ── 11. Anomaly Detection ─────────────────────────────────────────────────────
cat("\n🚨 Section 11: Anomaly Detection (Z-score & IQR)\n")

q1   <- quantile(daily$total_kwh, 0.25, na.rm = TRUE)
q3   <- quantile(daily$total_kwh, 0.75, na.rm = TRUE)
iqr  <- q3 - q1
mu   <- mean(daily$total_kwh, na.rm = TRUE)
sgm  <- sd(daily$total_kwh,   na.rm = TRUE)

anom_df <- daily %>%
  mutate(
    z_score     = (total_kwh - mu) / sgm,
    anomaly_z   = abs(z_score) > 3,
    anomaly_iqr = total_kwh < (q1 - 1.5 * iqr) | total_kwh > (q3 + 1.5 * iqr)
  )

cat(sprintf("Z-score anomalies (|z|>3) : %d days (%.1f%%)\n",
            sum(anom_df$anomaly_z),
            sum(anom_df$anomaly_z) / nrow(anom_df) * 100))
cat(sprintf("IQR anomalies             : %d days (%.1f%%)\n",
            sum(anom_df$anomaly_iqr),
            sum(anom_df$anomaly_iqr) / nrow(anom_df) * 100))

cat("\nTop 10 anomalous days (Z-score):\n")
print(anom_df %>% filter(anomaly_z) %>%
        arrange(desc(abs(z_score))) %>%
        select(Date, total_kwh, z_score) %>% head(10))

ggplot(anom_df, aes(x = Date, y = total_kwh, colour = anomaly_z)) +
  geom_point(size = 0.8, alpha = 0.7) +
  geom_hline(yintercept = mu + 3 * sgm, linetype = "dashed", colour = "red") +
  geom_hline(yintercept = mu - 3 * sgm, linetype = "dashed", colour = "red") +
  scale_colour_manual(values = c("FALSE" = "steelblue", "TRUE" = "red"),
                      labels = c("Normal", "Anomaly (|z|>3)")) +
  labs(title = "Anomaly Detection – Z-Score Method",
       x = NULL, y = "Daily kWh", colour = NULL) + theme_load

# ── 12. ARIMA Forecasting ─────────────────────────────────────────────────────
cat("\n🔮 Section 12: ARIMA Forecasting\n")

train_ts <- window(daily_ts, end   = c(2009, 365))
test_ts  <- window(daily_ts, start = c(2010, 1))

arima_m  <- auto.arima(na.approx(train_ts),
                       seasonal      = TRUE,
                       stepwise      = FALSE,
                       approximation = FALSE)
cat("Best ARIMA model:\n"); print(arima_m)

arima_fc <- forecast(arima_m, h = 90)
plot(arima_fc,
     main = "ARIMA 90-Day Energy Consumption Forecast",
     xlab = "Time", ylab = "Daily kWh")

acc_a <- forecast::accuracy(arima_fc, na.approx(test_ts))
cat("\nARIMA Forecast Accuracy:\n"); print(round(acc_a, 4))

checkresiduals(arima_m)
lb_p <- Box.test(residuals(arima_m), lag = 20, type = "Ljung-Box")$p.value
cat(sprintf("Ljung-Box p=%.4f → %s\n", lb_p,
            ifelse(lb_p > 0.05, "Residuals are white noise ✅",
                   "Residual autocorrelation detected ⚠️")))

# ── 13. ETS Forecasting ───────────────────────────────────────────────────────
# FIX: replaced ets() with stlf() so seasonality is captured correctly.
#      ets() silently drops seasonality when frequency > 24.
cat("\n🔮 Section 13: ETS (Exponential Smoothing State Space) Forecasting\n")

ets_fc <- stlf(na.approx(train_ts), h = 90)   # STL decomposition + ETS
ets_m  <- ets_fc$model                          # underlying ETS model object
cat("ETS model (via stlf):\n"); print(ets_m)
plot(ets_fc, main = "ETS 90-Day Forecast (stlf)", xlab = "Time", ylab = "Daily kWh")

acc_e <- forecast::accuracy(ets_fc, na.approx(test_ts))
cat("\nETS Accuracy:\n"); print(round(acc_e, 4))

# ── 14. TBATS Forecasting (handles complex seasonality) ───────────────────────
cat("\n🔮 Section 14: TBATS Forecast\n")

tbats_m  <- tbats(na.approx(train_ts))
tbats_fc <- forecast(tbats_m, h = 90)
plot(tbats_fc, main = "TBATS 90-Day Forecast", xlab = "Time", ylab = "Daily kWh")

acc_tb <- forecast::accuracy(tbats_fc, na.approx(test_ts))
cat("\nTBATS Accuracy:\n"); print(round(acc_tb, 4))

# ── 15. Model Comparison ─────────────────────────────────────────────────────
cat("\n📊 Section 15: Forecasting Model Comparison\n")

comp_df <- data.frame(
  Model = c("ARIMA", "ETS", "TBATS"),
  RMSE  = c(acc_a["Test set", "RMSE"],
            acc_e["Test set", "RMSE"],
            acc_tb["Test set", "RMSE"]),
  MAE   = c(acc_a["Test set", "MAE"],
            acc_e["Test set", "MAE"],
            acc_tb["Test set", "MAE"]),
  MAPE  = c(acc_a["Test set", "MAPE"],
            acc_e["Test set", "MAPE"],
            acc_tb["Test set", "MAPE"])
)
cat("Model Comparison Table:\n"); print(comp_df)
cat(sprintf("\n🏆 Best model by MAPE: %s\n",
            comp_df$Model[which.min(comp_df$MAPE)]))

comp_long <- comp_df %>%
  pivot_longer(-Model, names_to = "Metric", values_to = "Value")

ggplot(comp_long, aes(x = Model, y = Value, fill = Model)) +
  geom_col() +
  facet_wrap(~Metric, scales = "free_y") +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Forecasting Model Accuracy Comparison",
       x = NULL, y = "Value") +
  theme_load + theme(legend.position = "none")

# ── 16. Random Forest Regression ─────────────────────────────────────────────
cat("\n🌲 Section 16: Random Forest Load Prediction\n")

rf_df <- daily %>%
  mutate(
    dow     = as.integer(wday(Date, week_start = 1)),
    month_n = as.integer(month(Date)),
    week_n  = week(Date),
    lag1    = dplyr::lag(total_kwh, 1),
    lag7    = dplyr::lag(total_kwh, 7),
    lag30   = dplyr::lag(total_kwh, 30)
  ) %>%
  select(total_kwh, dow, month_n, week_n, day_of_year,
         lag1, lag7, lag30, mean_voltage, mean_pf, mean_react) %>%
  drop_na()

set.seed(42)
train_idx  <- sample(seq_len(nrow(rf_df)), size = 0.80 * nrow(rf_df))
rf_train   <- rf_df[ train_idx, ]
rf_test    <- rf_df[-train_idx, ]

rf_model <- randomForest(
  total_kwh ~ ., data = rf_train,
  ntree = 300, mtry = 4, importance = TRUE
)
cat("Random Forest summary:\n"); print(rf_model)

rf_pred <- predict(rf_model, rf_test)
rf_rmse <- Metrics::rmse(rf_test$total_kwh, rf_pred)
rf_mae  <- Metrics::mae(rf_test$total_kwh,  rf_pred)
rf_mape <- mean(abs((rf_test$total_kwh - rf_pred) /
                      rf_test$total_kwh)) * 100
rf_r2   <- cor(rf_test$total_kwh, rf_pred)^2

cat(sprintf("RF → RMSE=%.3f | MAE=%.3f | MAPE=%.2f%% | R²=%.4f\n",
            rf_rmse, rf_mae, rf_mape, rf_r2))

# Variable importance
imp_df <- data.frame(
  Variable  = rownames(importance(rf_model)),
  IncPurity = importance(rf_model)[, "IncNodePurity"]
) %>% arrange(desc(IncPurity))

ggplot(imp_df, aes(x = reorder(Variable, IncPurity),
                   y = IncPurity, fill = IncPurity)) +
  geom_col() + coord_flip() +
  scale_fill_viridis_c(direction = -1) +
  labs(title = "Random Forest Variable Importance",
       x = "Feature", y = "IncNodePurity") +
  theme_load + theme(legend.position = "none")

# Actual vs Predicted
ggplot(data.frame(Actual = rf_test$total_kwh, Predicted = rf_pred),
       aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.4, size = 1, colour = "steelblue") +
  geom_abline(slope = 1, intercept = 0, colour = "red", linetype = "dashed") +
  labs(title    = sprintf("RF: Actual vs Predicted (R²=%.4f)", rf_r2),
       subtitle = sprintf("RMSE=%.3f | MAE=%.3f | MAPE=%.2f%%",
                          rf_rmse, rf_mae, rf_mape),
       x = "Actual kWh", y = "Predicted kWh") + theme_load

# ── 17. Multiple Linear Regression ───────────────────────────────────────────
cat("\n📐 Section 17: Multiple Linear Regression\n")

lm_df <- daily %>%
  mutate(
    dow     = as.integer(wday(Date, week_start = 1)),
    month_n = as.integer(month(Date)),
    lag1    = dplyr::lag(total_kwh, 1),
    lag7    = dplyr::lag(total_kwh, 7)
  ) %>%
  select(total_kwh, dow, month_n, day_of_year,
         lag1, lag7, mean_voltage, mean_pf, mean_react) %>%
  drop_na()

lm_idx   <- sample(seq_len(nrow(lm_df)), size = 0.80 * nrow(lm_df))
lm_model <- lm(total_kwh ~ ., data = lm_df[lm_idx, ])
cat("\nLinear Regression Summary:\n"); print(summary(lm_model))

lm_pred   <- predict(lm_model, lm_df[-lm_idx, ])
# FIX: pull() extracts a plain numeric vector from the tibble column;
#      the original lm_df[-lm_idx, "total_kwh"] returned a 1-column tibble
#      which caused Metrics::rmse() to return NA silently.
lm_actual <- lm_df[-lm_idx, ] %>% pull(total_kwh)
lm_rmse   <- Metrics::rmse(lm_actual, lm_pred)
lm_r2     <- summary(lm_model)$r.squared
cat(sprintf("LM → RMSE=%.3f | R²=%.4f\n", lm_rmse, lm_r2))

par(mfrow = c(2, 2))
plot(lm_model, main = "Linear Regression Diagnostics")
par(mfrow = c(1, 1))

# ── 18. Load Duration Curve ───────────────────────────────────────────────────
cat("\n📉 Section 18: Load Duration Curve\n")

ldc_df <- df %>%
  group_by(hour_block = floor_date(datetime, "hour")) %>%
  summarise(mean_power = mean(Global_active_power, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mean_power)) %>%
  mutate(pct_time = row_number() / n() * 100)

base_load <- quantile(ldc_df$mean_power, 0.10)
avg_load  <- mean(ldc_df$mean_power, na.rm = TRUE)
pk_load   <- quantile(ldc_df$mean_power, 0.90)

cat(sprintf("Base load  (10th pctile): %.3f kW\n", base_load))
cat(sprintf("Average load            : %.3f kW\n", avg_load))
cat(sprintf("Peak load  (90th pctile): %.3f kW\n", pk_load))

ggplot(ldc_df, aes(x = pct_time, y = mean_power)) +
  geom_area(fill = "steelblue", alpha = 0.4) +
  geom_line(colour = "darkblue", linewidth = 0.8) +
  geom_hline(yintercept = pk_load,   colour = "red",    linetype = "dashed") +
  geom_hline(yintercept = avg_load,  colour = "orange", linetype = "dashed") +
  geom_hline(yintercept = base_load, colour = "green",  linetype = "dashed") +
  annotate("text", x = 80, y = pk_load   + 0.05, label = "90th pctile", colour = "red",    size = 3) +
  annotate("text", x = 80, y = avg_load  + 0.05, label = "Mean",        colour = "orange", size = 3) +
  annotate("text", x = 80, y = base_load + 0.05, label = "10th pctile", colour = "green",  size = 3) +
  labs(title    = "Load Duration Curve",
       subtitle = "% of time load is equalled or exceeded",
       x = "% of Time Load Exceeded", y = "Active Power (kW)") + theme_load

# ── 19. Energy Cost Analysis ─────────────────────────────────────────────────
cat("\n💰 Section 19: Electricity Cost Analysis\n")

# French EDF tariff approximations (€/kWh)
OFF_PEAK <- 0.1470   # Night (22:00–06:00)
PEAK     <- 0.1841   # Day
WKND     <- 0.1200   # Weekend flat

cost_df <- df %>%
  mutate(rate = case_when(
    is_weekend == "Weekend" ~ WKND,
    hour >= 22 | hour < 6  ~ OFF_PEAK,
    TRUE                    ~ PEAK
  ),
  cost_eur = energy_kwh * rate) %>%
  group_by(year, month, is_weekend) %>%
  summarise(total_kwh  = sum(energy_kwh,  na.rm = TRUE),
            total_cost = sum(cost_eur,     na.rm = TRUE), .groups = "drop")

annual_cost <- cost_df %>%
  group_by(year) %>%
  summarise(annual_kwh  = round(sum(total_kwh),  1),
            annual_cost = round(sum(total_cost),  2), .groups = "drop")

cat("Annual Energy & Cost Summary:\n")
print(annual_cost)

ggplot(cost_df, aes(x = month, y = total_cost, fill = is_weekend)) +
  geom_col(position = "stack") +
  facet_wrap(~year) +
  scale_fill_manual(values = c("Weekday" = "steelblue", "Weekend" = "tomato")) +
  scale_y_continuous(labels = label_dollar(prefix = "€")) +
  labs(title = "Monthly Electricity Cost by Year",
       x = "Month", y = "Cost (€)", fill = NULL) + theme_load

# ── 20. Peak Load Detection ───────────────────────────────────────────────────
cat("\n⚡ Section 20: Peak Load Detection & Analysis\n")

p90   <- quantile(daily$total_kwh, 0.90, na.rm = TRUE)
daily <- daily %>% mutate(is_peak = total_kwh >= p90)

peak_by_month <- daily %>%
  group_by(month = month(Date, label = TRUE)) %>%
  summarise(
    peak_days  = sum(is_peak, na.rm = TRUE),
    total_days = n(),
    peak_pct   = round(100 * peak_days / total_days, 1),
    .groups = "drop"
  )
cat("Peak day frequency (top 10% consumption) by month:\n")
print(peak_by_month)

ggplot(daily, aes(x = Date, y = total_kwh, colour = is_peak)) +
  geom_point(size = 0.7, alpha = 0.7) +
  geom_hline(yintercept = p90, linetype = "dashed", colour = "darkred") +
  scale_colour_manual(values = c("FALSE" = "steelblue", "TRUE" = "red"),
                      labels = c("Normal", "Peak Day")) +
  labs(title    = "Daily Energy – Peak Days Highlighted",
       subtitle = sprintf("Threshold = %.2f kWh (90th percentile)", p90),
       x = NULL, y = "Daily kWh", colour = NULL) + theme_load

ggplot(peak_by_month, aes(x = month, y = peak_pct, fill = peak_pct)) +
  geom_col() +
  scale_fill_viridis_c(name = "% Peak") +
  labs(title = "% of Peak Days per Month",
       x = "Month", y = "% Days Above 90th Percentile") + theme_load

# ── 21. Power System Sizing Recommendations ───────────────────────────────────
cat("\n🏗️  Section 21: Automated Power System Sizing\n")

peak_kw   <- max(df$Global_active_power,  na.rm = TRUE)
p95_kw    <- quantile(df$Global_active_power, 0.95, na.rm = TRUE)
avg_kw    <- mean(df$Global_active_power, na.rm = TRUE)
ann_kwh   <- mean(daily$total_kwh, na.rm = TRUE) * 365
load_fac  <- avg_kw / peak_kw
diversity <- 1.25

gen_kva     <- round(peak_kw * diversity / 0.80, 2)
gen_kva_p95 <- round(p95_kw  * diversity / 0.80, 2)
trafo_kva   <- round(peak_kw * 1.20, 2)
cable_230   <- round((peak_kw * 1000) / 230  * 1.25, 2)
cable_415   <- round((peak_kw * 1000) / 415  * 1.25, 2)
batt_kwh    <- round(avg_kw  * 4, 2)
solar_kwp   <- round(ann_kwh / (365 * 4.5), 2)

sizing_tbl <- data.frame(
  Component  = c("Generator (peak sizing)",
                 "Generator (P95 sizing)",
                 "Transformer",
                 "Cable – 230V single-phase",
                 "Cable – 415V three-phase",
                 "Battery Storage (4-hr backup)",
                 "Solar PV (self-consumption)"),
  Value      = c(paste(gen_kva,     "kVA"),
                 paste(gen_kva_p95, "kVA"),
                 paste(trafo_kva,   "kVA"),
                 paste(cable_230,   "A"),
                 paste(cable_415,   "A"),
                 paste(batt_kwh,    "kWh"),
                 paste(solar_kwp,   "kWp")),
  Engineering_Standard = c(
    "IEC 60034 – Peak × 1.25 diversity ÷ 0.8 PF",
    "IEC 60034 – P95 × 1.25 diversity ÷ 0.8 PF",
    "IEEE C57.91 – Peak × 1.20 safety margin",
    "NEC 230.79 – Peak kW ÷ 230V × 1.25",
    "IEC 60228 – Peak kW ÷ 415V × 1.25",
    "IEEE 1625 – Avg load × 4 h",
    "IEC 61724 – Annual kWh ÷ (365 d × 4.5 h)"
  )
)

cat("\n──────────────────────────────────────────────────────────────────────────\n")
cat("               AUTOMATED POWER SYSTEM SIZING REPORT\n")
cat("──────────────────────────────────────────────────────────────────────────\n")
cat(sprintf("  Peak Load (absolute)  : %.4f kW\n",  peak_kw))
cat(sprintf("  P95 Load              : %.4f kW\n",  p95_kw))
cat(sprintf("  Average Load          : %.4f kW\n",  avg_kw))
cat(sprintf("  Load Factor           : %.3f (%.1f%%)\n", load_fac, load_fac * 100))
cat(sprintf("  Estimated Annual kWh  : %.1f kWh\n", ann_kwh))
cat(sprintf("  Diversity Factor      : %.2f\n\n",   diversity))
print(sizing_tbl, row.names = FALSE)

# ── 22. Final Summary ─────────────────────────────────────────────────────────
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("               COMPLETE ANALYSIS – KEY FINDINGS SUMMARY\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat(sprintf("  Dataset Period       : %s → %s\n", min(df$Date), max(df$Date)))
cat(sprintf("  Total Observations   : %s\n",       format(nrow(df), big.mark = ",")))
cat(sprintf("  Load Clusters (k=3)  : Low / Medium / High consumption days\n"))
cat(sprintf("  ARIMA MAPE           : %.2f%%\n",   acc_a["Test set",  "MAPE"]))
cat(sprintf("  ETS   MAPE           : %.2f%%\n",   acc_e["Test set",  "MAPE"]))
cat(sprintf("  TBATS MAPE           : %.2f%%\n",   acc_tb["Test set", "MAPE"]))
cat(sprintf("  RF    MAPE           : %.2f%% | R²=%.4f\n", rf_mape, rf_r2))
cat(sprintf("  LM    RMSE           : %.3f | R²=%.4f\n",   lm_rmse, lm_r2))
cat(sprintf("  Best Forecaster      : %s\n",
            comp_df$Model[which.min(comp_df$MAPE)]))
cat(sprintf("  Z-score Anomalies    : %d days\n",  sum(anom_df$anomaly_z)))
cat(sprintf("  IQR Anomalies        : %d days\n",  sum(anom_df$anomaly_iqr)))
cat(sprintf("  Mean Change-Points   : %d detected\n", length(cpts(cpt_m))))
cat(sprintf("  Peak Threshold       : %.2f kWh/day (90th pctile)\n", p90))
cat(sprintf("  Generator Size       : %.2f kVA\n", gen_kva))
cat(sprintf("  Transformer Size     : %.2f kVA\n", trafo_kva))
cat(sprintf("  Battery Storage      : %.2f kWh\n", batt_kwh))
cat(sprintf("  Solar PV             : %.2f kWp\n", solar_kwp))