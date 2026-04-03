# ⚡ Automatic Learning of Electrical Load Patterns & Power System Sizing

## 📌 Project Overview

This project presents a **complete data science pipeline** for analyzing electrical consumption patterns and automatically generating **power system sizing recommendations**.

It combines:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Statistical testing
* Machine learning models
* Time series forecasting
* Power engineering calculations

📊 The system uses the **UCI Household Electric Power Consumption dataset** to extract meaningful insights and optimize energy system design.

---

## 📂 Dataset

* **Name:** Individual Household Electric Power Consumption
* **Source:** UCI Machine Learning Repository
* **Link:** https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

---

## 🚀 Features

### 🔧 1. Data Preprocessing

* Date-time parsing
* Handling missing values
* Feature engineering:

  * Energy (kWh)
  * Power factor
  * Reactive ratio
  * Time-based features (hour, season, weekday)

---

### 📊 2. Exploratory Data Analysis (EDA)

* Summary statistics (mean, median, skewness, kurtosis)
* Normality tests (Shapiro-Wilk)
* Seasonal & time-of-day analysis
* Missing value analysis

---

### 📈 3. Visualizations (12 Plots)

* Daily energy trends with rolling averages
* Hourly load profiles
* Monthly heatmaps
* Distribution plots
* Correlation plots
* Seasonal patterns

---

### 🔗 4. Correlation Analysis

* Correlation matrix
* Heatmaps with significance
* Pairwise relationships

---

### 🔍 5. Clustering

* K-Means clustering (Low / Medium / High load)
* Elbow method & silhouette score
* PCA visualization
* Hierarchical clustering (dendrogram)

---

### 📉 6. Time Series Analysis

* Stationarity tests (ADF, KPSS)
* STL decomposition
* ACF & PACF plots

---

### 🔎 7. Change Point Detection

* PELT algorithm with MBIC penalty
* Detects structural changes in energy usage

---

### 🚨 8. Anomaly Detection

* Z-score method
* IQR method
* Identifies abnormal energy consumption days

---

### 🔮 9. Forecasting Models

* ARIMA
* ETS (via STL decomposition)
* TBATS

📊 Evaluation Metrics:

* RMSE
* MAE
* MAPE

---

### 🌲 10. Machine Learning Models

#### Random Forest

* Predicts daily energy consumption
* Feature importance analysis

#### Linear Regression

* Baseline predictive model
* Diagnostic plots

---

### 📉 11. Load Duration Curve

* Base load
* Average load
* Peak load

---

### 💰 12. Energy Cost Analysis

* Time-based tariff modeling
* Monthly & annual cost estimation

---

### ⚡ 13. Peak Load Analysis

* Identifies high-demand days
* Monthly peak distribution

---

### 🏗️ 14. Power System Sizing (Key Output)

Automatically calculates:

* Generator size (kVA)
* Transformer rating
* Cable sizing (230V & 415V)
* Battery storage (kWh)
* Solar PV capacity (kWp)

📐 Based on:

* Peak load
* Diversity factor
* Engineering standards (IEC, IEEE, NEC)

---

## 🛠️ Technologies Used

* **R Programming**
* tidyverse
* ggplot2
* forecast
* randomForest
* changepoint
* cluster
* factoextra

---

## ▶️ How to Run

1. Install R (latest version)
2. Install required packages:

```r
install.packages(c("tidyverse","forecast","randomForest","changepoint"))
```

3. Run the script:

```r
source("your_script_name.R")
```

---

## 📊 Output Highlights

* 📈 12+ visualizations
* 📉 Forecast graphs
* 🔍 Clustering insights
* ⚡ Power system sizing report
* 📊 Model comparison table

---

## 🧠 Key Insights

* Load patterns vary significantly by **season and time of day**
* **Clustering** reveals distinct consumption behaviors
* **Random Forest** provides strong predictive performance
* **TBATS/ARIMA** effectively forecast energy usage
* System enables **real-world electrical infrastructure planning**

---

## 📌 Conclusion

This project demonstrates how **data science + electrical engineering** can be combined to:

✔ Understand energy consumption
✔ Predict future demand
✔ Detect anomalies
✔ Design efficient power systems

---

## 👨‍💻 Authors

* Puneeth (23BCE0191)
* Sanjay (23BCE0211)
* Sai Krishna (23BCE2005)
* Harshil (23BCE0900)

---

## 📜 License

This project is for academic and educational purposes.
