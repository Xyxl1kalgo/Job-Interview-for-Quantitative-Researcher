# 📈 Job Interview Project: Cryptocurrency Order Book Analysis & ML

Welcome to my repository for the Quantitative Researcher job interview technical task!

This project demonstrates my proficiency in data analysis, feature engineering from high-frequency market data, and machine learning model development for financial time series prediction.

---

## 🎯 Project Goal

The primary goal of this project was to:
1.  Process and analyze high-frequency order book and trade data for a cryptocurrency pair.
2.  Extract meaningful features, particularly focusing on Order Book Imbalance (OBI) at various depths.
3.  Develop a machine learning model to predict short-term price movements.

---

## 🛠️ Technologies Used

* **Python:** The core language for all data processing and ML.
* **Pandas:** For efficient data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib & Seaborn:** For data visualization and insights.
* **Scikit-learn:** For machine learning model implementation (RandomForestRegressor, RandomForestClassifier).

---

## 📁 Repository Structure

This repository contains Jupyter Notebooks, each representing a step or a specific task, along with the necessary raw data files.

* `order_book_snapshot.csv`: Raw order book snapshot data.
* `trades.csv`: Raw trade data.
* `create_data.ipynb`: (If you had a notebook for synthetic data generation or initial data setup)
* `first_task.ipynb`: Initial data loading and exploration.
* `second_task.ipynb`: (e.g., Mid-price calculation, basic OBI).
* `third_task.ipynb`: (e.g., Cumulative OBI calculation).
* `forth_task.ipynb`: (e.g., Basic visualization of OBI).
* `fivth_task.ipynb`: (e.g., Advanced OBI analysis / visualization).
* `sixth_task.ipynb`: (e.g., Price change target calculation).
* `seventth_task.ipynb`: **Machine Learning Model Development** - This notebook contains the core ML implementation, including feature engineering (OBI at various levels), defining different target variables (price change, log returns, direction), training, and evaluating regression and classification models.

---

## 🚀 Key Features & Accomplishments

* **High-Frequency Data Handling:** Demonstrated ability to process and clean tick-level order book and trade data.
* **Feature Engineering:** Successfully calculated various forms of Order Book Imbalance (OBI), including single-level and cumulative OBI at multiple depths, crucial for understanding market microstructure.
* **Target Variable Definition:** Explored different approaches to defining the target for price prediction, including continuous price change (regression) and price direction (classification).
* **Machine Learning Application:** Implemented and evaluated `RandomForestRegressor` and `RandomForestClassifier` to predict short-term price movements based on OBI features.
* **Code Structure:** Organized the project into modular Jupyter notebooks, allowing for clear progression through the analytical steps.
* **Visualization:** Utilized `Matplotlib` and `Seaborn` to create insightful plots for understanding OBI dynamics and model performance.

---

## 📊 Quick Look at ML Model Performance (from `seventth_task.ipynb`)

*(You can add a small summary or a key metric here after running the `seventth_task.ipynb`)*

**Regression Model (Predicting `mid_price_change`):**
* **Model:** RandomForestRegressor
* **Key Metric (RMSE):** [Insert your RMSE value here, e.g., `0.000123`]
* **Top 3 Important Features:**
    1.  `obi_cumulative_level_X`
    2.  `obi_level_Y`
    3.  `...`

**Classification Model (Predicting `price_direction`):**
* **Model:** RandomForestClassifier
* **Key Metric (Accuracy):** [Insert your Accuracy value here, e.g., `0.52`]
* **Classification Report Summary:**
    * Precision for Up/Down: [e.g., ~0.45]
    * Recall for Up/Down: [e.g., ~0.30]
    *(Consider adding a short sentence about what this means for your model's ability to predict directional moves vs. flat markets)*

---

## 📈 Next Steps & Potential Improvements

* **Time Series Cross-Validation:** Implement more robust validation strategies (e.g., `TimeSeriesSplit`) to prevent data leakage and provide a more realistic assessment of model performance.
* **Hyperparameter Tuning:** Fine-tune ML models using `GridSearchCV` or `RandomizedSearchCV` for optimal performance.
* **Additional Features:** Explore other market microstructure features (e.g., bid-ask spread, order book depth, volatility, trade imbalances).
* **Advanced Models:** Experiment with Gradient Boosting Machines (XGBoost, LightGBM) or deep learning models (LSTMs, CNNs) for time series.
* **Strategy Backtesting:** Develop a basic trading strategy based on model predictions and evaluate its profitability, accounting for transaction costs.
* **Multi-Exchange Analysis:** Extend the analysis to incorporate data from multiple exchanges to identify potential arbitrage opportunities or leading price discovery mechanisms.

---

Thank you for your time and consideration! Please feel free to explore the notebooks and provide any feedback.

---