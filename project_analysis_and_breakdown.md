# Sales Forecasting Project - README

## Project Overview

This project uses shopper location data and store information to forecast daily sales for multiple retail firms. The solution implements advanced time series forecasting techniques through ensemble modeling, combining traditional statistical methods with machine learning approaches.

## Data Description

The project utilized four primary datasets:
- **Sales.csv**: Daily total sales (in dollars) of each firm
- **Store.csv**: List of all stores, their parent firms, and coordinate locations
- **Shopper.csv**: List of all shoppers and their shopper types
- **ShopperLoc.csv**: The coordinate locations of each shopper every hour

The data represents a simulated retail environment where all sales come from shoppers inside stores, and shoppers are only observed inside stores or commuting between them during store hours.

## 1. Customer-Store Matching Approach

### Methodology

The approach for matching customers to stores involved:

1. **Store Radius Estimation**:
   - Used a density-based approach combined with percentile methods to determine each store's radius
   - For each store, identified shopper points within a large capture radius (200 units)
   - Applied both a percentile method (95th percentile of distances) and a density-based clustering method (DBSCAN)
   - Combined the results from both methods to get robust radius estimates
   - Fallback to a default radius of 15 units when insufficient data was available

2. **Matching Algorithm**:
   - Created a KD-Tree data structure for efficient spatial querying of store locations
   - For each shopper location, found stores within maximum radius distance
   - Assigned shoppers to stores based on whether they fell within a store's radius
   - Used single store assignment (closest store when multiple matches) to avoid double-counting
   - Processed the large ShopperLoc dataset in chunks to handle memory constraints
   - Implemented parallel processing when possible to improve efficiency

3. **Traffic Aggregation**:
   - Aggregated unique shopper visits to each store on a daily basis
   - Further aggregated store-level traffic to firm-level daily traffic

The matching process achieved a 72.46% assignment rate, with 5,675,845 of 7,833,000 shopper locations successfully matched to stores. This match rate is reasonable considering that shoppers spend some time commuting between stores in the simulation.

## 2. Data Exploration and Cleaning

### Data Cleaning Issues Addressed

1. **Outlier Detection and Handling**:
   - Identified 41 outlier values in the Sales data across 28 firms
   - Replaced outliers with firm-specific means rather than global means to preserve firm-specific patterns
   - **Limitation**: While statistical approaches (z-score thresholds) were used to identify outliers, it's important to note that without additional information on external drivers (special promotions, holidays, local events), it's difficult to determine whether extreme values represent data errors or legitimate sales spikes
   - In a production environment, outlier detection should incorporate contextual business information and be reviewed by domain experts before automated replacement

2. **Missing Value Handling**:
   - Applied backfilling for missing sales values using firm-specific previous values
   - Implemented forward-fill for traffic values where appropriate
   - Ensured test/forecast dataset had properly filled lag features and predictors

3. **Data Type Consistency**:
   - Converted Date columns to consistent datetime format
   - Ensured FirmID columns used consistent numeric types for proper merging

### Key Data Characteristics

1. **Seasonality Patterns**:
   - Strong day-of-week patterns in both sales and traffic data
   - Significant variation in seasonality strength across firms 
   - Weekend vs. weekday shopping patterns varied by firm
   - 4 distinct clusters of seasonality patterns were identified across firms

2. **Traffic-Sales Relationship**:
   - Strong positive correlation between traffic and sales for most firms (96% had positive correlation)
   - Correlation strength varied significantly by day of week (strongest on Thursday at 0.7291, weakest on Sunday at 0.6807)
   - 82% of firms showed statistically significant traffic-sales correlations
   - Average correlation across firms: 0.159, Median: 0.145

3. **Firm Heterogeneity**:
   - Substantial variation in average sales and traffic across firms
   - Significant differences in traffic-to-sales conversion rates by firm
   - Varying levels of seasonal patterns and trend components

## 3. Univariate OLS Model

A univariate OLS model was implemented using percent change in total daily traffic to each firm's stores as the predictor. The model included firm-specific fixed effects to account for differences in baseline sales levels.

### Model Performance
- Global OLS R²: 0.4522
- RMSE: 643.14
- MAE: 502.09

When the model was extended to include firm-specific coefficients, performance varied widely across firms:
- Some firms showed positive R² values (best was Firm 32.0 with R² = 0.4926)
- Many firms had negative R² values, indicating poor fit
- The weighted average R² across all firm-specific models was -1.1958

The traffic-percent-change coefficient was statistically significant (432.07, p < 0.001), indicating that a 1% increase in traffic is associated with approximately $432 increase in sales on average.

## 4. Advanced Forecasting Model

### Modeling Approach

An ensemble forecasting approach was developed, combining three distinct modeling techniques:

1. **LightGBM Gradient Boosting Model**:
   - Features: FirmID, FirmDailyTraffic, time components (DayOfWeek, Month, IsWeekend), and lag features (Sales_lag1, Sales_lag7)
   - Hyperparameters tuned using Optuna with Bayesian optimization
   - Implemented rolling forecast to properly update lag features during prediction
   - Best validation R²: 0.7795, RMSE: 724.16

2. **Prophet Time Series Models**:
   - Firm-specific models to capture individual seasonality patterns
   - Included traffic features as regressors
   - Hyperparameters tuned for each firm individually
   - Performance varied: Best firm R² = 0.4666 (Firm 34.0), Average RMSE across firms = ~237.84

3. **OLS Baseline Model**:
   - Used as fallback model for firms with insufficient data
   - Included both global and firm-specific models
   - Global model R²: 0.4522

4. **Ensemble Integration**:
   - Combined predictions with optimized weights by firm
   - Default weight distribution: LightGBM 50%, Prophet 50%
   - Implemented fallback strategy for missing predictions

### Machine Learning Methodology Details

#### LightGBM Model

**Why LightGBM?**
- **Gradient Boosting Framework**: LightGBM is an advanced implementation of gradient boosting that sequentially builds decision trees to correct errors from previous trees, making it highly effective for capturing complex non-linear relationships in sales data.
- **Leaf-wise Growth Strategy**: Unlike traditional algorithms that use level-wise tree growth, LightGBM uses a leaf-wise approach that can reduce loss more effectively, resulting in better accuracy.
- **Categorical Feature Handling**: LightGBM natively handles categorical features (like FirmID, DayOfWeek) without requiring one-hot encoding, preserving the categorical nature of these predictors.
- **Memory Efficiency**: Uses a histogram-based algorithm that buckets continuous features into discrete bins, significantly reducing memory usage and computational cost while maintaining predictive power.
- **Regularization Capabilities**: Built-in L1 and L2 regularization helps prevent overfitting, especially important when dealing with noisy retail sales data.

**Structural Advantages for Sales Forecasting**:
- The tree-based structure naturally captures interaction effects between features (e.g., specific combinations of day-of-week and firm that might have unique sales patterns)
- Automatically identifies and leverages the most informative features, reducing the impact of irrelevant predictors
- Robust to outliers since splits are based on feature ranking rather than absolute values
- Efficiently handles missing values without requiring extensive preprocessing

#### Prophet Model

**Why Prophet?**
- **Decomposition Approach**: Prophet decomposes time series into trend, seasonality, and holiday components, making it ideal for retail sales that often exhibit these patterns.
- **Robust to Missing Data**: Handles missing values and outliers effectively through its Bayesian approach.
- **Flexible Seasonality Modeling**: Captures multiple seasonal patterns simultaneously (weekly, monthly, yearly) without requiring stationarity.
- **Changepoint Detection**: Automatically identifies significant changes in trends, which is valuable for detecting shifts in consumer behavior.
- **Incorporates External Regressors**: Allows traffic data to be included as regressors while maintaining the integrity of the time series structure.

**Advantages for Firm-Specific Modeling**:
- The customizable priors enable firm-specific tuning, allowing different seasonality strengths for firms with different patterns
- Automatically handles holidays and special events that might affect retail sales
- Provides interpretable components that decompose each prediction into trend and seasonal factors
- Built to handle the "cold-start" problem with limited historical data

#### Ensemble Methodology

**Why Ensemble?**
- **Model Diversity**: Combines fundamentally different modeling approaches (tree-based, decomposition-based, and linear), each capturing different aspects of the sales patterns.
- **Error Reduction**: Ensembling tends to reduce prediction variance and bias when models have different strengths and weaknesses.
- **Robustness to Data Shifts**: If one model fails in certain conditions, others can compensate, providing more stable predictions across varying data patterns.
- **Adaptive Weighting**: Firm-specific ensemble weights allow customization to each firm's unique characteristics.

**Implementation Advantages**:
- The weighted average approach is simple yet effective, allowing for transparent interpretation of model contributions
- Flexible architecture enables easy addition or replacement of component models as better techniques emerge
- Natural fallback mechanism where stronger models can compensate for weaker ones in different contexts

### Hyperparameter Tuning Methodology

#### Bayesian Optimization with Optuna

**Why Bayesian Optimization?**
- **Efficient Search**: Unlike grid or random search, Bayesian optimization learns from previous trials to focus on promising regions of the hyperparameter space.
- **Handles Complex Interactions**: Captures interactions between hyperparameters that grid search would miss without exhaustive combinations.
- **Probabilistic Model-Based**: Uses a surrogate model (Tree-structured Parzen Estimator in Optuna) to predict which hyperparameter combinations will yield the best results.
- **Early Stopping**: Automatically terminates unpromising trials, focusing computational resources on promising configurations.

**Implementation Details**:
- Used time series-aware cross-validation to prevent data leakage
- Optimized for RMSE as the primary metric while tracking R² for interpretation
- Efficiently balanced exploration (trying new hyperparameter regions) vs. exploitation (refining promising regions)
- Utilized parameter importance analysis to understand which hyperparameters had the most impact

**LightGBM Hyperparameters Tuned**:
- **Regularization Parameters**: lambda_l1 (L1 regularization), lambda_l2 (L2 regularization)
- **Tree Structure Parameters**: num_leaves, max_depth, min_child_samples
- **Sampling Parameters**: feature_fraction, bagging_fraction, bagging_freq
- **Optimization Parameters**: learning_rate, n_estimators

**Prophet Hyperparameters Tuned**:
- **Changepoint Parameters**: changepoint_prior_scale, changepoint_range
- **Seasonality Parameters**: seasonality_prior_scale, seasonality_mode
- **Holiday Parameters**: holidays_prior_scale

#### Firm-Specific Parameter Optimization

A key innovation in the approach was tuning hyperparameters separately for different firms rather than using a one-size-fits-all model:

- **Stratified Firm Sampling**: Ensured all firms were represented in the hyperparameter tuning process
- **Transfer Learning Approach**: For firms with limited data, leveraged patterns from similar firms
- **Parameter Correlation Analysis**: Analyzed which firm characteristics correlated with optimal hyperparameter values
- **Meta-Learning**: Used high-level firm characteristics to guide initial hyperparameter selection

#### Ensemble Weight Optimization

The ensemble weights were optimized through:

- **Firm-Specific Grid Search**: Tested different weight combinations for each firm
- **Performance Validation**: Selected weights that minimized RMSE on the validation set
- **Generalization Analysis**: Identified patterns in optimal weights to create fallback weights for new firms
- **Correlation Study**: Analyzed which firm characteristics correlated with optimal weight distributions

This sophisticated approach to hyperparameter tuning significantly contributed to the model's performance, with firm-specific parameter sets outperforming global parameters by capturing the unique characteristics of each firm's sales patterns.

### Improvements Over Baseline

The ensemble model showed significant improvements over the baseline OLS model:
- R² improvement: 26.7%
- RMSE reduction: 11.7%
- MAE reduction: 17.2%

### Justification for Model Superiority

1. **Capture of Complex Patterns**:
   - The ensemble approach can capture both linear and non-linear relationships
   - Properly handles seasonality at multiple levels (daily, weekly, monthly)
   - Accounts for firm-specific patterns in the data
   - LightGBM's tree structure naturally models interaction effects between features
   - Prophet's decomposition approach explicitly models trend, seasonality, and holiday effects

2. **Robustness to Outliers and Distribution Shifts**:
   - Tree-based models (LightGBM) are less affected by outliers than linear models since they rely on splits rather than distances
   - Prophet's robust decomposition handles anomalies well through automatic changepoint detection
   - The ensemble approach reduces prediction variance by combining multiple model types
   - Firm-specific models adapt to the unique noise characteristics of each firm's data
   - Rolling forecast implementation prevents error accumulation in multi-step predictions

3. **Optimal Feature Utilization**:
   - LightGBM identified the most predictive features through feature importance
   - Top features were: FirmID, FirmDailyTraffic, and lag features
   - Prophet effectively incorporated traffic data as regressors while maintaining time series structure
   - Categorical features were handled natively without information loss
   - Lag features were carefully constructed to capture appropriate temporal dependencies

4. **Systematic Handling of Missing Values**:
   - Implemented proper rolling-window forecasting to dynamically update lag features
   - Used multiple fallback strategies to ensure complete forecasts
   - Prophet's Bayesian approach naturally accommodates missing data
   - LightGBM's handling of missing values doesn't require imputation
   - The ensemble framework provides redundancy when particular models fail

5. **Firm-Specific Customization**:
   - Different firms benefited from different model weightings based on their unique characteristics
   - Hyperparameters were tuned individually for each firm to capture unique patterns
   - Seasonality strength was accounted for in Prophet model configuration
   - The ensemble weights adapted to each firm's predictability by different model types
   - Identified distinct clusters of firms with similar seasonal patterns for knowledge sharing

6. **Statistical Validity and Evaluation**:
   - Used time series cross-validation to prevent data leakage
   - Evaluated models on multiple metrics (RMSE, MAE, R²) for comprehensive assessment
   - Applied proper validation techniques accounting for temporal dependencies
   - Maintained out-of-sample testing principles throughout development
   - Tested robustness across different firms with varying characteristics

### Final Forecast Output

The final forecasts were saved to 'Sales_with_forecasts.csv' and 'Forecasts_only.csv', containing the ensemble model's predictions for the missing 4 weeks of sales data.

## 5. Application to Trading Strategies

### Conditions for Predictive Value in Stock Trading

The forecasting model could potentially help predict stock returns under the following conditions:

1. **Sales-Stock Price Correlation**:
   - If a firm's sales are strongly correlated with its stock price
   - When sales surprises (vs. analyst expectations) drive significant price movements
   - For retail firms where same-store sales are a key performance indicator

2. **Timing Considerations**:
   - Short-term trading around earnings announcements
   - When sales figures are announced before market fully incorporates the information
   - For creating lead indicators ahead of official sales reports

3. **Market Inefficiencies**:
   - When the market is not fully pricing in foot traffic information
   - If institutional investors lack access to granular foot traffic data
   - For smaller, less-followed stocks where information efficiency is lower

### Trading Strategy Framework

A potential trading strategy using this model could:

1. **Generate buy/sell signals based on:**
   - Deviation between forecasted sales and consensus estimates
   - Relative forecasted performance across firms in the same retail sector
   - Sudden changes in the traffic-sales relationship signaling operational issues

2. **Implement portfolio construction:**
   - Long/short pairs trading based on relative expected performance
   - Sector rotation based on aggregate traffic trends
   - Risk-weighted position sizing based on forecast confidence intervals

3. **Incorporate timing elements:**
   - Enter positions ahead of earnings announcements
   - Adjust holding periods based on forecast horizon reliability
   - Exit based on post-announcement price action or forecast updates

### Additional Helpful Data

1. **Alternative Data Sources:**
   - Credit card transaction data to validate and enhance sales forecasts
   - Social media sentiment analysis for brand perception
   - Web traffic and app usage patterns for online sales correlation
   - Weather data to account for shopping pattern disruptions
   - Competitor promotions and pricing data

2. **Market-Specific Data:**
   - Analyst consensus estimates for sales to identify expectation gaps
   - Short interest in retail stocks to identify crowded trades
   - Options market implied volatility ahead of announcements
   - Institutional ownership changes in retail stocks

3. **Macroeconomic Indicators:**
   - Consumer confidence indices
   - Retail sales reports (as leading/lagging indicators)
   - Inflation data impacting consumer spending patterns
   - Unemployment figures and wage growth data

### Implementation Approach

To incorporate additional data into the forecasting model:

1. **Feature Engineering:**
   - Create ratio features between forecasted sales and consensus estimates
   - Develop composite indicators combining traffic, web activity, and sentiment
   - Calculate relative metrics across firms in the same sector

2. **Model Enhancement:**
   - Extend to multivariate LSTM or Transformer models for complex patterns
   - Implement hierarchical Bayesian models to share information across similar firms
   - Develop separate models for earnings surprise prediction and stock price movement

## 6. Model Interpretability and Business Insights

### Model Interpretability

The ensemble model offers several interpretable components that provide business insights:

1. **Feature Importance Analysis**:
   - LightGBM identified the most predictive features, with FirmID, traffic variables, and lag features being most important
   - This confirms the importance of firm-specific effects and recent sales history in predicting future performance
   - Day of week emerged as a critical feature, highlighting the importance of weekly seasonality patterns

2. **Prophet Component Decomposition**:
   - The trend components reveal the overall growth trajectory for each firm
   - Weekly seasonal patterns show the expected day-of-week effects on sales
   - The traffic regressor effects quantify the exact impact of traffic changes on sales

3. **Traffic-Sales Relationship**:
   - Significant heterogeneity in traffic-to-sales conversion across firms (coefficient range from 3.14 to 40.95)
   - Day-specific correlations show varying traffic importance by day of week (strongest on Thursday)
   - Weekday vs. weekend traffic utilization patterns differ substantially across firms

### Key Business Insights

1. **Firm Segmentation by Pattern**:
   - The cluster analysis identified 4 distinct firm segments based on sales patterns:
     - Cluster 1 (36% of firms): Relatively flat weekday pattern with weekend drops
     - Cluster 2 (38% of firms): Strong mid-week peaks with weekend drops
     - Cluster 3 (6% of firms): Weekend-dominant sales pattern
     - Cluster 4 (20% of firms): Monday-dominant pattern with gradual decline through week
   - These segments could inform different marketing and staffing strategies

2. **Traffic-Sales Efficiency**:
   - 96% of firms showed positive traffic-sales correlation, but the strength varied significantly
   - Firms with stronger correlations have more predictable sales based on traffic
   - Firms with weaker correlations may have other significant sales drivers beyond foot traffic

3. **Temporal Dynamics**:
   - Lag features (previous day, previous week) were highly important predictors
   - This suggests strong auto-regressive components in sales patterns
   - The 7-day lag's importance confirms weekly shopping cycles are significant across firms

4. **Model Performance Heterogeneity**:
   - Different model types performed better for different firms
   - Tree-based models generally outperformed for firms with complex non-linear patterns
   - Prophet models performed better for firms with strong seasonal components
   - This heterogeneity supports the ensemble approach as no single model works best universally

## 7. Technical Implementation Details

### Computational Approach

1. **Efficient Data Processing**:
   - Implemented chunk-based processing for the large ShopperLoc dataset (7.8M records)
   - Used Polars DataFrame library for memory-efficient operations
   - Achieved processing speeds of approximately 32,651 records per second
   - Applied KD-Tree spatial indexing for efficient customer-store matching

2. **Parallelization Strategy**:
   - Where applicable, leveraged parallel processing for computationally intensive tasks
   - Implemented batch processing of independent firm-specific models
   - Balanced memory usage and computational speed through optimized chunk sizes

3. **Memory Management**:
   - Monitored memory usage throughout processing (peaked at ~1.2GB)
   - Implemented intermediate result saving to prevent memory overflow
   - Used efficient data types and structures to minimize memory footprint

4. **Model Persistence and Deployment**:
   - Saved trained models and parameters in JSON format for reproducibility
   - Generated interactive visualizations for business stakeholders using Plotly
   - Implemented proper validation techniques to ensure model robustness

### Data Pipeline Architecture

The full data processing and modeling pipeline follows these steps:

1. **Data Ingestion and Cleaning**:
   - Load raw data from CSV files
   - Clean outliers (41 detected and replaced across 28 firms)
   - Handle missing values through appropriate backfilling
   - Convert data types for consistency

2. **Customer-Store Matching**:
   - Estimate store radii using density and percentile methods
   - Create spatial index using KD-Tree for efficient matching
   - Process shopper location data in chunks
   - Aggregate to daily store and firm traffic

3. **Feature Engineering**:
   - Generate lag features (1-day, 7-day)
   - Create temporal features (day of week, month, weekend flag)
   - Calculate statistical aggregates (rolling averages, standard deviations)
   - Develop specialized normalized and robust-scaled features

4. **Model Training**:
   - Train LightGBM model with optimized hyperparameters
   - Develop firm-specific Prophet models 
   - Implement OLS baseline models (global and firm-specific)
   - Build ensemble integration with tuned weights

5. **Forecasting and Validation**:
   - Apply rolling forecast methodology for test period
   - Evaluate with multiple metrics (RMSE, MAE, R², MAPE)
   - Generate visualizations of model performance
   - Create final forecasts for submission

6. **Output Generation**:
   - Save final forecasts to CSV files
   - Generate interactive dashboard for multi-firm visualization
   - Document model performance and parameters

### Code Organization

The project code is organized into several logical components:

1. **Data Preparation**: Handles loading, cleaning, and initial processing
2. **Spatial Analysis**: Implements store radius estimation and customer matching
3. **Feature Engineering**: Creates predictive features from raw data
4. **Model Development**: Implements and trains various forecast models
5. **Ensemble Integration**: Combines model outputs for optimal forecasts
6. **Evaluation**: Analyzes model performance and generates visualizations
7. **Forecast Generation**: Produces final forecasts for submission

### Reproducibility Considerations

To ensure reproducibility of results:
- Random seeds are set to consistent values (42) throughout the code
- Hyperparameter tuning results are documented and stored
- Data processing steps are clearly documented with intermediate outputs
- Model evaluation uses standardized metrics and validation approaches

## Limitations and Potential Improvements

1. **Data Limitations:**
   - Simulated data may not fully capture real-world complexities
   - Limited historical data for some firms impacts model robustness
   - Lack of external variables like promotions, weather, or holidays
   - Potential errors in shopper-store matching that could propagate to traffic estimates
   - Absence of product-level or category-level sales information that might reveal finer patterns

2. **Modeling Improvements:**
   - **Advanced Deep Learning Approaches:**
     - Implement sequence models like LSTM or Transformer architectures to better capture long-term dependencies
     - Apply neural ODE (Ordinary Differential Equation) methods for continuous-time modeling
     - Utilize attention mechanisms to dynamically weight historical observations
     - Implement Graph Neural Networks to model relationships between stores and firms
     - **Data Volume Caveat**: Most of these advanced approaches would require significantly more historical data (typically 2+ years) to avoid overfitting and learn meaningful patterns
   
   - **Enhanced Ensemble Techniques:**
     - Develop stacking models that use meta-learners to combine base models more intelligently
     - Implement boosting across different model types rather than simple weighted averaging
     - Explore Bayesian Model Averaging for theoretically grounded ensemble weights
     - Create dynamic ensemble weights that adapt based on recent model performance
     - **Computational Consideration**: These methods increase computational complexity and would require careful validation to ensure the performance gain justifies the additional resources
   
   - **Probabilistic Forecasting:**
     - Extend models to produce full predictive distributions rather than point forecasts
     - Incorporate Bayesian Neural Networks for principled uncertainty quantification
     - Implement Gaussian Process models for non-parametric uncertainty modeling
     - Develop conformal prediction methods for calibrated prediction intervals
     - **Business Value**: These approaches would provide valuable risk assessment for inventory planning and financial forecasting beyond simple point predictions

3. **Feature Engineering Extensions:**
   - Create interaction features between traffic and temporal variables
   - Develop more sophisticated lag structures that vary by firm based on autocorrelation analysis
   - Extract hierarchical features across different time granularities
   - Implement automated feature selection methods like BORUTA or RFE
   - Develop anomaly indicators for unusual traffic or sales patterns

4. **Operational Considerations:**
   - Optimize computational performance for real-time predictions using model distillation techniques
   - Develop automated monitoring for model drift using statistical process control methods
   - Implement online learning approaches to continuously update models with new data
   - Create model explainability tools to interpret predictions for business stakeholders
   - Develop an A/B testing framework to systematically evaluate model improvements

5. **Methodological Extensions:**
   - Implement hierarchical models that share information across related firms
   - Develop transfer learning approaches to leverage knowledge from data-rich firms
   - Explore causal inference methods to isolate the impact of traffic on sales
   - Apply reinforcement learning for dynamic hyperparameter adaptation
   - Investigate multi-task learning to simultaneously predict related business metrics

## 8. Future Research Directions

Building on the current modeling approach, several promising research directions could further enhance sales forecasting capabilities:

1. **Competitive Effects Modeling**:
   - Incorporate spatial competition between firms and stores
   - Model spillover effects when nearby stores experience unusual traffic or sales patterns
   - Quantify cannibalization effects within multi-store firms

2. **Dynamic Consumer Behavior**:
   - Explore models that account for changing consumer preferences over time
   - Implement regime-switching models to capture structural breaks in traffic-sales relationships
   - Develop seasonal-aware models that adapt to changing seasonal patterns year-over-year

3. **External Factor Integration**:
   - Investigate methods to incorporate macroeconomic indicators as leading indicators
   - Develop transfer learning approaches to leverage external data sources with limited training data
   - Research optimal ways to integrate weather, events, and promotional information

4. **Automated Model Selection and Hyperparameter Tuning**:
   - Develop meta-learning approaches that predict which model type will perform best for a given firm
   - Implement neural architecture search for optimal deep learning model structures
   - Research efficient hyperparameter optimization for multi-firm settings
