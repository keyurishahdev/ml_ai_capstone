### Project Title

Predicting stock prices at the close of market 

#### Executive summary

In the last ten minutes of the Nasdaq exchange trading session, market makers like Optiver merge traditional order book data with auction book data. This ability to consolidate information from both sources is critical for providing the best prices to all market participants. In this project we will develop a model that will accurately predict closing price movements of different stocks. 

#### Rationale
Each trading day on the Nasdaq Stock Exchange concludes with the Nasdaq Closing Cross auction. This process establishes the official closing prices for securities listed on the exchange. These closing prices serve as key indicators for investors, analysts and other market participants in evaluating the performance of individual securities and the market as a whole.

#### Data Sources
Data Sourcing: We will be using data source provided by Optiver on kaggle 

#### Methodology

Data Exploration: Explore and understand the dataset. Analyze distributions, correlations, and patterns present in the data to gain insights into the relationships between the features and the target variable.

Data Preprocessing: Clean and preprocess the data to ensure it is suitable for model training. This may involve handling missing values, scaling features, and encoding categorical variables.

Feature Engineering: Based on the data exploration, engineer additional features that could potentially improve the predictive power of the models. This may involve creating lagged features, rolling averages, or other transformations to capture relevant information for predicting future price movements.

Model Selection: Consider different models suitable for time series data, such as ARIMA, SARIMA, Prophet, Decision Tree, Random Forest, and K-Nearest Neighbors (KNN). Evaluate the performance of these models using appropriate evaluation metrics and choose the one that provides the best results for the specific problem.

Model Training and Evaluation: Split the dataset into training and testing sets, ensuring the temporal order of the data is maintained. Train the chosen model on the training set, tune hyperparameters if necessary, and evaluate its performance on the testing set using evaluation metrics such as Mean Squared Error (MSE) or Mean Absolute Error (MAE).

Model Optimization: Fine-tune the selected model by performing hyperparameter optimization. Explore different values for hyperparameters and evaluate their impact on model performance. Update the model with the optimal hyperparameters to improve its accuracy.

Deployment: Once the model is selected, trained, and optimized, deploy it to make predictions on new, unseen data. Follow best practices for deploying machine learning models, such as encapsulating the model in a function or API that can be easily accessed.

#### Results

After evaluating different models, the following Mean Squared Errors (MSE) were obtained:

Decision Tree: 98.91231211469076
Random Forest: 48.158900356412445
KNN  (before optimization): 75.87924790651887
KNN (after optimization): 72.25048841784779
The Random Forest Model model, achieved the lowest MSE, indicating its superior performance compared to the other models.

#### Next steps
More models can be utilized. May be add time series to evaluate the mse. 