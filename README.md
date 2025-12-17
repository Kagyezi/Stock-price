# Stock-price
This is a repository for a stock price prediction project.

## Problem recap
There are so many companies and individuals participating in trading the financial markets, for example the Uganda Stock exchange and not all of the have the luck of being successful and end up losing money.
Our project aims at building stock price prediction algorithm which uses previous stock data to predict the future price of the stock. This is algorithm will help these companies and individuals to become more profitable and maximize their profits.
The stock that we chose to use for our project is Activision Blizzard, which is an American video game holding company, from the years 2010 to 2022. We got this dataset from the link zenodo.

## Exploratory Data Analysis (EDA)
Using pandas, we imported the dataset and tried to understand the dataset.
We used descriptive statistics (the .describe() method) to summarize key metrics.
Using the .info() method, we discovered that the dataset has 3191 rows, 7 columns, no missing values, and the datatype of this column. From this, we identified that the date column needed to be converted from object format to datetime format.
We also checked for duplicated values, but the data was already clean.

## Heatmap Visualization
Since all our desired columns are numerical, we used a correlation heatmap to identify relationships between them. Our finding was that the prices are highly correlated with each other, but not highly correlated with the volume.

## Price Vs Date Plot.
A line plot of Close Price vs. Date reveals stock performance trends over time. This helps in observing patterns, volatility, and general stock growth.

## Volume Vs Date Plot.
A line plot of Volume vs. Date reveals how much the trading volume changes over time. This helps in observing patterns, volatility, and general stock growth.

## Model Building
After we were done with exploring the data and obtaining insights, we built the models to use in our predictions
We used two models so as to compare their results.

## Model 1 - Random Forest Regressor.
The first model we used was a Random Forest Regressor whose features, x, were; Open, High, Low, and Volume. Our target variable, y, was the close price.
We then split the data into training and testing data, using 30% of the data for testing, scaled the data using a Standard Scaler and then used it to train and test the model.
On evaluating the model, the results were as follows;
root_mean_absolute_error - 0.4272
mean_absolute_error - 0.2684
r2_score - 0.9998

## Model 2 - Recurrent Neural Network.
We used a recurrent neural network, particularly a Long Short Time Memory Model (LSTM) since our problem is using time series data. We used the the date as our only feature to predict the close price as we wanted to know how it changes over time.
We split the data into training and testing data using data from January 2010 to December 2020 for training and January 2021 to September 2022 for testing. We then scaled the data using a Min Max Scaler.
We then build our model with the input layer, two hidden layers and one output layer. The model was then used to train and test our data
On evaluating the model, the results were as follows;
root_mean_absolute_error - 0.0359
mean_absolute_error - 0.0239 
r2_score - 0.9142   .

## Conclusion.
For the first model, because the errors (RMSE & MAE) are NOT extremely small, the model might be overfitting. It memorized the training data but performed poorly on unseen test stock data is sequential but Random Forest ignores time order .
The second model demonstrated the best predictive performance, achieving errors nearly 14 times lower (0.0310 vs. 0.4177 RMSE).
The low RMSE and MAE, combined with the high R2 score confirm the second model's robust ability to model complex, sequential dependencies in time-series financial data.

## Deployment.
We saved our model as a .h5 file and the scaler as a pickle file and then used streamlit as our deployment library to build a simple interface for deploying and interacting with our model.
The deployed file takes in either a csv file of stock data or the previous 5 closing prices in order to give a prediction.
