## Ensemble-ANFIS model
Machine Learning price prediction of Indexes. Comparing ANFIS vs ARIMA and Hybrid SOM/ANFIS vs Ensemble ANFIS.

The study signifies the important role both hybrid and ensemble ANFIS to potentially
developing an automated prediction system for forecasting stocks prices, patterns and
volatility, with large potential application in the economic and financial sector, such as risk
management, asset pricing and allocation [Worrall, J (2018)](results/Dissertation_MSC8002_Q1222209_JohnWorrall.pdf).


The models in the study investigate adjusted closing prices of All Ordinaries, Financial Times Stock Index,
Dow Jones Industrial Average, Hang Seng Index and Nikkei Index from 1 January 2008 to 1 January 2018 represented in below times series.
<div class="box" align="center">
        <img src="img/stockIndexPrices.png" height=350 />
        <p style="text-align:center">
        <small>Figure 1. Actual daily returns of five stock indices</small>
        </p>
  </div>


A preliminary investigation was performed to display the relationship between the time series data shown in correlation matrix plot
<p align="center">
<img src="img\correlation-matrix.png" width=350 ></p><p style="text-align:center">
<small>Figure 2. Indexes correlation matrix</small>
</p>

<p></p>
<p style="text-align:left">
The challenge in ANFIS model is to control the parameters in order to avoid
overfitting. In this study ANFIS is based on first order Sugeno and Mamdani inference
system with a number of indexes. Below is a sample of training model parameters selection.
</p>

<div class="box" align="center">
        <img src="img\model-config1.png" height=300 />
        <img src="img\model-config2.png" height=300 />
        <p style="text-align:center">
        <small>Figure 3. SOM structure</small></p>
        <img src="img\model-fis-structure.png" height=350 />
        <p style="text-align:center">
        <small>Figure 4. FIS structure</small>
        </p>
  </div>

<div class="box" align="center">
<img src="img\model-training.png" height=400 />
<p style="text-align:center">
<small>Figure 5. SOM-ANFIS training results </small>
</div>

In the below Figure (5,10,50,100 ensemble model) we illustrate a boxplot of the
ensemble-ANFIS models forecasting error for All Ordinaries value. The outliers specified by
the end of whisker in each boxplot represent the extreme magnitudes of the forecasting error
within the testing phase along with other statistics; upper quartile, median and lower quartile.

<div class="box" align="center">
<img src="img\model-ensemble-prediction.png" height=500 />
<p style="text-align:center">
<small>Figure 6. n-ensemble ANFIS statistical forecast results (ensemble in green, actual value in red)</small>
</div>

Table below represents the predicted values of performance for developed models. The top two
rank ensemble ANFIS model (n= 50 and n =100) scored high r values of greater than 0.94 and
RMSE’s of below 54. The top performing SOM-ANFIS scored a 0.97 r value, a slightly
stronger relationship then the ensemble models. The SOM-ANFIS predictive error value,
RMSE, produced 41.967, the lowest of the developed models. Figure 5.3.1 boxplot of the
three models errors supported the error metrics findings, with SOM-ANFIS clearly the best
performing model, while the ensemble models showed a similar distribution. 

<div class="box" align="center">
<img src="img\result-tbl.png" height=300 />
<p style="text-align:center">
<small>Table 1. Testing performance of SOM-ANFIS vs n-ensemble ANFIS (n =50 and n=100)</small>
</div>

### Summary
Overall, the study highlights the appropriateness of the ANFIS and hybrid/ensemble based
approaches to modelling for daily forecasting of All Ordinaries. These approaches present
advantages to similar indexes incorporating ANFIS machine learning techniques for
improved performance with regard to different stocks in the financial market. Furthermore,
this research paper provides a baseline relevant to SOM-ANFIS and ensemble ANFIS index
forecasting, which have not yet studied. The study illustrates models’ ability to forecast All
Ordinaries stocks, providing insightful information on the phenomena of market events for
investors, policy makers and other stakeholders to make more informed and profitable
decisions.



#### Dependecies
 - Matlab R2017b - (Machine Learning Models) ANFIS / SOM 
 - R-3.6.1 -  ARIMA and feature selection.
