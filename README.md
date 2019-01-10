## Data Science Portfolio 

This portfolio consists of several notebooks and projects illustrating the work I have done during my studies (MSc Data Science & Entrepreneurship). It should be noted that some notebooks were not part of my master which I created merely to educate myself. 


###  [NLP: Analyzing WhatsApp Messages](https://github.com/MaartenGr/soan/blob/master/soan.ipynb)
* Analyses done on whatsapp messages between me and my fianciee to surprise her with on our wedding
* Methods::
  * Sentiment Analysis
  * Unique words (TF-IDF)
  * Topic Modeling
  * Wordclouds, etc.
  
<img src="https://github.com/MaartenGr/soan/blob/master/reddit.png" width="70%">

[Repository](https://github.com/MaartenGr/soan) | [Github](https://github.com/MaartenGr/soan/blob/master/soan.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/soan/blob/master/soan.ipynb)

---

###  [Optimizing Emté Routes](https://github.com/MaartenGr/Projects/blob/master/Notebooks/RouteOptimization.ipynb)
* Project for the course Business Analytics in the master
* Optimization of managers visiting a set of cities
* Total of 133 cities, max distance 400km with time and capacity constraints
* Thus, a vehicle routing problem
* Methods:
  * Integer Linear Programming
  * Tabu Search
  * Simmulated Annealing
  * Ant Colony Optimization

[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/RouteOptimization.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/RouteOptimization.ipynb)

![alt text](https://media.giphy.com/media/FDHO8sbi4hl8qsABDv/giphy.gif)

---

## Predicting and Optimizing Auction Prices

An auctioneer would like to know if it is possible to predict how well a certain item in an auction will do. If yes, can you then change the Starting Price of an item in such a way that it will result in the highest possible outcome of that item. Thus, I started with predicting whether an item will be sold for more than its estimated value or not (a classification problem) and validated using Leave-One-Out CV (leave one auction out). Then, I optimized one auction that was not previously seen using a Genetic Algorithm and a basic heuristic I constructed myself. I managed to optimize the auction such that more profit could be gained from items in an auction. 

[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/AuctionAnalysis.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/AuctionAnalysis.ipynb)

<img src="https://github.com/MaartenGr/Projects/blob/master/Images/auction_result.png">

---

## Statistical Analysis using the Hurdle Model

For the course Strategy and Business Models I worked on analyzing Apple Store data to get insight into what makes an app succesfull. The main research question of the paper is which business model aspects influence the performance of mobile games in the first five months after they are published between early and later entrants of the market. Because of the highly competitive mobile game market, this study aims to provide guidelines to developers and address the research gap in the literature on mobile app market, namely the impact of entry timing and the use of technological innovation on the performance for mobile games. 

[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/AppStoreAnalysis.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/AppStoreAnalysis.ipynb)

<img src="https://github.com/MaartenGr/Projects/blob/master/Images/appstore.png" width="700" height="260"/>

---

## Predict and optimize demand

The goal of this assignment was to create a simulation model that can be used to check the performance of different methods for calculating the expected demand. A retailer has indicated to wanting a tool that can be used to predict how much of a product is needed to fulfill the demand today and the next day. Actual demand was tracked and provided for 2014-2017.

A safety factor Z affecting the forecast error received from predictions (using XGBoost, LightGBM and CatBoost) was optimized using skopt's bayesian optimization techniques. 

[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/simulation.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/simulation.ipynb)

<img src="https://github.com/MaartenGr/Projects/blob/master/Images/simulation.png"/>

---

## Neural Style Transfer

For the course deep learning I worked on a paper researching the optimal selection of hidden layers to create the most appealing images while speeding up the process of optimization. I used some of the code [here](https://harishnarayanan.org/writing/artistic-style-transfer/) and mostly worked on researching which style layers performed best in the neural style transfer task. Notebook will follow. 

<img src="https://github.com/MaartenGr/Projects/blob/master/Images/neural_style_transfer.png" width="594" height="290"/>

---

## Predicting Housing Prices 

A project for the course Introduction to Machine Learning. All students were asked to take part into the following kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/). In order to get the highest grade for the course you simply had to end up at the highest position of your class. I scored 1st within the class and 33th on the leaderboard. I used a weighted average of the results of XGboost, Lasso, ElasticNet, Ridge and Gradient Boosting Regressor in order to get the best results. All feature engineering can be found in the notebook.

[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/HousingPrices.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/HousingPrices.ipynb)

![alt text](https://github.com/MaartenGr/Projects/blob/master/Images/kaggle.png)

---

## FitBit

During my master Data Science and Entrepreneurship I was looking at various ways of leveraging data for the purpose of entrepreneurship. I realized that my FitBit wasn't all that accurate and at times displayed weird values. For that reason I decided to create a model that could predict what my heart rate would have been if it were missing (e.g. due to a bad signal). 

This was the very first project I did during the pre-master. Looking back, there are some things I could have done differently. For example, I did a simple 10-fold CV to test my models which might lead to overfitting as the folds are chosen randomly.

[Repository](https://github.com/MaartenGr/fitbit/) | [Github](https://github.com/MaartenGr/fitbit/blob/master/3.%20The%20Final%20Product.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/fitbit/blob/master/3.%20The%20Final%20Product.ipynb)

<img src="https://github.com/MaartenGr/fitbit/blob/master/fitbit.png"/>

---


## To Be Added
* Description of project for Netscalers
