# Data science portfolio by Maarten Grootendorst

This portfolio consists of several notebooks and projects illustrating the work I have done during my studies (MSc Data Science & Entrepreneurship). It should be noted that some notebooks were not part of my master which I created merely to educate myself. 

## Optimize Emté Routes

A project for the course Business Analytics in which we were asked to optimize a certain number of managers visiting a set of cities in the most optimal way. There were 133 cities that needed to be visited. The maximum distance for a manager to be traveled was 400km and we had to minimize the total distance traveled (thereby implicitly minimizing the number of managers). Time and capacity constraints were added in later exercises within this project. 

[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/RouteOptimization.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/RouteOptimization.ipynb)

![alt text](https://media.giphy.com/media/FDHO8sbi4hl8qsABDv/giphy.gif)

## Predicting Housing Prices

A project for the course Introduction to Machine Learning. All students were asked to take part into the following kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/). In order to get the highest grade for the course you simply had to end up at the highest position of your class. I scored 1st within the class and 33th on the leaderbord. I averaged the results of XGboost, Lasso, ElasticNet, Ridge and Gradient Boosting Regressor in order to get the best results. All feature engineering can be found in the notebook.

[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/HousingPrices.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/HousingPrices.ipynb)

![alt text](https://github.com/MaartenGr/Projects/blob/master/Images/kaggle.png)

## Neural Style Transfer

For the course deep learning I worked on a paper researching the optimal selection of hidden layers to create the most appealing images while speeding up the process of optimization. I used some of the code [here](https://harishnarayanan.org/writing/artistic-style-transfer/) and mostly worked on researching which style layers performed best in the neural style transfer task. Notebook will follow. 

<img src="https://github.com/MaartenGr/Projects/blob/master/Images/neural_style_transfer.png" width="594" height="290"/>

## Predicting and Optimizing Auction Prices

For the course Business Analytics I worked on the following problem. An auctioneer would like to know if it is possible to predict how well a certain item in an auction will do. If yes, can you then change the Starting Price of an item in such a way that it will result in the highest possible outcome of that item. Thus, I started with predicting whether an item will be sold for more than its estimated value or not (a classification problem) and validated using Leave-One-Out CV (leave one auction out). Then, I optimized one auction that was not previously seen using a Genetic Algorithm and a basic heuristic I constructed myself. I managed to optimize the auction such that more profit could be gained from items in an auction. 

[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/AuctionAnalysis.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/AuctionAnalysis.ipynb)

<img src="https://github.com/MaartenGr/Projects/blob/master/Images/auction_result.png">
