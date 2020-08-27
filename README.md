## Data Science Portfolio 

This portfolio consists of several projects illustrating the work I have done in order to further develop my data science skills. 


## Table of Contents  
<!--ts-->

| Project | Tags |
| --- | --- |
| [**VLAC**: Vectors of Locally Aggregated Concepts](#vlac) | <img src="https://img.shields.io/badge/-Published-black"> <img src="https://img.shields.io/badge/-Word%20Embeddings-red"> <img src="https://img.shields.io/badge/-kMeans-81D4FA"> <img src="https://img.shields.io/badge/-Python-blue">|
| [**ReinLife**: Artificial Life with Reinforcement Learning](#reinlife) | <img src="https://img.shields.io/badge/-Reinforcement%20Learning-green"> <img src="https://img.shields.io/badge/-Deep%20Learning-yellow"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [**SoAn**: Analyzing WhatsApp Messages](#whatsapp) | <img src="https://img.shields.io/badge/-NLP-red"> <img src="https://img.shields.io/badge/-Text%20Mining-red"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [**Reviewer**: Character Popularity](#reviewer) | <img src="https://img.shields.io/badge/-BERT-red"> <img src="https://img.shields.io/badge/-NER-red"> <img src="https://img.shields.io/badge/-Sentiment-red"> <img src="https://img.shields.io/badge/-Scraper-red"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Board Game Exploration](#boardgame) | <img src="https://img.shields.io/badge/-Visualization-purple"> <img src="https://img.shields.io/badge/-Streamlit-purple"> <img src="https://img.shields.io/badge/-Heroku-90A4AE"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Statistically Generated Disney Tournament](#disney) | <img src="https://img.shields.io/badge/-Custom%20Statistic-grey"> <img src="https://img.shields.io/badge/-Scraping-red"> <img src="https://img.shields.io/badge/-Python-blue"> <img src="https://img.shields.io/badge/-R-blue"> |
| [Optimizing Emté Routes](#emte) | <img src="https://img.shields.io/badge/-ILP-90A4AE"> <img src="https://img.shields.io/badge/-Simmulated%20Annealing-90A4AE"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Pothole Detection](#pothole) | <img src="https://img.shields.io/badge/-Deep%20Learning-yellow"> <img src="https://img.shields.io/badge/-Keras-yellow"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Exploring Explainable ML](#explainable) | <img src="https://img.shields.io/badge/-SHAP-81D4FA"> <img src="https://img.shields.io/badge/-LIME-81D4FA"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Deploying a Machine Learning Model](#deploy) | <img src="https://img.shields.io/badge/-Docker-90A4AE"> <img src="https://img.shields.io/badge/-FastAPI-90A4AE"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Retro Games Reinforcement Learning](#reinforcementlearning) | <img src="https://img.shields.io/badge/-Reinforcement%20Learning-green"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Statistical Cross-Validation Techniques](#crossvalidation) | <img src="https://img.shields.io/badge/-Wilcoxon-grey"> <img src="https://img.shields.io/badge/-McNemar-grey"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Cluster Analysis: Creating Customer Segments](#clustering) | <img src="https://img.shields.io/badge/-DBSCAN-81D4FA"> <img src="https://img.shields.io/badge/-kMeans-81D4FA"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Exploring Advanced Feature Engineering Techniques](#featureengineering) | <img src="https://img.shields.io/badge/-SMOTE-90A4AE"> <img src="https://img.shields.io/badge/-DFS-90A4AE"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Predicting and Optimizing Auction Prices](#auction) | <img src="https://img.shields.io/badge/-LightGBM-81D4FA"> <img src="https://img.shields.io/badge/-Genetic%20Algorithms-90A4AE"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Statistical Analysis using the Hurdle Model](#hurdle) | <img src="https://img.shields.io/badge/-Hurdle-grey"> <img src="https://img.shields.io/badge/-ZINB%20Regression-grey"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Predict and optimize demand](#demand) | <img src="https://img.shields.io/badge/-XGBoost-81D4FA"> <img src="https://img.shields.io/badge/-Bayesian%20Optimization-90A4AE"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Analyzing Google Takeout Data](#takeout) | <img src="https://img.shields.io/badge/-Visualization-purple"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Cars Dashboard](#cars) | <img src="https://img.shields.io/badge/-Visualization-purple"> <img src="https://img.shields.io/badge/-Dash-purple">  <img src="https://img.shields.io/badge/-Python-blue"> |
| [Qwixx Visualization](#qwixx) | <img src="https://img.shields.io/badge/-Visualization-purple"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Academic Journey Visualization](#grades) | <img src="https://img.shields.io/badge/-Visualization-purple"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Predicting Housing Prices](#housing) | <img src="https://img.shields.io/badge/-XGBoost-81D4FA"> <img src="https://img.shields.io/badge/-Stacking-81D4FA"> <img src="https://img.shields.io/badge/-Python-blue"> |
| [Analyzing FitBit Data](#fitbit) | <img src="https://img.shields.io/badge/-XGBoost-81D4FA"> <img src="https://img.shields.io/badge/-Python-blue"> |

<!--te-->

## Projects

<a name="vlac"/></a>
###  [VLAC: Vectors of Locally Aggregated Concepts (VLAC)](https://github.com/MaartenGr/VLAC)
[Repository](https://github.com/MaartenGr/VLAC) | [Published Paper](https://www.ecmlpkdd2019.org/downloads/paper/489.pdf) 

* It leverages clusters of word embeddings (i.e., concepts) to create features from a collection of documents
allowing for classification of documents
* Inspiration was drawn from VLAD, which is a feature generation method for image classification
* The article was published in ECML-PKDD 2019
  
<img src="https://github.com/MaartenGr/VLAC/blob/master/Images/vlac.png" height="200"/>

---
<a name="reinlife"/></a>
###  [ReinLife: Artificial Life with Reinforcement Learning](https://github.com/MaartenGr/ReinLife)
[Repository](https://github.com/MaartenGr/ReinLife)

* Using Reinforcement Learning, entities learn to survive, reproduce, and make sure to maximize the fitness of their kin. 
* Implemented algorithms: DQN, PER-DQN, D3QN, PER-D3QN, and PPO

<p float="left">
  <img src="https://github.com/MaartenGr/ReinLife/blob/master/images/animation_medium.gif?raw=true" height="250"/>
  <img src="https://github.com/MaartenGr/ReinLife/blob/master/images/instruction.png?raw=true" height="250"/>
</p>


---
<a name="whatsapp"/></a>
###  [SoAn: Analyzing WhatsApp Messages](https://github.com/MaartenGr/soan)
[Repository](https://github.com/MaartenGr/soan) | [Notebook](https://github.com/MaartenGr/soan/blob/master/soan.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/soan/blob/master/soan.ipynb)

* Created a package that allows in-depth analyses on whatsapp conversations
* Analyses were initially done on whatsapp messages between me and my fianciee to surprise her with on our wedding
* Visualizations were done in such a way that it would make sense for someone not familiar with data science
* Methods: Sentiment Analysis, TF-IDF, Topic Modeling, Wordclouds, etc.
  
<img src="https://github.com/MaartenGr/soan/raw/master/images/portfolio_soan.jpg" height="200"/>

---
<a name="whatsapp"/></a>
###  [Reviewer: Character Popularity](https://github.com/MaartenGr/Reviewer)
[Repository](https://github.com/MaartenGr/Reviewer) | [Notebook](https://github.com/MaartenGr/Reviewer/blob/master/notebooks/Overview.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Reviewer/blob/master/notebooks/Overview.ipynb)

**Reviewer** can be used to scrape user reviews from IMDB, generate word clouds based on a custom class-based TF-IDF, and extract popular characters/actors from reviews.
Methods:
* Named Entity Recognition
* Sentiment Analysis
* Scraper
* BERT
  
<p align="center">
<img src="https://github.com/MaartenGr/Reviewer/raw/master/images/wordclouds/result_0.png" height="200"/>
</p>

---
<a name="boardgame"/></a>
###  [Board Game Exploration](https://bgexploration.herokuapp.com/)
[Github](https://github.com/MaartenGr/boardgame) | [site](https://bgexploration.herokuapp.com/)

* Created an application for exploring board game matches that I tracked over the last year
* The application was created for two reasons: 
  * First, I wanted to surprise my wife with this application as we played mostly together
  * Second, the data is relatively simple (5 columns) and small (~300 rows) and I wanted
  to demonstrate the possibilities of analyses with simple data
* Dashboard was created with streamlit and the deployment of the application was through Heroku

<img src="https://raw.githubusercontent.com/MaartenGr/boardgame/master/images/streamlit_gif_small.gif" height="250"/>

---

<a name="disney"/></a>
###  [Statistically Generated Disney Tournament](https://github.com/MaartenGr/DisneyTournament)
[Github](https://github.com/MaartenGr/DisneyTournament) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/DisneyTournament/blob/master/src/Overview.ipynb)

* Created a tournament game for a friend of mine
* Tournament brackets are generated based on a seed score calculated through scraping data from IMDB and RottenTomatoes
* Methods: BeautifulSoup
  
<img src="https://github.com/MaartenGr/DisneyTournament/blob/master/images/Unprotected/scoring.png" height="200"/>

---
<a name="emte"/></a>
###  [Optimizing Emté Routes](https://github.com/MaartenGr/Projects/blob/master/Notebooks/RouteOptimization.ipynb)
[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/RouteOptimization.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/RouteOptimization.ipynb)

* Project for the course Business Analytics in the master
* Optimization of managers visiting a set of cities
* Total of 133 cities, max distance 400km with time and capacity constraints
* Thus, a vehicle routing problem
* Methods: Integer Linear Programming, Tabu Search, Simmulated Annealing, Ant Colony Optimization, Python

<img src="https://github.com/MaartenGr/Projects/blob/master/Images/emte.gif" height="200"/>

---
<a name="pothole"/></a>
###  [Pothole Detection](https://github.com/MaartenGr/PotholeDetection)
[Repository](https://github.com/MaartenGr/PotholeDetection) | [Notebook](https://github.com/MaartenGr/PotholeDetection/blob/master/Pothole.ipynb)| [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/PotholeDetection/blob/master/Pothole.ipynb)

* Image classification of potholes in roads
* Explored different activation functions (Relu, Leaky Relu, Swish, Mish)
* Used EfficientNetB0 and applied transfer learning to quickly get a high accuracy (> 95%)
* Unfreezing certain layers helps with fine tuning the model
* Methods: Deep Learning, TF/Keras, EfficientNetB0

<img src="https://github.com/MaartenGr/PotholeDetection/blob/master/Results/pothole_images.png" height="250"/>

---
<a name="explainable"/></a>
###  [Exploring Explainable ML](https://github.com/MaartenGr/InterpretableML/blob/master/InterpretableML.ipynb)
[Repository](https://github.com/MaartenGr/InterpretableML) | [Notebook](https://github.com/MaartenGr/InterpretableML/blob/master/InterpretableML.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/MaartenGr/InterpretableML/blob/master/InterpretableML.ipynb) |
[Medium](https://towardsdatascience.com/interpretable-machine-learning-1dec0f2f3e6b) 

* Explored several methods for opening the black boxes that are tree-based prediction models
* Models included PDP, LIME, and SHAP

<img src="https://github.com/MaartenGr/InterpretableML/blob/master/Images/occupation.png" height="200"/>

---
<a name="deploy"/></a>
###  [Deploying a Machine Learning Model](https://github.com/MaartenGr/ML-API)
[Repository](https://github.com/MaartenGr/ML-API) | [Medium](https://towardsdatascience.com/how-to-deploy-a-machine-learning-model-dc51200fe8cf) 

* Developed a set of steps necessary to quickly develop your machine learning model
* Used a combination of FastAPI, Uvicorn and Gunicorn to lay the foundation of the API
* The repository contains all code necessary (including dockerfile)

<img src="https://github.com/MaartenGr/ML-API/blob/master/deploy.jpg" height="200"/>

---

<a name="reinforcementlearning"/></a>
###  [Retro Games Reinforcement Learning](https://github.com/MaartenGr/ReinforcementLearning)
[Github](https://github.com/MaartenGr/ReinforcementLearning) | [Notebook](https://github.com/MaartenGr/ReinforcementLearning/blob/master/Reinforcement%20Learning.ipynb)

* An overview of methods for creating state-of-the-art RL-algorithms
* Makes use of Gym, Retro-gym, Procgen, and Stable-baselines
* Associated article will added when published. 

<img src="https://github.com/MaartenGr/ReinforcementLearning/blob/master/Images/procgen_small.gif" height="250"/>

---

<a name="crossvalidation"/></a>
###  [Statistical Cross-Validation Techniques](https://github.com/MaartenGr/validation)
[Repository](https://github.com/MaartenGr/validation) | [Notebook](https://github.com/MaartenGr/validation/blob/master/Validation.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/MaartenGr/validation/blob/master/Validation.ipynb) | [Medium](https://towardsdatascience.com/validating-your-machine-learning-model-25b4c8643fb7) 

* Dived into several more in-depth techniques for validating a model
* Statistical methods were explored for comparing models including the Wilcoxon signed-rank test, McNemar's test, 5x2CV paired t-test and 5x2CV combined F test

<img src="https://github.com/MaartenGr/validation/blob/master/Images/validation.png" height="200"/>

---
<a name="clustering"/></a>
###  [Cluster Analysis: Create, Visualize and Interpret Customer Segments](https://github.com/MaartenGr/CustomerSegmentation/blob/master/Customer%20Segmentation.ipynb)
[Repository](https://github.com/MaartenGr/CustomerSegmentation) | [Notebook](https://nbviewer.jupyter.org/github/MaartenGr/CustomerSegmentation/blob/master/Customer%20Segmentation.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/MaartenGr/InterpretableML/blob/master/Interpretable%20ML.ipynb) | [Medium](https://towardsdatascience.com/cluster-analysis-create-visualize-and-interpret-customer-segments-474e55d00ebb)

* Explored several methods for creating customer segments; k-Means (Cosine & Euclidean) vs. DBSCAN
* Applied PCA and t-SNE for the 3 dimensional exploration of clusters
* Used variance between averages of clusters per variable to detect important differences between clusters

<img src="https://github.com/MaartenGr/CustomerSegmentation/blob/master/dbscan.gif" height="200"/>


---
<a name="featureengineering"/></a>
###  [Exploring Advanced Feature Engineering Techniques](https://github.com/MaartenGr/feature-engineering)
[Repository](https://github.com/MaartenGr/feature-engineering) | [Notebook](https://github.com/MaartenGr/feature-engineering/blob/master/Engineering%20Tips.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/MaartenGr/feature-engineering/blob/master/Engineering%20Tips.ipynb) | [Medium](https://towardsdatascience.com/4-tips-for-advanced-feature-engineering-and-preprocessing-ec11575c09ea)

* Several methods are described for advanced feature engineering including:
  * Resampling Imbalanced Data using SMOTE
  * Creating New Features with Deep Feature Synthesis
  * Handling Missing Values with the Iterative Imputer and CatBoost
  * Outlier Detection with IsolationForest

<img src="https://github.com/MaartenGr/feature-engineering/blob/master/Images/feature.jpg" height="200"/>

---
<a name="auction"/></a>
###  [Predicting and Optimizing Auction Prices](https://github.com/MaartenGr/Projects/blob/master/Notebooks/AuctionAnalysis.ipynb)
[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/AuctionAnalysis.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/AuctionAnalysis.ipynb)

* Data received from an auction house and therefore not made public
* Prediction of value at which an item will be sold to be used as an objective measure
* Optimize starting price such that predicted value will be as high as possible
* Methods: Classification (KNN, LightGBM, RF, XGBoost, etc.), LOO-CV, Genetic Algorithms, Python

<img src="https://github.com/MaartenGr/Projects/blob/master/Images/auction_result.png" height="200"/>

---
<a name="hurdle"/></a>
###  [Statistical Analysis using the Hurdle Model](https://github.com/MaartenGr/Projects/blob/master/Notebooks/AppStoreAnalysis.ipynb)
[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/AppStoreAnalysis.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/AppStoreAnalysis.ipynb)

* Used Apple Store data to analyze which business model aspects influence performance of mobile games
* Two groups were identified and compared, namley early and later entrants of the market
* The impact of entry timing and the use of technological innovation was analyzed on performance
* Methods: Zero-Inflated Negative Binomial Regression, Hurdle model, Python, R

<img src="https://github.com/MaartenGr/Projects/blob/master/Images/appstore.png" height="200"/>

---
<a name="demand"/></a>
###  [Predict and optimize demand](https://github.com/MaartenGr/Projects/blob/master/Notebooks/simulation.ipynb)
[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/simulation.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/simulation.ipynb)

* Part of the course Data-Driven SCM
* Optimizing order quantity based on predicted demand using machine learning methods
* Simulation model was created to check performance of method calculating expected demand
* Additional weather features were included
* Methods: Regression (XGBoost, LightGBM and CatBoost), Bayesian Optimization (Skopt)
  
<img src="https://github.com/MaartenGr/Projects/blob/master/Images/simulation.png" height="200"/>

---
<a name="takeout"/></a>
###  [Analyzing Google Takeout Data](https://github.com/MaartenGr/Projects/blob/master/Notebooks/GoogleTakeout.ipynb)
[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/GoogleTakeout.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/GoogleTakeout.ipynb)

* Analyzing my own data provided by Google Takeout
* Location data, browser search history, mail data, etc.
* Code to analyze browser history is included
* Code to create animation will follow

<p float="left">
  <img src="https://github.com/MaartenGr/Projects/blob/master/Images/location.gif" height="200"/>
  <img src="https://github.com/MaartenGr/Projects/blob/master/Images/website_visits.png" height="200"/>
</p>


---
<a name="cars"/></a>
###  [Cars Dashboard](https://github.com/MaartenGr/cars_dashboard)
[Github](https://github.com/MaartenGr/cars_dashboard) 

* Created a dashboard for the cars dataset using Python, HTML and CSS
* It allows for several crossfilters (see below)

<img src="https://github.com/MaartenGr/cars_dashboard/blob/master/Images/dashboard.gif" height="200"/>

---
<a name="qwixx"/></a>
###  [Qwixx Visualization](https://github.com/MaartenGr/Projects/blob/master/Notebooks/Scorecard.ipynb)
[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/Scorecard.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/Scorecard.ipynb)

* Visualized 16 games I played with my wife
* The result is a heatmap in using the scorecard of Qwixx
* All rights belong to Qwixx (Nuernberger-Spielkarten-Verlag GmbH) 
* The scorecard used for this visualization was retrieved from:
  * https://commons.wikimedia.org/wiki/File:Qwixx_scorecard_nofonts.svg

<img src="https://github.com/MaartenGr/Projects/blob/master/Images/Qwixx.png" height="200"/>

---
<a name="grades"/></a>
###  [Academic Journey Visualization](https://github.com/MaartenGr/Projects/blob/master/Notebooks/Grades.ipynb)
[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/Grades.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/Grades.ipynb)

* A visualization of my 9! year academic journey
* My grades and average across six educational programs versus the average of all other students in the same classes. 
* Unfortunately, no data on the students’ average grade was available for classes in my bachelor. 
* Grades range from 0 to 10 with 6 being a passing grade. 
* Lastly, Cum Laude is the highest academic distinction for the master’s program I followed. 


<p float="left">
  <img src="https://github.com/MaartenGr/Projects/blob/master/Images/Grades.jpg" height="200"/>
  <img src="https://github.com/MaartenGr/Projects/blob/master/Images/Grades_d3.png" height="200"/>
</p>

---

<a name="style"/></a>
### [Neural Style Transfer]()
* For the course deep learning I worked on a paper researching optimal selection of hidden layers to create the most appealing images in neural style transfer while speeding up the process of optimization
* Code is not yet provided as I used most of the following code from [here](https://harishnarayanan.org/writing/artistic-style-transfer/) and I merely explored different layers

<img src="https://github.com/MaartenGr/Projects/blob/master/Images/neural_style_transfer.jpg" height="200"/>

---
<a name="housing"/></a>
###  [Predicting Housing Prices](https://github.com/MaartenGr/Projects/blob/master/Notebooks/HousingPrices.ipynb)
[Github](https://github.com/MaartenGr/Projects/blob/master/Notebooks/HousingPrices.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/Projects/blob/master/Notebooks/HousingPrices.ipynb)

* Project for the course Machine Learning
* Participation of the kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/)
* We were graded on leaderboard scores and I scored 1st of the class with position 33th of the leaderboard
* I used a weighted average of XGBoost, Lasso, ElasticNet, Ridge and Gradient Boosting Regressor
* The focus of this project was mostly on feature engineering

<img src="https://github.com/MaartenGr/Projects/blob/master/Images/kaggle.png" width="70%"/>

---
<a name="fitbit"/></a>
###  [Analyzing FitBit Data](https://github.com/MaartenGr/fitbit/)
[Repository](https://github.com/MaartenGr/fitbit/) | [Github](https://github.com/MaartenGr/fitbit/blob/master/3.%20The%20Final%20Product.ipynb) | [nbviewer](http://nbviewer.jupyter.org/github/MaartenGr/fitbit/blob/master/3.%20The%20Final%20Product.ipynb)

* My first data science project
* I looked at improving the quality of FitBit's heartrate measurement as I realized it wasn't all that accurate
* Used a simple 10-fold CV with regression techniques to fill missing heart rates
* There are many things that I would do differently now like using other validation techniques that are not as sensitive to overfitting on timeseries

<img src="https://github.com/MaartenGr/fitbit/blob/master/fitbit.jpg" height="200"/>

---

## Contact
If you are looking to contact me personally, please do so via E-mail or Linkedin:
- E-mail: maartengrootendorst@gmail.com
- [LinkedIn](https://www.linkedin.com/in/mgrootendorst/)

