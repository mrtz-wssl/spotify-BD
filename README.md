# Welcome to RythmRadar!

## Project Intro
- Motivation: This code is the baseline for the Rhythm Radar Program. The motivation is to give insights on a songs popularity and predict its success. 
- Objective: Our objective is to provide the music industry a tool to assess songs. Our deliverable is threefold; giving a hit/ flop prediciton for a song, predicitng its social media popularity and giving it a recommendation which feature to tune to make it a Hit.

## Dataset
We have used two datasets: A spotify dataset and a tiktok dataset
- Spotify dataset: This is a dataset consisting of features for tracks fetched using Spotify's Web API. The tracks are labeled '1' or '0' ('Hit' or 'Flop') depending on some criterias of the author.
	This dataset can be used to make a classification model that predicts whether a track would be a 'Hit' or not.
- Tiktok dataset: This dataset comprises various features extracted from TikTok. The dataset includes different attributes, including a popularity variable reaching from 1 to 100. This dataset can be utilized to develop classification models aimed at predicting whether a song will be successfull on TikTok and trend on the platform. 
	
## Files Overview
The code is divided in different files. The main code is saved in the .ipynb files, naming:
- 00-merging dataframes
- 01-data_clean
- 02-ml_models
- 03-xgb
- 04-social media prediction
- 05-feature recommendation
- 06-song recommendation
The app, which creates a user interface for the model and concatenates the model to a product, is saved in the folder "App"

### Assi.py (Short form for Assignment)
- The code imports the necessary modules, including Flask, matplotlib, and sklearn.
- There are two graph generation functions: generate_and_save_graph() and generate_feature_graph(). These functions generate and save different types of graphs using matplotlib.
- The code defines a Flask web application using the Flask class and sets up routes for ('/') and a form submission ('/getdata').
- When the user triggers this route '/', the index() function is called, which renders an HTML template named index2.html.
- When the user submits the form ('/getdata'), the getdata() function is called. This function performs various tasks, including making requests to the Spotify API to retrieve track information, loading machine learning models, and making predictions based on the input data.
- The predicted results are then used to generate additional graphs using the generate_feature_graph() function.
- The code also includes a section for tuning recommendation features based on specific values for different features.
### index2.html
- The HTML file starts with the <!DOCTYPE html> declaration and contains an HTML <head> section where metadata and stylesheets are defined.
- The <body> section begins with a container <div> and includes a heading, tabs navigation, and tab content.
- The tabs navigation is created using an unordered list (<ul>) with each tab item represented by a list item (<li>). Each list item contains an anchor (<a>) element with a unique ID and a data-toggle attribute to enable tab switching.
- The tab content is defined within a <div> with the class tab-content. Each tab pane is represented by a <div> with a unique ID and the class tab-pane. The first tab pane has an additional class semitransparent-box for styling.
- Inside each tab pane, there is content specific to that tab. The first tab pane contains a welcome message and a form with three buttons. The form has an action attribute pointing to /getdata and a method of post.
### 00-merging dataframes.ipynb
This dataframe first creates a merged dataset, then conducts different tests on the merged dataset and finally exports the data to make it accessible for further use.
### 01-data_clean.ipynb
Dataframe cleaning for modelling and scale of all the variabels.
### 02-ml_models.ipynb
Evaluation of different machine learning models (LR, RF, XGB, ...) and selection of the most appropriate based on evaluation metrics.
### 03-xgb.ipynb
Selected XGBoost model fitting and prediction.
### 04-social media prediction.ipynb
This code creates a model to predict social media success built with a random forest classifier. The social media target is a binary variable.
### 05-feature_recommendation.ipynb
Based on the model created to predict hit and flop this code gives a recommendation on how to tune a feature to reach a hit. It goes through a selection of features (starting with the one that is easyest to change) and tries to adapt them.
### 06-song recommendation.ipynb
This code gives similar songs to the one inputed.
### 07-genre extraction.ipynb
This code extracts the genre assigned to a an artist on Spotify and adds it to the dataframe to be merged at a later point. Additionally, it also groups the genres to parent-genres that are easier to work with. 
### xgb_model_genre.pkl
Saved the XGBoost model in a pickle file to use across webapp, without long computing time.
### spotify_new_songs.ipynb
Populate our data with new songs from Spotify through API requests. Also, the songs go through the audio analysis tool of Spotify to extract audio features for our model.
