# Welcome to RythmRadar!

## Project Intro
- Motivation: This code is the baseline for the Rhythm Radar Program. The motivation is to give insights on a songs popularity and predict its success. Our deliverable is threefold: giving a hit/ flop prediciton for a song, predicitng its social media popularity and giving it a recommendation which feature to tune to make it a Hit.
- Objective:

## Dataset

## Files Overview

### Assi.py (Short form for Aissignment)
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
### 01-data_clean.ipynb
Dataframe cleaning for modelling and scale of all the variabels
### 02-ml_models.ipynb
Evaluation of different machine learning models (LR, RF, XGB, ...) and selection of the most appropriate based on evaluation metrics
### 03-xgb.ipynb
Selected XGBoost model fitting and prediction
### xgb_model_genre.pkl
Saved the XGBoost model in a pickle file to use across webapp, without long computing time
