from flask import Flask, render_template, request, send_file
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.preprocessing import MaxAbsScaler
import webbrowser
import requests
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# def generate_and_save_graph(file_path, x, y, graph_type):
#     plt.figure(figsize=(8, 6))  # Adjust the figure size if needed 

#     # if graph_type == 'line':
#     #     plt.plot(x, y, color='gray', linewidth=2)
#     # elif graph_type == 'bar':
#     #     plt.bar(x, y, color='gray')
#     # elif graph_type == 'scatter':
#     #     plt.scatter(x, y, color='gray')
#     plt.savefig('mean-features-hits.png', dpi=300, bbox_inches='tight') 
#     # plt.title('Fancy Graph')
#     # plt.xlabel('X')
#     # plt.ylabel('Y')
#     #plt.savefig(file_path)
#     plt.close()

def generate_and_save_graph():
    # Generate the chart
    # ...

    # Save the chart as "mean-features-hits.png"
    plt.savefig('mean-features-hits.png', dpi=300, bbox_inches='tight')

    #Save the chart /static/images/genre_heatmap.png
    plt.savefig('genre_heatmap.png')
    plt.close()
    # plt.show
    return 'Chart generated and saved!'

def generate_feature_graph(file_path, x, y):
    # Calculation of the mean song tempo
    mean_tempo = np.mean(y)

    x.drop()
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x, y, color='#1DB954')
    ax.plot(x, y, 'o', color='white', linewidth=5)
    ax.axhline(y=0, color='white')
    ax.grid(axis='x', color='white', linestyle='-', linewidth=0.5)

    # Set the background color to black
    fig.set_facecolor('black')
    
    ax.fill_between(x, y, color='#1DB954', alpha=0.3)

    fig.savefig(file_path, facecolor=fig.get_facecolor())  # Save with black background
    plt.close(fig)

# Specify the save directory
save_directory = 'App/static/images/'

# Generate and save the first graph (line plot)
file_name = 'genre_heatmap.png'
file_path = os.path.join(save_directory, file_name)
x = np.linspace(0, 10, 100)
y = np.sin(x)
# generate_and_save_graph(file_path, x, y, 'line')
generate_and_save_graph()
# Generate and save the second graph (spider chart)
# file_name = 'mean-features-hits.png'
# file_path = os.path.join(save_directory, file_name)
# # x = ['A', 'B', 'C', 'D', 'E']
# # y = [4, 7, 2, 5, 9]
# generate_and_save_graph(file_path, x, y, 'spider')

#Generate and save second graph (spider chart) 

# Generate and save the third graph (scatter plot)
file_name = 'graph3.png'
file_path = os.path.join(save_directory, file_name)
x = np.random.rand(100)
y = np.random.rand(100)
# generate_and_save_graph(file_path, x, y, 'scatter')


app = Flask(__name__)

# Open the web page on Safari
webbrowser.get('safari').open_new_tab('http://127.0.0.1:5000/')

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/getdata', methods=['POST'])
def getdata():
    
    # Get the artist and track name from the form 
    artist_name = request.form['artist-name-input']
    track_name = request.form['track-id-input']
    print(artist_name, track_name)  

    # Make a POST request to obtain the access token
    token_url = 'https://accounts.spotify.com/api/token'
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': '08af0849debb4c818c1819497dd7e9c1',
        'client_secret': 'df10fe40e77b4e159af69b087dcc78f0'
    }
    token_response = requests.post(token_url, data=token_data)
    token_response_data = token_response.json()
    access_token = token_response_data['access_token']

    # Search for the track on Spotify
    search_url = 'https://api.spotify.com/v1/search'
    search_params = {
        'q': f'artist:{artist_name} track:{track_name}',
        'type': 'track',
        'limit': 1
    }

    headers = {'Authorization': f'Bearer {access_token}'}

    search_response = requests.get(search_url, headers=headers, params=search_params)
    print('Search Prompt', search_response)
    search_data_json = search_response.json()   

    track_id = search_data_json['tracks']['items'][0]['id']

    # artist_name = search_data_json['tracks']['items'][0]['artists'][0]['name'] - THIBAUD

    #Match song name with the trackid
    # track_name = search_data_json['tracks']['items'][0]['name'] - THIBAUD

    #Make a GET request to retrive information about the track based on either the track name or the artist name 
    # track_url = f'https://api.spotify.com/v1/search?q=track:{track_name}&type=track&limit=1'
    # track_url = f'https://api.spotify.com/v1/search?q=artist:{artist_name}&type=artist&limit=1'
    # track_url = f'https://api.spotify.com/v1/search?q=artist:{artist_name} track:{track_name}&type=track&limit=1'

    # Make a GET request to retrieve information about the track
    track_url = f'https://api.spotify.com/v1/audio-features/{track_id}'  # Construct the URL with the track ID
    
    track_response = requests.get(track_url, headers=headers)
    track_data_json = track_response.json()
    # Print the track data in the terminal
    print('Spotify responded: ', track_response)

    # Extract the track data from the response
    danceability = track_data_json['danceability']
    energy = track_data_json['energy']
    key = track_data_json['key']
    loudness = track_data_json['loudness']
    mode = track_data_json['mode']
    speechiness = track_data_json['speechiness']
    acousticness = track_data_json['acousticness']
    instrumentalness = track_data_json['instrumentalness']
    liveness = track_data_json['liveness']
    valence = track_data_json['valence']
    tempo = track_data_json['tempo']
    # duration_ms = track_data_json['duration_ms']
    time_signature = track_data_json['time_signature']
    chorus_hit = 30.215
    sections = 10

    X_test = [[danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, chorus_hit, sections]]
    
    #Create a df with the feature names and X Test as first row
    feature_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness','valence', 'tempo', 'time_signature', 'chorus_hit', 'sections']
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    #load pickle and test if it works
    xgb_model_loaded = pickle.load(open('xgb_model.pkl', 'rb'))
    #print(xgb_model_loaded.predict(X_test_df))
    # Make predictions using the loaded model
    prediction = xgb_model_loaded.predict(X_test_df)
    if prediction > 0.5:
        prediction_label = "It's a Hit!"
    else:
        prediction_label = "It's a Flop!"



# FOR SM-TARGET PREDICTION
    X_test_sm = [[danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, chorus_hit, sections]]
    
    #Create a df with the feature names and X Test as first row
    feature_names_sm = ['duration_ms','danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness','valence', 'tempo', 'era', 'main_parent_genre']
    X_test_df_sm = pd.DataFrame(X_test_sm, columns=feature_names_sm)
    
    #load pickle and test if it works
    rf_model_loaded = pickle.load(open('randomforest_model.pkl', 'rb'))
    #print(xgb_model_loaded.predict(X_test_df))
    # Make predictions using the loaded model
    prediction2 = rf_model_loaded.predict(X_test_df_sm)
    if prediction2 > 0.5:
        prediction_label2 = "It's a Social Media Hit!"
    else:
        prediction_label2 = "It's a Social Media Flop!"

#END SM-TARGET PREDICTION
    #### Create a graph: ####
    graph_file_path = os.path.join('App/static/images', 'feature_graph.png')

    
    # Load your data into a pandas DataFrame
    df = pd.read_csv('TikTokSpotifyMerged.csv')

    # Calculate mean values
    mean_tempo = df['tempo'].mean()
    mean_chorus_hit = df['chorus_hit'].mean()

    deviation_tempo = abs(tempo - mean_tempo)
    deviation_chorus_hit = abs(chorus_hit - mean_chorus_hit)

    # graph_data = [[danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, chorus_hit, sections]]
    # feature_names = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness','valence', 'deviation_tempo', 'time_signature', 'deviation_chorus_hit', 'sections']
    graph_data = [[danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, time_signature, sections]]
    feature_names = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness','valence', 'time_signature', 'sections']


    # Modify your feature data
    df = df.drop(columns=['track_id', 'track', 'artist', 'popularity', 'duration_ms', 'key', 'mode', 'main_parent_genre', 'era', 'target', 'sm_target', 'tiktok', 'spotify' ])
    scaler = MaxAbsScaler()
    scaler.fit(df)

    # Scale the DataFrame
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

    # Scale the graph_data in relation to the entire dataset   
    graph_data_scaled = scaler.transform(graph_data)

    # Generate the graph
    generate_feature_graph(graph_file_path, feature_names, graph_data_scaled[0]) 





# Process the track data as needed
    return render_template('index2.html', track_data=track_data_json, danceability=danceability, energy=energy, key=key, loudness=loudness, mode=mode, speechiness=speechiness, acousticness=acousticness, instrumentalness=instrumentalness, liveness=liveness, valence=valence, tempo=tempo, time_signature=time_signature, chorus_hit=chorus_hit, sections=sections, prediction=prediction_label, prediction2=prediction_label2)

if __name__ == '__main__':
    app.run(debug=True)


