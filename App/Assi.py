from flask import Flask, render_template, request
import webbrowser
import requests
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_and_save_graph(file_path, x, y, graph_type):
    plt.figure(figsize=(8, 6))  # Adjust the figure size if needed

    if graph_type == 'line':
        plt.plot(x, y, color='gray', linewidth=2)
    elif graph_type == 'bar':
        plt.bar(x, y, color='gray')
    elif graph_type == 'scatter':
        plt.scatter(x, y, color='gray')

    plt.title('Fancy Graph')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(file_path)
    plt.close()

# Specify the save directory
save_directory = 'App/static/images/'

# Generate and save the first graph (line plot)
file_name = 'graph1.png'
file_path = os.path.join(save_directory, file_name)
x = np.linspace(0, 10, 100)
y = np.sin(x)
generate_and_save_graph(file_path, x, y, 'line')

# Generate and save the second graph (bar plot)
file_name = 'graph2.png'
file_path = os.path.join(save_directory, file_name)
x = ['A', 'B', 'C', 'D', 'E']
y = [4, 7, 2, 5, 9]
generate_and_save_graph(file_path, x, y, 'bar')

# Generate and save the third graph (scatter plot)
file_name = 'graph3.png'
file_path = os.path.join(save_directory, file_name)
x = np.random.rand(100)
y = np.random.rand(100)
generate_and_save_graph(file_path, x, y, 'scatter')


app = Flask(__name__)

# Open the web page on Safari
webbrowser.get('safari').open_new_tab('http://127.0.0.1:5000/')

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/getdata', methods=['POST'])
def getdata():
    
    # Get the track ID from the form
    track_id = request.form['track-id-input']
    #track_id = '11dFghVXANMlKmJXsNCbNl'
    print(track_id)  

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

    # Make a GET request to retrieve information about the track
    track_url = f'https://api.spotify.com/v1/audio-features/{track_id}'  # Construct the URL with the track ID
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    track_response = requests.get(track_url, headers=headers)
    track_data_json = track_response.json()

    # Print the track data in the terminal
    print(track_response)

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
    duration_ms = track_data_json['duration_ms']
    time_signature = track_data_json['time_signature']
    chorus_hit = 30.215
    sections = 10

    X_test = [[danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, time_signature, chorus_hit, sections]]

    #Create a df with the feature names and X Test as first row
    feature_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness','valence', 'tempo', 'duration_ms', 'time_signature', 'chorus_hit', 'sections']
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    #load pickle and test if it works
    xgb_model_loaded = pickle.load(open('xgb_model.pkl', 'rb'))
    #print(xgb_model_loaded.predict(X_test_df))
    # Make predictions using the loaded model
    prediction = xgb_model_loaded.predict(X_test_df)
    if prediction > 0.5:
        prediction_label = 'Hit'
    else:
        prediction_label = 'Flop'

    # Process the track data as needed
    return render_template('index2.html', track_data=track_data_json, danceability=danceability, energy=energy, key=key, loudness=loudness, mode=mode, speechiness=speechiness, acousticness=acousticness, instrumentalness=instrumentalness, liveness=liveness, valence=valence, tempo=tempo, duration_ms=duration_ms, time_signature=time_signature, chorus_hit=chorus_hit, sections=sections, prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)