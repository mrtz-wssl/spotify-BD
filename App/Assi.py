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
#     plt.figure(figsize=(8, 6))  # Adjust the figure      size if needed 

#     # if graph_type == 'line':
#     #     plt.plot(x, y, color='gray', linewidth=2)      
#     # elif graph_type == 'bar':
#     #     plt.bar(x, y, color='gray')
#     # elif graph_type == 'scatter':
#     #     plt.scatter(x, y, color='gray')©
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

    #Save the chart /static/images/genre_heatmap.pngd
    plt.savefig('genre_heatmap.png')
    plt.close()
    # plt.show
    return 'Chart generated and saved!'

def generate_feature_graph(file_path, x, y):
    # Calculation of the mean song tempo
    mean_tempo = np.mean(y)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x, y, color='#1DB954', linewidth=0.3)
    ax.plot(x, y, 'o', color='#1DB954', linewidth=0.3, markersize=10)  # Green points
    ax.axhline(y=0, color='#1DB954')  # Green horizontal line
    ax.grid(axis='x', color='#1DB954', linestyle='-', linewidth=0.5)  # Green grid lines

    # Set the background color to slightly transparent black
    fig.patch.set_facecolor('black')
    fig.patch.set_alpha(0.7)  # Adjust transparency
    ax.fill_between(x, y, color='#1DB954', alpha=0.2)

    # Set the text color to white for the grid labels and border
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    fig.savefig(file_path, facecolor=fig.get_facecolor(), transparent=True)  # Save with black background
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
webbrowser.get('safari').open_new_tab('http://127.0.0.1:5000')

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
    print('Search Data', search_data_json)   

    track_id = search_data_json['tracks']['items'][0]['id']

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
    duration_ms = track_data_json['duration_ms']
    time_signature = track_data_json['time_signature']
    chorus_hit = 30.215
    sections = 10
    
    # Variables for Social media prediction
    key_A = 0
    key_A__Bb = 0
    key_B = 0
    key_C = 0
    key_C__Db = 0
    key_D = 0
    key_D__Eb = 0
    key_E = 0
    key_F = 0
    key_F__Gb = 0
    key_G = 0
    key_G__Ab = 0
    mode_major = 1
    mode_minor = 0
    era_00s = 0
    era_10s = 0
    era_20s = 0
    era_60s = 1
    era_70s = 0
    era_80s = 0
    era_90s = 0
    main_parent_genre_Blues_and_Jazz = 0
    main_parent_genre_Classical_and_Opera = 0
    main_parent_genre_Country_and_Folk = 0
    main_parent_genre_Electronic_Music_and_Dance = 0
    main_parent_genre_Other = 0
    main_parent_genre_Pop = 1
    main_parent_genre_Rap_and_Hip_Hop = 0
    main_parent_genre_Reggae_and_Ska = 0
    main_parent_genre_Rock = 0
    main_parent_genre_World_Music = 0

    X_test = [[danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, chorus_hit, sections]]
    
    #Create a df with the feature names and X Test as first row
    feature_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness','valence', 'tempo', 'time_signature', 'chorus_hit', 'sections']
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    #load pickle and test if it works
    xgb_model_loaded = pickle.load(open('xgb_model.pkl', 'rb'))
    #print(xgb_model_loaded.predict(X_test_df))
    # Make predictions using the loaded model


    df = pd.read_csv('TikTokSpotifyMerged.csv')
    df = df.reset_index(drop=True)

    # Check whether the song is in the Spotify data set
    print(f"Checking track ID: {track_id}")
    if track_id in df['track_id'].values:
        # If yes, retrieve the classification from the data set
        prediction = df.loc[df['track_id'] == track_id, 'target'].values[0]
        print('Failsafe: ', prediction)
        print(df.loc[df['track_id'] == track_id, 'target'])
        if prediction > 0.5:
            prediction_label = "This Song has HIT Potential!"
        else:
            prediction_label = "This Song has FLOP Potential!"
    else:
        # If not, predict it with the model
        prediction = xgb_model_loaded.predict(X_test_df)
        print('Song will be predicted')
        if prediction > 0.5:
            prediction_label = "This Song has HIT Potential!"
        else:
            prediction_label = "This Song has FLOP Potential!"



# FOR SM-TARGET PREDICTION
    X_test_sm = [[duration_ms, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, key_A, key_A__Bb, key_B, key_C, key_C__Db, key_D, key_D__Eb, key_E, key_F, key_F__Gb, key_G, key_G__Ab, mode_major, mode_minor, era_00s, era_10s, era_20s, era_60s, era_70s, era_80s, era_90s, main_parent_genre_Blues_and_Jazz, main_parent_genre_Classical_and_Opera, main_parent_genre_Country_and_Folk, main_parent_genre_Electronic_Music_and_Dance, main_parent_genre_Other, main_parent_genre_Pop, main_parent_genre_Rap_and_Hip_Hop, main_parent_genre_Reggae_and_Ska, main_parent_genre_Rock, main_parent_genre_World_Music ]]
    
    #Create a df with the feature names and X Test as first row
    feature_names_sm = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness','valence', 'tempo', 'key_A', 'key_A#_/_Bb', 'key_B', 'key_C', 'key_C#_/_Db', 'key_D','key_D#_/_Eb', 'key_E', 'key_F', 'key_F#_/_Gb', 'key_G', 'key_G#_/_Ab', 'mode_major', 'mode_minor', 'era_00s', 'era_10s', 'era_20s', 'era_60s', 'era_70s', 'era_80s', 'era_90s', 'main_parent_genre_Blues_and_Jazz', 'main_parent_genre_Classical_and_Opera', 'main_parent_genre_Country_and_Folk', 'main_parent_genre_Electronic_Music_and_Dance', 'main_parent_genre_Other', 'main_parent_genre_Pop', 'main_parent_genre_Rap_and_Hip_Hop', 'main_parent_genre_Reggae_and_Ska', 'main_parent_genre_Rock', 'main_parent_genre_World_Music']
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
    df = df.drop(columns=['track_id', 'track', 'artist', 'popularity', 'duration_ms', 'key', 'mode', 'main_parent_genre', 'era', 'target', 'sm_target', 'tiktok', 'spotify', 'chorus_hit','tempo' ])
    print (df.columns)
    scaler = MaxAbsScaler()
    scaler.fit(df)

    # Scale the DataFrame
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

    # Scale the graph_data in relation to the entire dataset   
    graph_data_scaled = scaler.transform(graph_data)

    # Generate the graph
    generate_feature_graph(graph_file_path, feature_names, graph_data_scaled[0]) 

    data_tofindtrack = pd.read_csv('Spotify Data/data-clean.csv')
    #create duration_ms
    data_tofindtrack['track_seconds'] = data_tofindtrack['duration_ms'] / 1000
    # Drop unnecessary columns
    data_tofindtrack = data_tofindtrack.drop(["era", "sm_target", "popularity", "tiktok", "spotify", "track", "artist", "duration_ms", "key", "mode", "main_parent_genre"], axis=1)

    tuningfeatures = ["loudness", "danceability", "acousticness","chorus_hit","sections", 
                  "energy", "speechiness","instrumentalness","liveness",
                  "valence","tempo","time_signature"]
    
    values = [1, 0.8, 1.2, 0.6, 1.4, 0.4, 1.6, 0.2]
    
    output1 = "Very sorry, I got no recommendation for you."
    #---------------------------------------------------------------------------------
    
    track_df = data_tofindtrack[data_tofindtrack['track_id'] == track_id]
    track_df = track_df.drop(["track_id"], axis=1)
    print(track_df)
    for feature in tuningfeatures:
        for value in values:
            song_copy = track_df.copy()  # Create a copy of the DataFrame
            song_copy[feature] = song_copy[feature] * value 
            print("check2")
            if song_copy.empty:
                continue
            pred = xgb_model_loaded.predict(song_copy)
            #print(value)
            print("check3")

            if pred[0] > 0 and prediction > 0.5:
                print ("HIT reached")
                output1 = "You already hava a HIT prediction but maybe try to change " + str(feature) +" by " + str(value) + " to improve your song."
            else:
                print ("FLOP reached")
                output1 = "Hey, maybe try to change " + str(feature) +" by " + str(value) + " to reach a HIT prediction."
    
    #---------------------------------------------------------------------------------

    # Process the track data as needed
    return render_template('index2.html', 
                           track_data=track_data_json,
                           danceability=danceability, 
                           energy=energy, 
                           key=key, 
                           loudness=loudness, 
                           mode=mode, 
                           speechiness=speechiness,
                           acousticness=acousticness, 
                           instrumentalness=instrumentalness, 
                           liveness=liveness, 
                           valence=valence, 
                           tempo=tempo, 
                           time_signature=time_signature, 
                           chorus_hit=chorus_hit, 
                           sections=sections, 
                           prediction=prediction_label, 
                           prediction2=prediction_label2,
                           output1=output1
                        )

if __name__ == '__main__':
    app.run(debug=True)


