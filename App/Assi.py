from flask import Flask, render_template, request
import webbrowser
import requests
import json

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
    acousticness = track_data_json['acousticness']
    danceability = track_data_json['danceability']
    duration_ms = track_data_json['duration_ms']
    energy = track_data_json['energy']
    instrumentalness = track_data_json['instrumentalness']

    # Process the track data as needed
    return render_template('index2.html', track_data=track_data_json, acousticness=acousticness, danceability=danceability, duration_ms=duration_ms, energy=energy, instrumentalness=instrumentalness)
    
if __name__ == '__main__':
    app.run(debug=True)


