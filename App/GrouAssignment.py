from flask import Flask, render_template, request
import webbrowser

app = Flask(__name__)

# Open the web page on Safari
webbrowser.get('safari').open_new_tab('http://127.0.0.1:5000/')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    danceability = float(request.form['danceability'])
    energy = float(request.form['energy'])
    key = int(request.form['key'])
    loudness = float(request.form['loudness'])
    mode = int(request.form['mode'])
    
    #load pickle and test if it works
    #xgb_model_loaded = pickle.load(open('xgb_model.pkl', 'rb'))
    #print(xgb_model_loaded.predict(X_test))

    # Make the prediction based on the input values
    if danceability > 0.5 and energy > 0.5 and key > 1 and loudness > 0.5 and mode == 0:
        prediction = "Hit"
    else:
        prediction = "Flop"
        
    # Return the prediction result to the webpage
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)