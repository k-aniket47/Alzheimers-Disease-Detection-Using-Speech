import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np 
import matplotlib.pyplot as plt
import base64


import warnings
warnings.filterwarnings("ignore")


#load the model
path_to_model='model.hdf5'
print("Loading the model..")
model = load_model(path_to_model,compile=False)
model.compile(optimizer='adam',
              metrics=['accuracy'],
              loss='categorical_crossentropy')
print("Done!")


# Define the categories
category = {
    0: 'The Person is not having Alzheimer',
    1: 'The Person is having Alzheimer'
}


# Define the path where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Define a function to predict the category of an audio file
def predict_audio_category(audio_file_path):
    # Create spectrogram of the audio
    save_path = os.path.dirname(audio_file_path) + '/'
    create_spectogram(os.path.basename(audio_file_path), os.path.dirname(audio_file_path) + '/', save_path)
    
    # Load and preprocess the spectrogram image
    # img = image.load_img(save_path + os.path.basename(audio_file_path).replace('.wav', '.jpg'), target_size=(250, 250))
    img = image.load_img('static/' + os.path.basename(audio_file_path).replace('.wav', '.jpg'), target_size=(250, 250))
    img_array = image.img_to_array(img)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.
    
    # Make prediction using the trained model
    prediction = model.predict(img_processed)
    index = np.argmax(prediction)
    
    # Get the predicted category
    predicted_category = category.get(index)
    
    return predicted_category


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define a route for the file upload page
@app.route('/upload')
def upload_file():
    return render_template('upload.html')


# Define a route for the file upload action
@app.route('/uploader', methods=['GET', 'POST'])
def upload_file_action():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        # Save the file to the uploads folder
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Predict the category of the file
        audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        predicted_category = predict_audio_category(audio_file_path)
        # Generate spectrogram image URL
        image_url = '/static/' + filename.replace('.wav', '.jpg')
        # Render the result on the same page
        return render_template('index.html', prediction=predicted_category, image_url=image_url)




if __name__ == '__main__':
    app.run(debug=True)
