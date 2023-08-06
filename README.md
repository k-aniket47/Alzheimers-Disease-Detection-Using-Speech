# Alzheimers-Disease-Detection-Using-Spontaneous-Speech

This repository contains a Flask web application for detecting Alzheimer's disease from audio recordings using a deep learning model. The app uses librosa for audio processing, matplotlib for generating spectrograms, and tensorflow for model inference.
#### Dataset:
https://luzs.gitlab.io/adress/

![et_classification-Page-9 drawio](https://github.com/k-aniket47/Alzheimers-Disease-Detection-Using-Speech/assets/79148315/a2db829d-6a06-4399-9652-bcacc0175d66)


### How to Run
- Clone this repository to your local machine.
- Navigate to the project directory.
  
#### To Train the model:
- First, Preprocess the data 
- In the Notebook Folder, open Process_data.ipynb file.
- Give the link of dataset and path for a new directory to save the spectogram images.

- 2: Train the model 
- In the notebook folder, open XceptionNet.ipybn file 
- Give the path of Spectogram images dataset path for training and audio dataset path for prediction.
- Train the model and give the path to save the weights of best model.


#### To run the Webapp
- Open app.py code and run it.
- It will give a link of Flask webapp, open it in a browser.
- Upload the audio file from the test dataset to predict.

The application will be available at http://localhost:5000.

### Web-App

![image](https://github.com/k-aniket47/Alzheimers-Disease-Detection-Using-Speech/assets/79148315/642bd651-fb61-4a22-b88e-09a3904e2c4b)

![image](https://github.com/k-aniket47/Alzheimers-Disease-Detection-Using-Speech/assets/79148315/8bc7b6ac-2cc8-494f-b826-33d350fa22f1)

![Screenshot 2023-04-10 220544](https://github.com/k-aniket47/Alzheimers-Disease-Detection-Using-Speech/assets/79148315/759f12bf-5a75-4fc8-b11a-b2f353f89446)



Project Structure
The project is organized as follows:

- static/: Contains static assets such as uploaded audio files and generated spectrogram images.
- templates/: Contains HTML templates for rendering the web pages.
- index.html: Home page template displaying the file upload form and the prediction result.
- upload.html: File upload page template for uploading audio files.
- app.py: Flask application script containing the routes and model prediction functions.
### How It Works
The user visits the home page and can access the file upload page to submit an audio recording.
Once the user uploads an audio file, the server creates a spectrogram image from the audio using librosa and saves it.
The model then predicts the category of the audio file based on the spectrogram image.
The result (whether the person has Alzheimer's or not) is displayed to the user along with the generated spectrogram image.



