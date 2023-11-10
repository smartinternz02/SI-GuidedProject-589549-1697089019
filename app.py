import numpy as np
import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

# Remove spaces in the model file name
model = load_model("wcv.h5")

app = Flask(__name__)

@app.route('/')
def index():  # Remove parentheses and provide a function name
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template("index.html")  # Remove extra spaces

@app.route('/input')
def input1():
    return render_template("input.html")  # Remove extra spaces

@app.route('/predict', methods=["GET", "POST"])
def res():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        folder_path = os.path.join(basepath, 'uploads')  # Rename the folder variable
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # Create the 'uploads' folder if it doesn't exist
        filepath = os.path.join(folder_path, f.filename)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data), axis=1)
        index = ['Boletus', 'Lactarius', 'Russula']
        result = index[prediction[0]]
        print(result)
        return render_template('output.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)





















