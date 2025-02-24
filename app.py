from flask import Flask, render_template, request, jsonify
from c1 import fetch_website_content, extract_pdf_text, initialize_vector_store
from c2 import llm, setup_retrieval_qa
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pickle

app = Flask(__name__)

# Load the disease prediction model and class indices
model = tf.keras.models.load_model('plant_disease_model.h5')
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Example URLs and PDF files
urls = ["https://mospi.gov.in/4-agricultural-statistics"]   #"https://desagri.gov.in/",
pdf_files = ["Data/Farming Schemes.pdf", "Data/farmerbook.pdf", "Data/Disease_treat.pdf", "Data/2.pdf", "Data/1.pdf"]

website_contents = [fetch_website_content(url) for url in urls]

pdf_texts = [extract_pdf_text(pdf_file) for pdf_file in pdf_files]

all_contents = website_contents + pdf_texts
db = initialize_vector_store(all_contents)
chain = setup_retrieval_qa(db)

# Route to handle the plant disease prediction
@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    # Ensure that a file is provided
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess the image (ensure it's resized and normalized as per the model's requirements)
    img = image.load_img(filepath, target_size=(128, 128))  # Adjust target size if needed
    img_array = image.img_to_array(img) / 255.0  # Normalize the image to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the disease using the model
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability class

    # Get class name from the class indices dictionary
    class_name = list(class_indices.keys())[list(class_indices.values()).index(class_idx)]

    print(class_name)
    # Return the predicted disease class name as JSON
    return jsonify({'disease': class_name})

# Chatbot route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the chatbot's questions
@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['messageText'].strip().lower()

    # Handle disease-related queries
    if "disease" in query or "leaf disease" in query:
        return jsonify({"answer": "Please upload an image of the plant leaf for disease prediction."})

    # Handle general chatbot queries
    if query in ["who developed you?", "who created you?", "who made you?", "Name of your creators", "who have created you"]:
        return jsonify({"answer": "I was developed by Udhay and Team"})

    response = chain(query)
    return jsonify({"answer": response['result']})

@app.route("/agribot")
def agribot():
    return render_template("agribot.html")

# Dashboard route (already existing)
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)