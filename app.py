import os  
from flask import Flask, request, send_from_directory  
import cv2  
import numpy as np  
  
app = Flask(__name__)  
UPLOAD_FOLDER = "uploaded_images"  
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER  
  
@app.route("/upload", methods=["POST"])  
def upload_image():  
    if request.method == "POST":  
        if "file" not in request.files:  
            return "No file part", 400  
  
        file = request.files["file"]  
  
        if file.filename == "":  
            return "No selected file", 400  
  
        if file:  
            filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)  
            file.save(filename)  
  
            # Load the image to project  
            image_to_project = cv2.imread(filename)    
  
            return "Image uploaded and displayed", 200  
  
    return "Invalid request method", 405  
  
if __name__ == "__main__":  
    if not os.path.exists(UPLOAD_FOLDER):  
        os.makedirs(UPLOAD_FOLDER)  
  
    app.run(host="0.0.0.0", port=5001, debug=True)  
