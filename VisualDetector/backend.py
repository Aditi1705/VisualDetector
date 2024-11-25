from flask import Flask, request, jsonify, send_from_directory
from deepface import DeepFace
import cv2
import os
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            logging.error("No image file in request")
            return jsonify({'error': 'No image file in request'}), 400
        
        file = request.files['image']
        if file.filename == '':
            logging.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        try:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.error(f"Error reading image: {str(e)}")
            return jsonify({'error': 'Failed to read image'}), 400

        if img is None:
            logging.error("Image not loaded correctly")
            return jsonify({'error': 'Failed to load image'}), 400

        logging.info(f"Image shape: {img.shape}")

        try:
            result = DeepFace.analyze(img, actions=['emotion', 'age', 'gender', 'race'], 
                                      enforce_detection=False)
            logging.info(f"DeepFace result: {result}")
            
            if isinstance(result, list) and len(result) > 0:
                analysis = result[0]
                emotion = analysis.get('dominant_emotion', 'unknown')
                age = analysis.get('age', 'unknown')
                gender = analysis.get('dominant_gender', 'unknown')
                ethnicity = analysis.get('dominant_race', 'unknown')
            else:
                emotion = age = gender = ethnicity = 'unknown'
                logging.warning("Unexpected result format from DeepFace.analyze")

            return jsonify({
                'emotion': emotion, 
                'age': age, 
                'gender': gender, 
                'ethnicity': ethnicity,
                'face_detected': True
            })

        except ValueError as ve:
            if "Face could not be detected" in str(ve):
                logging.warning("Face could not be detected in the image")
                return jsonify({
                    'error': 'No face detected in the image',
                    'face_detected': False
                }), 200
            else:
                logging.error(f"Unexpected ValueError: {str(ve)}")
                return jsonify({'error': 'Failed to analyze image'}), 500

        except Exception as e:
            logging.error(f"Error analyzing image: {str(e)}")
            return jsonify({'error': 'Failed to analyze image'}), 500
    else:
        return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'frontend.html')

if __name__ == '__main__':
    app.run(debug=True)