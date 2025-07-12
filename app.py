from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
import logging
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and resources
try:
    # Try loading with compile=False first to avoid optimizer issues
    model = tf.keras.models.load_model("emotion_model.h5", compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy')  # Recompile with simple optimizer
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    logger.info("Model and resources loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or resources: {str(e)}")
    raise

def validate_image(image_bytes):
    """Validate the image and convert to proper format"""
    try:
        # Try with OpenCV first
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback to PIL if OpenCV fails
            img = Image.open(io.BytesIO(image_bytes))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        raise ValueError("Invalid image format")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        # Handle base64 image (with or without data URL prefix)
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({"error": "Invalid base64 encoding"}), 400

        # Process image
        try:
            img = validate_image(image_bytes)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        # Detect face
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return jsonify({"error": "No face detected. Try a clearer photo."}), 400

        # Process first detected face
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        
        try:
            face_resized = cv2.resize(face, (48, 48))
            face_normalized = face_resized.astype("float32") / 255.0
            img_tensor = np.expand_dims(face_normalized, axis=(0, -1))
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return jsonify({"error": "Error processing face image"}), 400

        # Make prediction
        try:
            prediction = model.predict(img_tensor)[0]
            result = [
                {
                    "emotion": EMOTIONS[i], 
                    "confidence": float(f"{p*100:.2f}")
                } 
                for i, p in enumerate(prediction)
            ]
            result.sort(key=lambda x: -x["confidence"])
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({"error": "Error making prediction"}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
