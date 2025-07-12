# Emotion Detection Web App

[![Live Demo](https://img.shields.io/badge/Demo-Live-green)](https://emoji-weex.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange)](https://tensorflow.org)

A real-time emotion classification system trained on FER-2013 dataset, deployed as a web application.


## 🧠 AI/ML Implementation

### Model Architecture
```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 46, 46, 32)        320       
                                                                 
 batch_normalization (BatchN  (None, 46, 46, 32)       128       
 ormalization)                                                   
                                                                 
 max_pooling2d (MaxPooling2D  (None, 23, 23, 32)       0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 23, 23, 32)        0         
                                                                 
 conv2d_1 (Conv2D)           (None, 21, 21, 128)       36992     
                                                                 
 global_average_pooling2d (G  (None, 128)              0         
 lobalAveragePooling2D)                                          
                                                                 
 dense (Dense)               (None, 7)                 903       
                                                                 
=================================================================
Total params: 38,343
Trainable params: 38,279
Non-trainable params: 64
Training Details
Dataset: FER-2013 (35,887 grayscale 48x48 images)

Augmentation:
ImageDataGenerator(rotation_range=15,
                  zoom_range=0.1,
                  horizontal_flip=True,
                  brightness_range=[0.9,1.1])
Optimization:
Adam optimizer (lr=0.001)

Class weights for imbalance (Disgust: 10x weight)

Early stopping (patience=5)

Metric	Value
Val Accuracy	68.2%
Inference Speed	120ms
Model Size	8.4MB

🌐 Web Implementation

Tech Stack
Frontend: Vanilla JS + Canvas API

Backend: Flask (Python)

Deployment: Dockerized on Render

Key Features
1. Real-time camera emotion detection

2. Image upload processing

3. Confidence visualization

4. Mobile-responsive design

🚀 Deployment
Local Setup

git clone https://github.com/your-repo/emotion-detection.git
cd emotion-detection
pip install -r requirements.txt
python app.py
Docker Build
docker build -t emotion-app .
docker run -p 5000:5000 emotion-app

📂 Project Structure
emotion-detection/
├── app.py                # Flask application
├── emotion_model.h5      # Trained model
├── static/               # JS/CSS assets
├── templates/            # HTML files
├── requirements.txt      # Python dependencies
└── Dockerfile            # Container configuration

📈 Performance Comparison
Model	Accuracy	Size	Speed
Our CNN	68.2%	8.4MB	120ms
Mini-Xception	72.1%	22MB	210ms

🤝 Contributing
PRs welcome! Please open an issue first to discuss changes.
