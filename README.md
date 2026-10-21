# AI Gesture-to-Speech Assistive System 🤟🗣️

An end-to-end computer vision and machine learning pipeline that translates real-time hand gestures into natural spoken sentences for individuals with speech impairments. 

## 🚀 Features
- **Custom AI Model:** Trained on a custom dataset of 1,200 samples across 6 gestures.
- **Smart Sentence Generation:** Translates raw gestures (e.g., `help` + `water`) into natural sentences (*"Please help me get some water"*).
- **Emergency Alert:** Holding a gesture for 5 seconds triggers an audio emergency alert.
- **Real-time HUD:** On-screen UI showing confidence percentages, hold timers, and gesture legends.

## 🛠️ Tech Stack
- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning:** Scikit-Learn (SVM Classifier with RBF Kernel)
- **Text-to-Speech:** pyttsx3
- **Data Processing:** Pandas, NumPy

## 🎮 How to Run Locally
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `python main_speech.py`

## 🧠 How it Works
1. **Data Collection:** `collect_data.py` uses MediaPipe to extract 21 3D hand landmarks and 10 joint angles.
2. **Training:** `train_model.py` trains a Support Vector Machine to classify 6 distinct gestures.
3. **Inference:** `main_speech.py` uses the webcam to detect gestures in real-time, builds a sentence, and speaks it using Windows SAPI5.