import streamlit as st
import cv2
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Assuming your existing Python modules (video_processing.py, face_detection.py,
# feature_extraction.py, classification.py) are in the same directory or
# accessible in your Python path. If not, adjust the imports accordingly.
from src.video_processing import load_video, extract_frames
from src.face_detection import FaceDetector
from src.feature_extraction import FeatureExtractor
from src.classification import DeepfakeClassifier

# Set device for model inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pre-trained models (load them once outside the prediction function for efficiency)
face_detector = FaceDetector(device=device)
feature_extractor = FeatureExtractor(device=device)
deepfake_classifier = DeepfakeClassifier()

def predict_deepfake_probability(video_file):
    """
    Predicts the probability of a video being a deepfake.

    Args:
        video_file: Uploaded video file object.

    Returns:
        float: Probability of the video being a deepfake.
    """
    try:
        # Save the uploaded video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())
        video_path = "temp_video.mp4"

        cap = load_video(video_path)
        frames = extract_frames(cap, frame_interval=5)  # Adjust frame interval as needed

        faces = face_detector.detect_faces(frames)
        embeddings = feature_extractor.extract_features(faces) if faces else None

        probability = deepfake_classifier.predict_deepfake(embeddings)

        return probability

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        return None
    finally:
        import os
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")

# Streamlit App
st.title("Deepfake Video Detector")
st.markdown("Upload a video to check if it's likely a deepfake.")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)  # Display the uploaded video

    if st.button("Analyze Video"):
        with st.spinner("Analyzing video..."):
            probability = predict_deepfake_probability(uploaded_file)

        if probability is not None:
            st.subheader("Prediction:")
            if probability > 0.5:
                st.error(f"Likely **FAKE** (Probability: {probability:.4f})")
            else:
                st.success(f"Likely **REAL** (Probability: {1 - probability:.4f})")
            st.write(f"Probability of being FAKE: **{probability * 100:.2f}%**")
            st.write(f"Probability of being REAL: **{(1 - probability) * 100:.2f}%**")