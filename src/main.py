import argparse
import torch
from video_processing import load_video, extract_frames
from face_detection import FaceDetector
from feature_extraction import FeatureExtractor
from classification import DeepfakeClassifier

def main():
    parser = argparse.ArgumentParser(description="Deepfake detection using pre-trained models.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    args = parser.parse_args()

    video_path = args.video_path

    cap = load_video(video_path)
    frames = extract_frames(cap, frame_interval=5)  # Adjust frame interval as needed

    detector = FaceDetector(device='cuda' if torch.cuda.is_available() else 'cpu')
    faces = detector.detect_faces(frames)

    extractor = FeatureExtractor(device='cuda' if torch.cuda.is_available() else 'cpu')
    embeddings = extractor.extract_features(faces) if faces else None

    classifier = DeepfakeClassifier()
    probability = classifier.predict_deepfake(embeddings)

    print(f"Video: {video_path}")
    print(f"Deepfake Probability: {probability:.4f}")
    print(f"Prediction: {'FAKE' if probability > 0.5 else 'REAL'} (Confidence: {probability:.4f} if FAKE, {1-probability:.4f} if REAL)")

if __name__ == "__main__":
    main()