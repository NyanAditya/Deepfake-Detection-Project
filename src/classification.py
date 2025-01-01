import numpy as np

class DeepfakeClassifier:
    def __init__(self, bias=-0.2942, weight=0.68235746):
        # Weights from the baseline logistic regression
        self.bias = bias
        self.weight = weight

    def predict_deepfake(self, embeddings):
        """Predicts the probability of a video being a deepfake based on face embeddings."""
        if embeddings is None or len(embeddings) == 0:
            return 0.5  # Default if no faces are detected

        centroid = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        avg_distance = distances.mean()

        # Use the sigmoid function with the pre-trained bias and weight
        probability = 1 / (1 + np.exp(-(self.bias + (self.weight * avg_distance))))
        return probability

if __name__ == '__main__':
    from video_processing import load_video, extract_frames
    from face_detection import FaceDetector
    from feature_extraction import FeatureExtractor

    video_path = 'path/to/your/video.mp4'
    cap = load_video(video_path)
    frames = extract_frames(cap, frame_interval=10)

    detector = FaceDetector(device='cpu')
    faces = detector.detect_faces(frames)

    extractor = FeatureExtractor(device='cpu')
    embeddings = extractor.extract_features(faces) if faces else None

    classifier = DeepfakeClassifier()
    probability = classifier.predict_deepfake(embeddings)

    print(f"Deepfake Probability: {probability:.4f}")