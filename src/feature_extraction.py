from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms

class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.transform = transforms.ToTensor() # Convert PIL image to tensor

    def extract_features(self, face_images):
        """Extracts feature embeddings from a list of face images."""
        face_tensors = [self.transform(face).unsqueeze(0).to(self.device) for face in face_images]
        if not face_tensors:
            return None
        face_batches = torch.cat(face_tensors, dim=0)
        with torch.no_grad():
            embeddings = self.resnet(face_batches)
        return embeddings.cpu().numpy()

if __name__ == '__main__':
    from video_processing import load_video, extract_frames
    from face_detection import FaceDetector
    video_path = 'path/to/your/video.mp4'
    cap = load_video(video_path)
    frames = extract_frames(cap, frame_interval=10)

    detector = FaceDetector(device='cpu')
    faces = detector.detect_faces(frames)

    extractor = FeatureExtractor(device='cpu')
    if faces:
        features = extractor.extract_features(faces)
        print(f"Extracted features for {len(features)} faces. Shape: {features.shape}")
    else:
        print("No faces detected.")