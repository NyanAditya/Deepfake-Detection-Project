from facenet_pytorch import MTCNN
from PIL import Image

class FaceDetector:
    def __init__(self, device='cpu'):
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

    def detect_faces(self, frames):
        """Detects faces in a list of frames."""
        batch_boxes, _, batch_landmarks = self.mtcnn.detect(frames, landmarks=True)
        detected_faces = []
        for i, boxes in enumerate(batch_boxes):
            if boxes is not None:
                for box in boxes:
                    face_img = frames[i][int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    if face_img.size > 0: # Check if the cropped image is valid
                        detected_faces.append(Image.fromarray(face_img))
        return detected_faces

if __name__ == '__main__':
    from video_processing import load_video, extract_frames
    video_path = 'path/to/your/video.mp4'
    cap = load_video(video_path)
    frames = extract_frames(cap, frame_interval=10)

    detector = FaceDetector(device='cpu') # Or 'cuda' if you have a GPU
    faces = detector.detect_faces(frames)
    print(f"Detected {len(faces)} faces.")
    # You can save or display the detected faces here