import os
import glob
import time
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
import streamlit as st

# Define device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

# Load face detector
mtcnn = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()

# Load facial recognition model
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

# Define DetectionPipeline class
class DetectionPipeline:
    def __init__(self, detector, n_frames=None, batch_size=60, resize=None):
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize

    def __call__(self, filename):
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        faces = []
        frames = []
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                frames.append(frame)

                if len(frames) % self.batch_size == 0 or j == sample[-1]:
                    faces.extend(self.detector(frames))
                    frames = []

        v_cap.release()
        return faces

# Process embeddings
def process_faces(faces, resnet):
    faces = [f for f in faces if f is not None]
    faces = torch.cat(faces).to(device)

    embeddings = resnet(faces)
    centroid = embeddings.mean(dim=0)
    distances = (embeddings - centroid).norm(dim=1).cpu().numpy()
    
    return distances

# Load video filenames
video_dir = 'test_videos/'  # Replace with your video directory
filenames = glob.glob(os.path.join(video_dir, '*.mp4'))

# Initialize pipeline and process videos
detection_pipeline = DetectionPipeline(detector=mtcnn, batch_size=60, resize=0.25)
X = []
start = time.time()
n_processed = 0

with torch.no_grad():
    for filename in tqdm(filenames):
        try:
            faces = detection_pipeline(filename)
            X.append(process_faces(faces, resnet))
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            X.append(None)
        
        n_processed += len(faces)
        print(f'Frames per second: {n_processed / (time.time() - start):6.3}', end='\r')

# Placeholder for predictions
bias = -0.2942
weight = 0.68235746
submission = []

for filename, x_i in zip(filenames, X):
    if x_i is not None:
        prob = 1 / (1 + np.exp(-(bias + (weight * x_i).mean())))
    else:
        prob = 0.5
    submission.append([os.path.basename(filename), prob])

# Save results
submission = pd.DataFrame(submission, columns=['filename', 'label'])
submission.sort_values('filename').to_csv('submission.csv', index=False)

# Plot histogram
plt.hist(submission.label, 20)
st.pyplot(plt)
