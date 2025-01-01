import os
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""

    def __init__(self, detector, n_frames=None, batch_size=60, resize=None):
        """Constructor for DetectionPipeline class.

        Keyword Arguments:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {60})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 results in upsampling. (default: {None})
        """
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize

    def __call__(self, filename):
        """Load frames from an MP4 video and detect faces.

        Arguments:
            filename {str} -- Path to video.
        """
        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        faces = []
        frames = []
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                frames.append(frame)

                # When batch is full, detect faces and reset frame list
                if len(frames) % self.batch_size == 0 or j == sample[-1]:
                    faces.extend(self.detector(frames))
                    frames = []

        v_cap.release()

        return faces

def process_faces(faces, resnet, device):
    """Process faces to calculate embeddings and distances.

    Arguments:
        faces {list} -- List of face tensors.
        resnet {InceptionResnetV1} -- Pretrained face recognition model.
        device {str} -- Device to run the computation on (e.g., 'cpu' or 'cuda').

    Returns:
        numpy.ndarray -- Array of distances from centroid for each face.
    """
    # Filter out frames without faces
    faces = [f for f in faces if f is not None]
    if not faces:
        return np.array([])  # Return empty array if no faces detected

    faces = torch.cat(faces).to(device)

    # Generate facial feature vectors using a pretrained model
    embeddings = resnet(faces)

    # Calculate centroid for video and distance of each face's feature vector from centroid
    centroid = embeddings.mean(dim=0)
    distances = (embeddings - centroid).norm(dim=1).cpu().numpy()

    return distances

def get_models(device):
    """Initialize and return MTCNN and InceptionResnetV1 models.

    Arguments:
        device {str} -- Device to run the models on (e.g., 'cpu' or 'cuda').

    Returns:
        tuple -- Tuple containing MTCNN and InceptionResnetV1 models.
    """
    # Load face detector
    mtcnn = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()

    # Load facial recognition model
    resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

    return mtcnn, resnet
