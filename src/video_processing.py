import cv2

def load_video(video_path):
    """Loads a video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    return cap

def extract_frames(cap, frame_interval=1):
    """Extracts frames from a video at a specified interval."""
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB
        frame_count += 1
    cap.release()
    return frames

if __name__ == '__main__':
    video_path = 'path/to/your/video.mp4'  # Replace with your video path
    cap = load_video(video_path)
    frames = extract_frames(cap, frame_interval=10)  # Extract every 10th frame
    print(f"Extracted {len(frames)} frames.")
    # You can save or display the frames here for testing