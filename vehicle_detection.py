import pickle
import cv2
import numpy as np
from utils import slide_window, extract_hog_features, apply_threshold

# Load the trained model and scaler
with open('vehicle_classifier.pkl', 'rb') as model_file:
    data = pickle.load(model_file)
    scaler = data['scaler']
    classifier = data['classifier']

# Parameters for sliding window search
window_params = {
    'x_start_stop': [None, None],  # Min and max in x
    'y_start_stop': [400, 656],   # Min and max in y
    'xy_window': (96, 96),        # Size of the window
    'xy_overlap': (0.5, 0.5)      # Overlap fraction
}

def process_frame(frame):
    draw_img = np.copy(frame)
    windows = slide_window(frame, **window_params)
    hot_windows = []

    for window in windows:
        test_img = cv2.resize(frame[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = extract_hog_features(test_img)
        features = scaler.transform(features.reshape(1, -1))
        prediction = classifier.predict(features)
        if prediction == 1:
            hot_windows.append(window)

    for window in hot_windows:
        cv2.rectangle(draw_img, window[0], window[1], (0, 0, 255), 6)
    return draw_img

# Video processing
input_video = './videos/input_video.mp4'
output_video = './videos/output_video.mp4'

cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    processed_frame = process_frame(frame)
    out.write(processed_frame)

cap.release()
out.release()
print("Vehicle detection completed. Output saved to", output_video)
