import os
import pickle
import cv2
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Define a consistent image size
IMAGE_SIZE = (64, 64)

# Extract HOG features from a single image
def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    # Convert to grayscale if the image has multiple channels
    if len(image.shape) > 2:
        image = rgb2gray(image)
    features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block, block_norm='L2-Hys', transform_sqrt=True)
    return features

# Extract features from all images in a directory
def extract_features_from_dir(directory):
    features = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        image = cv2.imread(filepath)
        if image is not None:
            # Resize to ensure consistent dimensions
            image = cv2.resize(image, IMAGE_SIZE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure correct color format
            features.append(extract_hog_features(image))
    return features

# Train the SVM model
def train_svm():
    # Directories for vehicle and non-vehicle images
    vehicle_dir = './data/vehicles/'
    non_vehicle_dir = './data/non-vehicles/'

    # Extract features from both vehicle and non-vehicle datasets
    vehicle_features = extract_features_from_dir(vehicle_dir)
    non_vehicle_features = extract_features_from_dir(non_vehicle_dir)

    # Check dimensions of extracted features
    print(f"Vehicle features shape: {np.array(vehicle_features).shape}")
    print(f"Non-vehicle features shape: {np.array(non_vehicle_features).shape}")

    # Create labels for the datasets
    X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
    y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

    # Normalize the feature vector
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the Linear SVM
    clf = LinearSVC(max_iter=10000)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    # Save the model and scaler
    with open('vehicle_classifier.pkl', 'wb') as model_file:
        pickle.dump({'scaler': scaler, 'classifier': clf}, model_file)
    print("Model and scaler saved to 'vehicle_classifier.pkl'")

# Main script execution
if __name__ == '__main__':
    train_svm()
