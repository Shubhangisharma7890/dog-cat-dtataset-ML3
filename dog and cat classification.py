import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Data Loading and Preprocessing (Simplified - assumes image loading)
# Replace with your actual image loading and preprocessing logic
def load_and_preprocess_images(image_paths, labels, target_size=(64, 64)):
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, target_size)
        images.append(img.flatten()) # Flatten for traditional ML
    return np.array(images), np.array(labels)

# Example: dummy data for illustration
# In a real scenario, you'd load actual dog/cat images and labels
X = np.random.rand(100, 64 * 64 * 3) # 100 images, 64x64x3 flattened
y = np.random.randint(0, 2, 100) # 0 for cat, 1 for dog

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Selection and Training (Example with SVM)
model = SVC(kernel='linear') # You can experiment with different kernels
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
