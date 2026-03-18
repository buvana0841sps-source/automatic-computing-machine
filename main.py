import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_images(dataset_path):
    images, labels = [], []
    label = 0
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100,100))
            images.append(img.flatten())
            labels.append(label)
        label += 1
    return np.array(images), np.array(labels)

def PCA(X, k):
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face
    C = np.dot(X_centered, X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenfaces = np.dot(X_centered.T, eigenvectors)
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)
    eigenfaces = eigenfaces[:, :k]
    X_pca = np.dot(X_centered, eigenfaces)
    return X_pca, mean_face, eigenfaces

dataset_path = "dataset"
X, y = load_images(dataset_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

k = 30
X_train_pca, mean_face, eigenfaces = PCA(X_train, k)

X_test_centered = X_test - mean_face
X_test_pca = np.dot(X_test_centered, eigenfaces)

ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
ann.fit(X_train_pca, y_train)

y_pred = ann.predict(X_test_pca)
print("Accuracy:", accuracy_score(y_test, y_pred))
