import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = []
labels = []

path = "train"

cat_count = 0
dog_count = 0

print("Loading images...")

for file in os.listdir(path):

    img_path = os.path.join(path, file)

    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.resize(img, (64, 64))
    img = img.flatten()

    if file.startswith("cat") and cat_count < 1000:
        data.append(img)
        labels.append(0)
        cat_count += 1

    elif file.startswith("dog") and dog_count < 1000:
        data.append(img)
        labels.append(1)
        dog_count += 1

    if cat_count >= 1000 and dog_count >= 1000:
        break

print("Images loaded.")

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training SVM...")

model = SVC(kernel="linear")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")