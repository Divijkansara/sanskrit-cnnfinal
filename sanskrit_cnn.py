"""
Sanskrit Character Recognition using CNN
-----------------------------------------
Dataset : Sanskrit Letter Dataset (DevDigitizer Project)
          https://github.com/avadesh02/Sanskrit-letter-dataset
          7702 images of Sanskrit letters segmented from real documents.

We use a small subset: 20 letter classes, 25 images each = 500 images.
"""

import pickle
import collections
import numpy as np
import cv2


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

np.random.seed(42)
tf.random.set_seed(42)

# ------------------------- Settings -------------------------
NUM_CLASSES     = 20
IMAGES_PER_CLASS = 25
IMG_SIZE        = 32
EPOCHS          = 30

# ------------------------- Load dataset ---------------------
db = pickle.load(open("dev_letter_D.p", "rb"), encoding="latin1")
print("Total images in dataset:", len(db))

counts = collections.Counter([item[2] for item in db])
all_letters = [c for c, n in counts.most_common() if c.isalpha()]
selected_classes = all_letters[:NUM_CLASSES]
print("Selected Sanskrit letters:", selected_classes)

X = []
Y = []
per_class = collections.Counter()

for img, class_idx, label in db:
    if label in selected_classes and per_class[label] < IMAGES_PER_CLASS:
        per_class[label] += 1
        img = np.array(img, dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        X.append(gray)
        Y.append(label)

X = np.array(X)
Y = np.array(Y)
print("Subset shape:", X.shape)

# ------------------------- Train/Test split -----------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)

X_train = X_train / 255.0
X_test  = X_test / 255.0
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

le = LabelEncoder()
Y_train_enc = le.fit_transform(Y_train)
Y_test_enc  = le.transform(Y_test)
Y_train_cat = to_categorical(Y_train_enc, NUM_CLASSES)
class_names = le.classes_

print("Training:", X_train.shape, " Testing:", X_test.shape)

# ------------------------- CNN Model ------------------------
model = Sequential([
    Input((IMG_SIZE, IMG_SIZE, 1)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(NUM_CLASSES, activation='softmax')
])
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train_cat,
                    epochs=EPOCHS, batch_size=32,
                    validation_split=0.2, verbose=2)

# ------------------------- Evaluation -----------------------
Y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\nTest Accuracy:", accuracy_score(Y_test_enc, Y_pred))
print("\nClassification Report:")
print(classification_report(Y_test_enc, Y_pred,
                            target_names=class_names, zero_division=0))

cm = confusion_matrix(Y_test_enc, Y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig("confusion_matrix.png"); plt.close()

plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.title("Training vs Validation Accuracy")
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title("Training vs Validation Loss")
plt.tight_layout(); plt.savefig("training_curves.png"); plt.close()

plt.figure(figsize=(12, 6))
for i in range(12):
    idx = np.random.randint(0, len(X_test))
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_test[idx].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    actual = class_names[Y_test_enc[idx]]
    pred = class_names[Y_pred[idx]]
    plt.title("A: " + actual + "\nP: " + pred, fontsize=8,
              color='green' if actual == pred else 'red')
    plt.axis('off')
plt.tight_layout(); plt.savefig("sample_predictions.png"); plt.close()

cv2.imwrite("test_char.png", (X_test[0].reshape(IMG_SIZE, IMG_SIZE) * 255).astype(np.uint8))
img = cv2.imread("test_char.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
p = model.predict(img.reshape(1, IMG_SIZE, IMG_SIZE, 1), verbose=0)
print("Single image -> Predicted:", class_names[np.argmax(p)],
      "Actual:", class_names[Y_test_enc[0]], "Confidence:", round(float(np.max(p)), 3))

model.save("sanskrit_cnn_model.h5")
print("Done.")
