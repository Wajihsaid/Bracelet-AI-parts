import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Dataset path
DATASET_PATH = "data/RAVDESS/"

# Emotion mapping
emotions_dict = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

# Feature extraction
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Extract features and labels
features = []
labels = []

for subdir, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(subdir, file)
            emotion_code = file.split("-")[2]
            if emotion_code in emotions_dict:
                features.append(extract_features(file_path))
                labels.append(emotions_dict[emotion_code])

# Save features to CSV
df = pd.DataFrame(features)
df["emotion"] = labels
df.to_csv("ravdess_features.csv", index=False)
print("Feature extraction complete!")

# Load and prepare data
df = pd.read_csv("ravdess_features.csv")
X = df.iloc[:, :-1].values
y = df["emotion"].values

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Reshape for Conv1D: (samples, timesteps, features)
X_train = X_train.reshape(-1, 40, 1)
X_test = X_test.reshape(-1, 40, 1)

# Build Keras model
def build_keras_emotion_cnn(input_shape=(40,1), num_classes=8):
    model = models.Sequential()
    model.add(layers.Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Instantiate model
num_classes = len(np.unique(y_encoded))
model = build_keras_emotion_cnn(num_classes=num_classes)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
# Save as .h5
model.save('emotion_cnn_keras.h5')
print("Keras model saved as 'emotion_cnn_keras.h5'!")
