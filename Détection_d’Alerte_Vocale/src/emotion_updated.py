import os
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# Dataset path
DATASET_PATH = "data/RAVDESS/"

emotions_dict = {
    "01": "non-alert", "02": "non-alert", "03": "non-alert",
    "04": "alert", "05": "alert", "06": "alert",
    "07": "alert", "08": "non-alert"
}

# Feature extraction function
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)

    combined = np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(delta.T, axis=0),
        np.mean(delta2.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(contrast.T, axis=0),
        np.mean(zcr.T, axis=0)
    ])
    return combined

# Data loading
features_list = []
labels_list = []

for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split('-')[2]
            label = emotions_dict.get(emotion_code)
            if label:
                features = extract_features(os.path.join(root, file))
                features_list.append(features)
                labels_list.append(label)

X = np.array(features_list)
y = np.array(labels_list)

# Label encoding
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Normalize features
mean = np.mean(X, axis=0)
std = np.std(X, axis=0) + 1e-6
X = (X - mean) / std

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.25, random_state=42)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Class weights
class_weights_arr = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights = dict(enumerate(class_weights_arr))

# Model definition
def build_attention_model(input_shape=(X_train.shape[1], 1)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, 5, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Squeeze-Excitation block
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(16, activation='relu')(se)
    se = layers.Dense(32, activation='sigmoid')(se)
    x = layers.Multiply()([x, layers.Reshape((1, 32))(se)])

    x = layers.Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)

    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-4,
        first_decay_steps=500,
        t_mul=2.0,
        m_mul=0.8
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_attention_model()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_emotion_model.h5', monitor='val_loss', save_best_only=True)

# Training
history = model.fit(
    X_train, y_train,
    epochs=190,
    batch_size=8,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluation
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTrain Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Train Loss: {train_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Training plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy over epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss over epochs')
plt.legend()
plt.show()

# Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Predict new audio
def predict_emotion(file_path):
    features = extract_features(file_path)
    features = (features - mean) / std
    features = features.reshape(1, -1, 1)
    prediction = model.predict(features)
    emotion_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    emotion = encoder.inverse_transform([emotion_idx])[0]
    print(f"\nFile: {os.path.basename(file_path)}")
    print(f"Predicted Emotion: {emotion} (Confidence: {confidence*100:.1f}%)")

# Example usage
predict_emotion("data/test/hap.wav")
