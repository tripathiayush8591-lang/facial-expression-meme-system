import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# -------------------------------
# Load face detector
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# -------------------------------
# Load emotion model (safe mode)
# -------------------------------
emotion_model = load_model("emotion_model.h5", compile=False)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# -------------------------------
# Emotion → Meme mapping
# -------------------------------
MEME_MAP = {
    "Happy": "memes/happy.jpg",
    "Sad": "memes/sad.jpg",
    "Angry": "memes/angry.jpg",
    "Surprise": "memes/surprise.jpg",
    "Neutral": "memes/neutral.jpg",
    "Fear": "memes/neutral.jpg",
    "Disgust": "memes/neutral.jpg"
}

# -------------------------------
# Camera (Iriun = index 1)
# -------------------------------
camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# -------------------------------
# Main loop
# -------------------------------
while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]

        # Prepare face for model
        face_gray = cv2.resize(face_gray, (64, 64))
        face_gray = face_gray / 255.0
        face_gray = face_gray.reshape(1, 64, 64, 1)

        preds = emotion_model.predict(face_gray, verbose=0)
        emotion = EMOTIONS[np.argmax(preds)]

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, emotion, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

        # -------------------------------
        # SHOW MEME (ALWAYS)
        # -------------------------------
        meme_path = MEME_MAP.get(emotion)

        if meme_path and os.path.exists(meme_path):
            meme = cv2.imread(meme_path)
            meme = cv2.resize(meme, (400, 400))
            cv2.imshow("Meme Reaction 😈", meme)
            cv2.waitKey(1)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
