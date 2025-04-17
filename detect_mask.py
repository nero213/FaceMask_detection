import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("mask_detector.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]

        # Preprocess face image
        resized = cv2.resize(face_roi, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))

        # Get prediction and confidence
        pred = model.predict(reshaped, verbose=0)
        confidence = np.max(pred) * 100  # Convert to percentage
        predicted_class = np.argmax(pred)

        # Determine label and color
        label = "Mask" if predicted_class == 0 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Display label with confidence percentage
        display_text = f"{label} {confidence:.1f}%"
        cv2.putText(frame, display_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Debug output
        print(f"Prediction: {pred[0]} | Class: {predicted_class} | Confidence: {confidence:.1f}%")

    cv2.imshow("Mask Detection with Confidence", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()