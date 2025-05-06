import cv2
from deepface import DeepFace

# Open the webcam
cap = cv2.VideoCapture(0)
print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze the frame for emotions
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Get the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Display it on the screen
        cv2.putText(frame, f'Emotion: {emotion}', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("Error:", e)

    # Show the video
    cv2.imshow("Emotion Detection", frame)

    # Break the loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
