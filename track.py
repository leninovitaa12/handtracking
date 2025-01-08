import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Membuat direktori 'Output' jika belum ada
if not os.path.exists('Output'):
    os.makedirs('Output')

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi frame dari BGR ke RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip horizontal
        image = cv2.flip(image, 1)

        # Tandai frame sebagai tidak bisa ditulis untuk meningkatkan kinerja
        image.flags.writeable = False

        # Proses deteksi tangan
        results = hands.process(image)

        # Tandai frame sebagai bisa ditulis kembali
        image.flags.writeable = True

        # Konversi kembali frame dari RGB ke BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # RENDERING
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image, hand, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(12, 44, 250), thickness=2, circle_radius=2))

        # Tampilkan frame dengan landmark tangan
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # Simpan gambar
        cv2.imwrite(
            os.path.join('Output', '{}.jpg'.format(uuid.uuid1())),
            image
        )

cap.release()
cv2.destroyAllWindows()

