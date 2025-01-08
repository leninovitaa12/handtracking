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

# Warna untuk setiap jari
finger_colors = [
    (0, 0, 255),    # Jempol (merah)
    (0, 255, 255),  # Telunjuk (kuning)
    (255, 0, 0),    # Tengah (biru)
    (0, 0, 255),    # Manis (merah)
    (255, 0, 255)   # Kelingking (ungu)
]

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
            for hand_landmarks in results.multi_hand_landmarks:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])

                    # Warna untuk setiap jari
                    if i in range(0, 4+1):
                        color = finger_colors[0]  # Jempol
                    elif i in range(5, 8+1):
                        color = finger_colors[1]  # Telunjuk
                    elif i in range(9, 12+1):
                        color = finger_colors[2]  # Tengah
                    elif i in range(13, 16+1):
                        color = finger_colors[3]  # Manis
                    elif i in range(17, 20+1):
                        color = finger_colors[4]  # Kelingking

                    cv2.circle(image, (x, y), 4, color, -1)

                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        # Tampilkan frame dengan landmark tangan
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # Simpan gambar
        cv2.imwrite(os.path.join('Output', '{}.jpg'.format(uuid.uuid1())), image)

cap.release()
cv2.destroyAllWindows()
