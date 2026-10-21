import cv2
import mediapipe as mp
import csv
import os
import math

# ===== CHANGE THIS LINE FOR EACH GESTURE =====
GESTURE_NAME = "bathroom"
# ============================================
SAMPLES_TO_COLLECT = 200

def extract_features(hand_landmarks):
    lm = hand_landmarks.landmark
    features = []
    for p in lm:
        features.extend([p.x, p.y, p.z])

    def angle(a, b, c):
        ba = (a.x - b.x, a.y - b.y, a.z - b.z)
        bc = (c.x - b.x, c.y - b.y, c.z - b.z)
        dot = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2]
        mag = math.sqrt(sum(v*v for v in ba)) * math.sqrt(sum(v*v for v in bc))
        if mag == 0:
            return 0.0
        return math.acos(max(-1.0, min(1.0, dot/mag)))

    fingers = [(0,1,2,4),(0,5,6,8),(0,9,10,12),(0,13,14,16),(0,17,18,20)]
    for base, j1, j2, tip in fingers:
        features.append(angle(lm[base], lm[j1], lm[j2]))
        features.append(angle(lm[j1], lm[j2], lm[tip]))
    return features

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

file_exists = os.path.isfile('gesture_data.csv')
csv_file = open('gesture_data.csv', mode='a', newline='')
writer = csv.writer(csv_file)

if not file_exists:
    header = []
    for i in range(21):
        header.extend([f'x{i}', f'y{i}', f'z{i}'])
    for i in range(10):
        header.append(f'angle{i}')
    header.append('label')
    writer.writerow(header)

count = 0
print(f"Recording for: {GESTURE_NAME}")
print("Hold the gesture, press 's' to save a frame. Press 'q' to quit.")

while cap.isOpened() and count < SAMPLES_TO_COLLECT:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if key == ord('s'):
                row = extract_features(hand_landmarks)
                row.append(GESTURE_NAME)
                writer.writerow(row)
                count += 1
                print(f"Saved {count}/{SAMPLES_TO_COLLECT}")

    cv2.putText(frame, f"{GESTURE_NAME}: {count}/{SAMPLES_TO_COLLECT}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Data Collection - S to Save, Q to Quit', frame)

    if key == ord('q'):
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
print(f"Done. Total saved for '{GESTURE_NAME}': {count}")