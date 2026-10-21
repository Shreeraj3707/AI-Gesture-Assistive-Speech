import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import time
import math
from collections import deque, Counter

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

def format_sentence(words):
    if not words:
        return ""
    
    # Special combinations
    if "help" in words and "water" in words:
        return "Please help me get some water."
    if "help" in words and "food" in words:
        return "Please help me get some food."
    if "help" in words and "bathroom" in words:
        return "Please help me get to the bathroom."
    if "pain" in words:
        return "I am in pain, please help."
    
    # Single words
    if words == ["help"]: return "Please help me."
    if words == ["water"]: return "I am thirsty, I need water."
    if words == ["food"]: return "I am hungry, I need food."
    if words == ["yes"]: return "Yes, I agree."
    if words == ["bathroom"]: return "I need to use the bathroom."
    
    # Multiple words
    if "yes" in words and len(words) > 1:
        return "Yes, I need " + " and ".join([w for w in words if w != "yes"]) + "."
    
    return "I need " + " and ".join(words) + "."

# ===== CONFIG =====
CONFIDENCE_THRESHOLD = 0.75   
STABLE_FRAMES = 15            
HOLD_SPEAK_SECONDS = 3.0      
EMERGENCY_SECONDS = 5.0       
COOLDOWN_AFTER_ADD = 1.0      
# ==================

engine = pyttsx3.init()
engine.setProperty('rate', 150)

with open('gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

recent = deque(maxlen=5)
sentence = []
last_added_time = 0
stable_gesture = None
stable_since = None
last_spoken = None
emergency_triggered = False

print("Ready. 6 Gestures Active: help, water, food, yes, pain, bathroom.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_gesture = None
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            features = np.array(extract_features(hand_landmarks)).reshape(1, -1)
            proba = model.predict_proba(features)[0]
            confidence = float(max(proba))
            prediction = model.classes_[proba.argmax()]

            if confidence > CONFIDENCE_THRESHOLD:
                recent.append(prediction)
                if recent:
                    current_gesture = Counter(recent).most_common(1)[0][0]

    now = time.time()
    
    if current_gesture:
        if current_gesture != stable_gesture:
            stable_gesture = current_gesture
            stable_since = now
            emergency_triggered = False
        held = now - stable_since

        if held >= (STABLE_FRAMES / 30.0) and (now - last_added_time) > COOLDOWN_AFTER_ADD:
            sentence.append(stable_gesture)
            last_added_time = now

        if held >= HOLD_SPEAK_SECONDS and held < EMERGENCY_SECONDS and sentence:
            final_text = format_sentence(sentence)
            if last_spoken != final_text:
                engine.say(final_text)
                engine.runAndWait()
                last_spoken = final_text
                
        if held >= EMERGENCY_SECONDS and not emergency_triggered:
            emergency_triggered = True
            engine.say("Emergency! I need immediate assistance.")
            engine.runAndWait()

    else:
        stable_gesture = None
        stable_since = None
        emergency_triggered = False

    # ===== DRAW UI =====
    h, w = frame.shape[:2]
    
    if emergency_triggered:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 20)
        cv2.putText(frame, "EMERGENCY!", (w//2 - 150, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    if current_gesture:
        cv2.putText(frame, f"Gesture: {current_gesture} ({confidence*100:.0f}%)",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if stable_since:
            held = now - stable_since
            timer_color = (0, 255, 255)
            if held > HOLD_SPEAK_SECONDS: timer_color = (0, 165, 255)
            if held > 4.0: timer_color = (0, 0, 255)
            cv2.putText(frame, f"Hold: {held:.1f}s", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)
    else:
        cv2.putText(frame, "Gesture: ...", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    # Legend (Top Right)
    cv2.putText(frame, "✋: help   ✊: water", (w - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "👍: food   ✌️: yes", (w - 280, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "🤏: pain   ☝️: bathroom", (w - 280, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Sentence bar
    raw_text = "Raw: " + " ".join(sentence) if sentence else "Raw: (empty)"
    smart_text = "AI: " + format_sentence(sentence) if sentence else "AI: (waiting)"
    
    cv2.rectangle(frame, (0, h - 80), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, raw_text, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, smart_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Assistive AI - 6 Gesture System', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = []
        last_spoken = None
    elif key == ord('s') and sentence:
        final_text = format_sentence(sentence)
        engine.say(final_text)
        engine.runAndWait()
        last_spoken = final_text

cap.release()
cv2.destroyAllWindows()
engine.stop()