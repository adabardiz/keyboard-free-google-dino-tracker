import cv2
import mediapipe as mp
import pyautogui
import math
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

last_jump_time = 0
JUMP_COOLDOWN = 0.6  
FINGER_JUMP_THRESHOLD = 100 
BLINK_THRESHOLD = 0.015      

def calculate_distance(p1, p2):
    """calculates the distance between two normalized landmarks"""
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

print("starting Camera... press q to quit")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    h, w, c = image.shape
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(image_rgb)
    face_results = face_mesh.process(image_rgb)

    jump_triggered = False
    current_time = time.time()

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            
            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
            
            # line between fingers
            cv2.line(image, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 3)
            
            if distance > FINGER_JUMP_THRESHOLD:
                cv2.putText(image, f"JUMP! ({int(distance)})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                jump_triggered = True
            else:
                cv2.putText(image, f"WALK ({int(distance)})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # 159 (upper) and 145 (lower) for  right eye
            upper_lid = face_landmarks.landmark[159]
            lower_lid = face_landmarks.landmark[145]
            
            eye_distance = calculate_distance(upper_lid, lower_lid)
            
            if eye_distance < BLINK_THRESHOLD:
                cv2.putText(image, "BLINK JUMP!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                jump_triggered = True

    if jump_triggered and (current_time - last_jump_time > JUMP_COOLDOWN):
        pyautogui.keyDown('space') 
        time.sleep(0.05) # Tiny delay to register the press
        pyautogui.keyUp('space')
        last_jump_time = current_time

    cv2.imshow('dino Controller', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()