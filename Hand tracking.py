import cv2 as cv
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cam = cv.VideoCapture(0)
hand = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cam.isOpened():
    success, image = cam.read()
    if not success:
        continue
    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hand.process(image)

    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_style.get_default_hand_landmarks_style(), mp_style.get_default_hand_connections_style())
        cv.imshow("Hand Track", cv.flip(image,1))
        if cv.waitKey(5) & 0xFF == 27:
            break
cam.release()