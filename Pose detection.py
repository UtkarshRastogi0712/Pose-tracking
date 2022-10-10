import cv2 as cv
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cam = cv.VideoCapture(0)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cam.isOpened():
    success, image = cam.read()
    if not success:
        continue
    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = pose.process(image)

    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_style.get_default_pose_landmarks_style())
    cv.imshow("Pose Track", cv.flip(image,1))
    if cv.waitKey(5) & 0xFF == 27:
      break
cam.release()