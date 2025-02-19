import cv2
import mediapipe as mp
import time

start = time.time()

mp_face_detection = mp.solutions.face_detection.FaceDetection()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture('testVideo.mp4')

prev_frame_time = 0

new_frame_time = 0

fps_array = []
time_detection_array = []

width = int(cap.get(3))
height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('OutMediaPipe.avi', fourcc, 30.0, (width,  height))


while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    start_detection = time.time()

    #face detection with mediapipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(imgRGB)

    #draw face detection
    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(imgBGR, detection)

    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps_array.append(fps)
    fps = str(fps)
    cv2.putText(imgBGR, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    end = time.time()
    time_detection_array.append( end - start_detection )

    out.write(imgBGR)
    cv2.imshow('Video', imgBGR)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # if end - start > 10:
out.release()
cap.release()
cv2.destroyAllWindows()
print("fps average", sum(fps_array) / len(fps_array))
print("time detection average", sum(time_detection_array) / len(time_detection_array))
exit()