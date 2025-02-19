import cv2
from retinaface import RetinaFace
import time

start = time.time()

cap = cv2.VideoCapture('testVideo.mp4')

prev_frame_time = 0

new_frame_time = 0

fps_array = []
time_detection_array = []

width = int(cap.get(3))
height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('OutRetinaFace.avi', fourcc, 30.0, (width,  height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    start_detection = time.time()
    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(frame)

    for key, face in faces.items():
        facial_area = face['facial_area']
        cv2.rectangle(frame, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (255, 0, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps_array.append(fps)
    fps = str(fps)
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    end = time.time()

    time_detection_array.append( end - start_detection )

    out.write(frame)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # if end - start > 100:
out.release()
cap.release()
cv2.destroyAllWindows()
print("fps average", sum(fps_array) / len(fps_array))
print("time detection average", sum(time_detection_array) / len(time_detection_array))
exit()


# cap.release()
# cv2.destroyAllWindows()