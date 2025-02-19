import cv2
import time

start = time.time()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('testVideo.mp4')

prev_frame_time = 0

new_frame_time = 0

fps_array = []
time_detection_array = []

width = int(cap.get(3))
height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('OutOpenCV.avi', fourcc, 30.0, (width,  height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    start_detection = time.time()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps_array.append(fps)
    fps = str(fps)
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    end = time.time()
    time_detection_array.append(end - start_detection)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # out.write(frame)
            # cv2.imshow('frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
            if key == ord('s'):
                cv2.imwrite('test1.png', frame)
            # if end - start > 10:
            #
            #     exit()
    else:

        # out.write(frame)
        # cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        if key == ord('s'):
            cv2.imwrite('test1.png', frame)
        # if end - start > 10:
        #
        #     exit()
    out.write(frame)
    cv2.imshow('frame', frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("fps average", sum(fps_array) / len(fps_array))
print("time detection average", sum(time_detection_array) / len(time_detection_array))