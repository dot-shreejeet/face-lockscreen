import cv2 as cv
import os
import numpy as np
import config
import json

def lockscreen():
    if not os.path.exists(config.MODEL):
        print("Trained model not found.")
        return

    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read(config.MODEL)
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    cam = cv.VideoCapture(config.CAMERA)
    cam.set(3, config.FRAME_WIDTH)
    cam.set(4, config.FRAME_HEIGHT)

    state = False
    chances = 0

    while True:
        ret, img = cam.read()
        if not ret:
            continue
        img = cv.flip(img, 1)
        h, w, _ = img.shape
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        user_detected = False

        if len(faces) == 0:
            if not state:
                chances += 1
        
        for (x, y, w_f, h_f) in faces:
            cv.rectangle(img, (x, y), (x + w_f, y + h_f), (0, 255, 0), 2)
            id_, confidence = recognizer.predict(gray[y:y + h_f, x:x + w_f])

            if confidence < config.CONFIDENCE_THRESHOLD:
                current_user = config.load_users()
                detected_name = current_user.get(id_, "Unknown")
                if detected_name != "Unknown":
                    user_detected = True
            else:
                detected_name = "Unknown"
                if not state:
                    chances += 1
                
            if not state:
                cv.rectangle(img, (x, y), (x + w_f, y + h_f), (0, 255, 0), 2)
                cv.putText(img, f"{detected_name} {round(100 - confidence)}%", (x + 5, y - 5), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if not state and chances >= config.LOCK_LIMIT:
            print("Locking the system due to multiple failed attempts.")
            state = True
            chances = 0

        if state:
            lock_screen = np.zeros((h, w, 3), dtype=np.uint8)

            cv.putText(lock_screen, "System Locked", (w // 8, h // 3), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv.namedWindow("Lock Screen", cv.WND_PROP_FULLSCREEN)
            cv.setWindowProperty("Lock Screen", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            cv.imshow("Lock Screen", lock_screen)

            try:
                cv.destroyWindow("image")
            except:
                pass
            
            if user_detected:
                chances += 1
                if chances >= config.UNLOCK_LIMIT:
                    print("Unlocking the system.")
                    state = False
                    chances = 0
                    cv.destroyWindow("Lock Screen")
            else:
                chances = 0
        else:
            cv.imshow("image", img)

        k = cv.waitKey(10) & 0xff
        if k == 27:
            break
    
    cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    lockscreen()
