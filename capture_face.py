import cv2 as cv
import os
import numpy as np
import config
import json

def capture_face(user_id, user_name):
    vid = cv.VideoCapture(config.CAMERA)
    vid.set(3, config.FRAME_WIDTH)
    vid.set(4, config.FRAME_HEIGHT)

    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    path = os.path.join(config.DATASET_DIR, user_name)
    os.makedirs(path, exist_ok=True)

    count = 0
    print("Capturing face images....")
    while True:
        ret, image = vid.read()
        if not ret:
            continue
        image = cv.flip(image, 1)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1
            file_name = f"{user_name}.{user_id}.{count}.jpg"
            file_path = os.path.join(path, file_name)
            cv.imwrite(file_path, gray[y:y + h, x:x + w])
        
        cv.imshow("Capturing Face", image)
        k = cv.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 40:
            break
    
    vid.release()
    cv.destroyAllWindows()
    print("Face images captured successfully.")


if __name__ == "__main__":
    print("-------FACE CAPTURE SYSTEM--------")
    
    try:
        user_id = int(input("Enter a unique numerical ID (e.g., 1, 2, 3): "))
        user_name = input("Enter the user's name: ").strip()

        if not user_name:
            print("!! Name cannot be blank.")
        else:
            capture_face(user_id=user_id, user_name=user_name)

            user_map = {}
            if os.path.exists(config.USER_FILE):
                with open(config.USER_FILE, 'r') as f:
                    user_map = json.load(f)
            
            user_map[str(user_id)] = user_name
            with open(config.USER_FILE, 'w') as f:
                json.dump(user_map, f)
            
    except ValueError:
        print("!! ID must be a valid whole number.")