import cv2 as cv
import numpy as np
import os
import config
from PIL import Image

def train_model():
    data = config.DATASET_DIR
    img_dir = [x[0] for x in os.walk(data)][1::]
    recognizer = cv.face.LBPHFaceRecognizer_create()
    detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    facesamples = []
    ids = []

    print("Train model.....")

    for path in img_dir:
        path = str(path)
        imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
        for imagepath in imagepaths:
            if os.path.basename(imagepath).startswith('.'):
                continue
            try:
                PIL_img = Image.open(imagepath).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')
            
                id_ = int(os.path.split(imagepath)[-1].split(".")[1])

                faces = detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    facesamples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(id_)
            except (IndexError, ValueError, IOError):
                continue
    if len(facesamples) == 0 or len(ids) == 0:
        print("No faces found for training.")
        return

    recognizer.train(facesamples, np.array(ids))
    os.makedirs(config.TRAINED_MODEL_DIR, exist_ok=True)
    recognizer.write(config.MODEL)
    print(f"Model trained and saved at {config.MODEL}")

if __name__ == "__main__":
    train_model()