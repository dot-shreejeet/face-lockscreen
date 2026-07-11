import os
import json

CAMERA = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

DATASET_DIR = 'dataset'
TRAINED_MODEL_DIR = 'trainer'
MODEL = os.path.join(TRAINED_MODEL_DIR, 'trainer.yml')

LOCK_LIMIT = 30
UNLOCK_LIMIT = 15
CONFIDENCE_THRESHOLD = 60

USER_FILE = 'users.json'

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r') as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}