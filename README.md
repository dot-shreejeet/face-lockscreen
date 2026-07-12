# Face Recognition Screen Lock

A Python-based system that uses your webcam and face recognition to lock and unlock your screen automatically. If your face isn't detected (or an unrecognized face is), the system locks itself. When you look back at the camera and are recognized, it unlocks.

Built with OpenCV's Haar Cascade for face detection and the LBPH (Local Binary Patterns Histograms) algorithm for face recognition.

## Features

- Capture and register face samples for one or more users
- Train a local face recognition model (LBPH) on captured faces
- Real-time monitoring that locks the screen when no known face is present
- Automatic unlock when a registered face is recognized again
- Simple, centralized configuration for thresholds and camera settings

## How It Works

1. **Capture** — `capture_face.py` opens your webcam, detects your face, and saves ~40 grayscale face images to a per-user folder in `dataset/`. It also records your ID → name mapping in `users.json`.
2. **Train** — `train_model.py` reads every image in `dataset/`, extracts face regions, and trains an LBPH recognizer, saving the result to `trainer/trainer.yml`.
3. **Lock** — `lock.py` continuously reads webcam frames. If no known face is seen for too long (`LOCK_LIMIT` frames), it triggers a full-screen "System Locked" overlay. It unlocks once a registered face is recognized consistently for `UNLOCK_LIMIT` frames.

## Tech Stack

- Python 3
- [OpenCV](https://opencv.org/) (`opencv-contrib-python` — needed for `cv.face.LBPHFaceRecognizer_create`)
- NumPy
- Pillow (PIL)

## Prerequisites

- Python 3.7+
- A working webcam
- `haarcascade_frontalface_default.xml` (included in this repo)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install opencv-contrib-python numpy pillow
   ```

   > **Note:** Use `opencv-contrib-python`, not plain `opencv-python` — the `cv.face` module (used for LBPH recognition) only ships with the contrib package.

## Usage

### 1. Capture face data

Run the capture script and follow the prompts:

```bash
python capture_face.py
```

You'll be asked for:
- A unique numerical ID for the user (e.g., `1`, `2`, `3`)
- The user's name

The script opens your webcam and captures 40 face images, saved under `dataset/<user_name>/`. Press `Esc` to stop early.

Repeat this step for every person you want the system to recognize.

### 2. Train the model

Once you've captured at least one user's faces:

```bash
python train_model.py
```

This processes all images in `dataset/` and saves a trained model to `trainer/trainer.yml`.

### 3. Run the lock screen

```bash
python lock.py
```

This starts real-time monitoring via your webcam. Press `Esc` to exit.

## Configuration

All tunable settings live in `config.py`:

| Setting | Description | Default |
|---|---|---|
| `CAMERA` | Webcam index | `0` |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | Capture resolution | `640` / `480` |
| `DATASET_DIR` | Folder where captured face images are stored | `dataset` |
| `TRAINED_MODEL_DIR` | Folder for the trained model | `trainer` |
| `LOCK_LIMIT` | Frames without a known face before locking | `30` |
| `UNLOCK_LIMIT` | Consecutive recognized frames needed to unlock | `15` |
| `CONFIDENCE_THRESHOLD` | Max LBPH distance to count as a recognized match (lower = stricter) | `60` |
| `USER_FILE` | JSON file mapping user IDs to names | `users.json` |

## Limitations

- This is a demo/proof-of-concept project — it does **not** lock the actual OS session (e.g., Windows/macOS/Linux login), it only shows an in-app full-screen overlay while `lock.py` is running.
- Face recognition accuracy depends heavily on lighting, camera quality, and the number/variety of training images per user.
- Not intended as a substitute for a real security/authentication mechanism.

## License

No license has been chosen yet for this project. All rights reserved by the author unless a license is added.
