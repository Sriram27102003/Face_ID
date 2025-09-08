import os
import time
import cv2
import numpy as np
import face_recognition
from collections import deque, Counter

ENC_FILE  = "encodings.npz"      # persistent store for face encodings + labels
FACES_DIR = "known_faces"        # where cropped enrolled faces are saved
os.makedirs(FACES_DIR, exist_ok=True)

# ---------- persistence helpers ----------
def load_encodings():
    if not os.path.exists(ENC_FILE):
        return [], []
    data = np.load(ENC_FILE, allow_pickle=True)
    encs = data["encs"].tolist()
    labels = data["labels"].tolist()
    return encs, labels

def save_encodings(encs, labels):
    np.savez(ENC_FILE, encs=np.array(encs, dtype=object), labels=np.array(labels, dtype=object))

KNOWN_ENCODINGS, KNOWN_LABELS = load_encodings()
print(f"[init] loaded {len(KNOWN_LABELS)} face(s) from {ENC_FILE}")

# ---------- simple temporal smoothing to reduce flicker ----------
SMOOTH_N = 5
name_history = deque(maxlen=SMOOTH_N)
def smooth(name):
    name_history.append(name)
    return Counter(name_history).most_common(1)[0][0]

# ---------- main loop ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("No webcam found.")

print("[keys] ESC: quit | E: enroll current face | R: reload encodings")

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

    frame_bgr = cv2.flip(frame_bgr, 1)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # detect faces
    boxes = face_recognition.face_locations(rgb, model="hog")  # fast; use "cnn" if you have GPU
    encs  = face_recognition.face_encodings(rgb, boxes) if boxes else []

    names = []
    for enc in encs:
        name = "Unknown"
        if KNOWN_ENCODINGS:
            matches = face_recognition.compare_faces(KNOWN_ENCODINGS, enc, tolerance=0.45)
            if True in matches:
                # choose the closest match by distance
                dists = face_recognition.face_distance(KNOWN_ENCODINGS, enc)
                idx = int(np.argmin(dists))
                if matches[idx]:
                    name = KNOWN_LABELS[idx]
        names.append(smooth(name))

    # draw boxes + labels
    for (top, right, bottom, left), name in zip(boxes, names):
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 255), 2)
        bar_h = 26
        cv2.rectangle(frame_bgr, (left, bottom), (right, bottom + bar_h), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame_bgr, name, (left + 6, bottom + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # HUD
    h, w = frame_bgr.shape[:2]
    cv2.rectangle(frame_bgr, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame_bgr, "E: Enroll  |  R: Reload  |  ESC: Quit",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Face Enrollment + Recognition", frame_bgr)
    k = cv2.waitKey(1) & 0xFF

    if k == 27:  # ESC
        break

    elif k in (ord('e'), ord('E')):
        # enroll only if exactly one face is visible
        if len(boxes) != 1 or len(encs) != 1:
            print("[enroll] please show exactly ONE clear face to the camera.")
            continue

        name = input("Enter name for this face: ").strip()
        if not name:
            print("[enroll] name cannot be empty.")
            continue

        # crop and save face image
        top, right, bottom, left = boxes[0]
        face_crop = frame_bgr[max(0, top):max(0, bottom), max(0, left):max(0, right)]
        ts = int(time.time())
        out_path = os.path.join(FACES_DIR, f"{name}_{ts}.jpg")
        cv2.imwrite(out_path, face_crop)
        print(f"[enroll] saved {out_path}")

        # append encoding + label, persist, and update in-memory
        KNOWN_ENCODINGS.append(encs[0])
        KNOWN_LABELS.append(name)
        save_encodings(KNOWN_ENCODINGS, KNOWN_LABELS)
        print(f"[enroll] stored encoding. total known faces: {len(KNOWN_LABELS)}")

    elif k in (ord('r'), ord('R')):
        KNOWN_ENCODINGS, KNOWN_LABELS = load_encodings()
        print(f"[reload] loaded {len(KNOWN_LABELS)} face(s) from {ENC_FILE}")

cap.release()
cv2.destroyAllWindows()

