from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np
import joblib
import cvzone
import math
from ultralytics import YOLO

app = Flask(__name__)

svm = joblib.load('svm_model.pkl')
pca = joblib.load('pca_transform.pkl')
scaler = joblib.load('scaler.pkl')
names = joblib.load('names.pkl')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_COMPLEX

helmet_model = YOLO('weights/hemletYoloV8_100epochs.pt')
helmet_classNames = ['Without Helmet', 'With Helmet']

def generate_face_mask_frames():
    capture = cv2.VideoCapture(0)
    while True:
        flag, img = capture.read()
        if flag:
            faces = face_cascade.detectMultiScale(img)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = img[y:y + h, x:x + w, :]
                face = cv2.resize(face, (50, 50))
                face = face.reshape(1, -1)
                face_scaled = scaler.transform(face)
                face_pca = pca.transform(face_scaled)
                pred = svm.predict(face_pca)
                n = names[int(pred)]
                cv2.putText(img, n, (x, y), font, 1, (244, 250, 250), 2)
            ret, buffer = cv2.imencode('.jpg', img)
            img_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
        else:
            break
    capture.release()
    cv2.destroyAllWindows()
def generate_helmet_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break

        results = helmet_model(img)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                if cls < len(helmet_classNames):
                    cvzone.putTextRect(img, f'{helmet_classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2, colorR=255, colorB=0, colorT=0)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def landing_page():
    return render_template('frontpage.html')

@app.route('/index')

def index():
    return render_template('index.html')

@app.route('/webcam_detect')
def webcam_detect():
    return render_template('webcam_detect.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_face_mask_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate_helmet_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
