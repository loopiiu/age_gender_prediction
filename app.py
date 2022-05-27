from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename

import time
import cv2
from PIL import Image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_face_box(net, frame, conf_threshold=0.7):
    opencv_dnn_frame = frame.copy()
    frame_height = opencv_dnn_frame.shape[0]
    frame_width = opencv_dnn_frame.shape[1]
    blob_img = cv2.dnn.blobFromImage(opencv_dnn_frame, 1.0, (300, 300), [
        104, 117, 123], True, False)

    net.setInput(blob_img)
    detections = net.forward()
    b_boxes_detect = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            b_boxes_detect.append([x1, y1, x2, y2])
            cv2.rectangle(opencv_dnn_frame, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frame_height / 150)), 8)
    return opencv_dnn_frame, b_boxes_detect





@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        path = os.path.abspath(file.filename)
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below')
        print("P:", path)
        # file_to_string(filetosave.file.path)

        image = Image.open('static/uploads/'+filename)
        # print(image)
        cap = np.array(image)

        cv2.imwrite('temp.jpg', cv2.cvtColor(cap, cv2.COLOR_BGR2BGRA))
        cap = cv2.imread("temp.jpg")

        face_txt_path = "opencv_face_detector.pbtxt"
        face_model_path = "opencv_face_detector_uint8.pb"

        age_txt_path = "age_deploy.prototxt"
        age_model_path = "age_net.caffemodel"

        gender_txt_path = "gender_deploy.prototxt"
        gender_model_path = "gender_net.caffemodel"

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        age_classes = ['Age: ~1-2', 'Age: ~3-5', 'Age: ~6-14', 'Age: ~16-22',
                       'Age: ~25-30', 'Age: ~32-40', 'Age: ~45-50', 'Age: age is greater than 60']
        gender_classes = ['Gender:Male', 'Gender:Female']

        age_net = cv2.dnn.readNet(age_model_path, age_txt_path)
        gender_net = cv2.dnn.readNet(gender_model_path, gender_txt_path)
        face_net = cv2.dnn.readNet(face_model_path, face_txt_path)

        padding = 20

        t = time.time()
        preds=[]
        frameFace, b_boxes = get_face_box(face_net, cap)
        if not b_boxes:
            flash("No face Detected, Checking next frame")

        for bbox in b_boxes:
            face = cap[max(0, bbox[1] - padding): min(bbox[3] + padding, cap.shape[0] - 1),
                   max(0, bbox[0] - padding): min(bbox[2] + padding, cap.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender_pred_list = gender_net.forward()
            gender = gender_classes[gender_pred_list[0].argmax()]


            age_net.setInput(blob)
            age_pred_list = age_net.forward()
            age = age_classes[age_pred_list[0].argmax()]

            label = "{},{}".format(gender, age)
            cv2.putText( frameFace, label, (bbox[0], bbox[1] - 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            preds.append(frameFace)

        img = Image.fromarray(preds[-1], 'RGB')
        # print(img)
        # img.save('my.png')
        filename = 'res'+filename
        img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()