# Flask Application Script

''' This Application Script is created as a part of Major-Project of 8th Semester
MVJ College of Engineering Under Visvesvaraya Technological University
Department of Electronics and Communication Engineering
Team Name  - Team OneShot
1MJ18EC122 - Satyam Oza R
1MJ18EC123 - Shankar S
1MJ18EC126 - Shireesha D C
1MJ18EC146 - Vedashree H A'''

# Import necessary modules
import os
import cv2
import numpy as np
from time import time
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, session, redirect, url_for, flash

# Defining Path of Assets Folder
UPLOAD_FOLDER = './flask_app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Creating Application Object Using Flask
app = Flask(__name__, static_url_path='/assets', static_folder='./flask_app/assets', template_folder='./flask_app')

# Configuring the Assets Path
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to Set Cache-Configuration to 'no-cache' in our case
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

# Routing default to 'index.html'
@app.route('/')
def root():
    return render_template('index.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/camera.html')
def camera():
    return render_template('camera.html')

@app.route('/capture.html')
def capture():
    return render_template('capture.html')

@app.route('/detect.html')
def detect():
    return render_template('detect.html')

@app.route('/upload.html')
def upload():
    return render_template('upload.html')


# When User Chooses the X-ray this method will be called
@app.route('/upload', methods=['POST', 'GET'])
def upload_pic():
    t_init = time()
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            file.save(os.path.join(
                app.config['UPLOAD_FOLDER'],'image.jpg'))
            
    # Loading Models to the Runtime
    inception_resnet_v2_glaucoma = load_model('inception_resnet_v2_glaucoma.h5')
    

    # Converting Image to Processable format
    img = cv2.imread('./flask_app/assets/images/image.jpg')

    # Resizing the Input Image
    img = cv2.resize(img, (224, 224))

    # Performing the Image Processing Using OpenCV
    transformed_glaucoma = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
    transformed_glaucoma = cv2.resize(transformed_glaucoma, (224, 224))

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret,thresh = cv2.threshold(gray_image,100,300,0) 
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contoured_glaucoma = cv2.drawContours(img,contours,-1,(0,300,0),1)
    contoured_glaucoma = cv2.resize(contoured_glaucoma, (224, 224))

    os.remove('./flask_app/assets/images/transformed_image.jpg')
    os.remove('./flask_app/assets/images/contoured_image.jpg')

    cv2.imwrite('./flask_app/assets/images/transformed_image.jpg', transformed_glaucoma)
    cv2.imwrite('./flask_app/assets/images/contoured_image.jpg', contoured_glaucoma)

    # Predicting the Probability of the result case
    inception_resnet_v2_glaucoma_pred = inception_resnet_v2_glaucoma.predict(image)
    probability = inception_resnet_v2_glaucoma_pred[0]
    print("inception_resnet_v2_glaucoma Predictions:")
    if probability[0] > 0.5:
        inception_resnet_v2_glaucoma_glaucoma_pred = str('%.2f' % (probability[0] * 100) + '% Glaucoma POSITIVE')
    else:
        inception_resnet_v2_glaucoma_glaucoma_pred = str('%.2f' % ((1 - probability[0]) * 100) + '% Glaucoma NEGATIVE')
    print(inception_resnet_v2_glaucoma_glaucoma_pred)


    t_final = time()

    elapsed_time = t_final - t_init
    elapsed_min = int(elapsed_time // 60)
    elapsed_sec = int(elapsed_time % 60)

    return render_template('detect.html', glaucoma_pred=inception_resnet_v2_glaucoma_glaucoma_pred, elapsed_min=elapsed_min, elapsed_sec=elapsed_sec)


#server
if __name__ == '__main__':
    #app.run(host='0.0.0.0', debug=True, port=80)
    app.run(host='0.0.0.0', port=8080)