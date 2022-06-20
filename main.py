from distutils.log import debug
from fileinput import filename
from cv2 import FileStorage_NAME_EXPECTED
from flask import Flask ,request,jsonify, send_file
from pyparsing import original_text_for
import werkzeug
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from keras.models import Model
import pickle
import os
import shutil
import cv2
import numpy as np
from keras.applications.mobilenet import preprocess_input
import pandas as pd
from flask_jsonpify import jsonpify
from flask import send_file
import joblib
import timeit




def crop_img(img, y, x):

    img_size = 200
    height, width, _ = img.shape

    y = round(y * height)
    x = round(x * width)

    y_min = (y - round(img_size / 2)) if (y - round(img_size / 2)) >= 0 else 0
    x_min = (x - round(img_size / 2)) if (x - round(img_size / 2)) >= 0 else 0
    y_max = (y + round(img_size / 2)) if (y + round(img_size / 2)) <= height else height
    x_max = (x + round(img_size / 2)) if (x + round(img_size / 2)) <= width else width

    if (y_max - y_min) != img_size:
        if y_min == 0:
            y_max += img_size - (y_max - y_min)
        elif y_max == height:
            y_min -= img_size - (y_max - y_min)
    if (x_max - x_min) != img_size:
        if x_min == 0:
            x_max += img_size - (x_max - x_min)
        elif x_max == width:
            x_min -= img_size - (x_max - x_min)

    return img[y_min:y_max, x_min:x_max]

def predict(filenames):

    output_dir = 'c:/Users/sondos/StudioProjects/Medical/test'
    yolo_dir = 'C:/Users/sondos/StudioProjects/Medical/yolov5'
    logistic_path = 'C:/Users/sondos/StudioProjects/Medical/logisticModel.sav'

    original_image = cv2.imread("/Users/sondos/StudioProjects/Medical/input/" + filenames[0])
    o_height, o_width, _ = original_image.shape
    o_size = max(o_height, o_width) if max(o_height, o_width) > 416 else 416

    # some attr
    final_output = []
    wbc_count = 0
    types_count = [0, 0, 0, 0]

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

        os.mkdir('c:/Users/sondos/StudioProjects/Medical/test')

    # remove old yolo prediction
    if os.path.isdir(yolo_dir + r'/runs/detect'):
        shutil.rmtree(yolo_dir + r'/runs/detect')

    
    # load classification model
    base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3))
    mobileNet = Model(base_model.input, base_model.get_layer('out_relu').output)
    mobileNet.compile()
    logistic_Model = joblib.load(logistic_path)

    start = timeit.default_timer()
    os.system(r'python "' + yolo_dir + '/detect.py" --weights "' + yolo_dir
              + r'/runs/train/yolov5s_results/weights/best.pt" --img ' + str(o_size) + ' --conf 0.68 --source "' +
              "C:/Users/sondos/StudioProjects/Medical/input/" + '" --save-txt')

    
    
    for filename in filenames:

        original_image = cv2.imread("/Users/sondos/StudioProjects/Medical/input/" + filename)

        # get yolo prediction and loop through them
        if os.path.exists(yolo_dir + r'/runs/detect/exp/labels/'+os.path.splitext(filename)[0]+'.txt'):

            f = open(yolo_dir + r'/runs/detect/exp/labels/' + os.path.splitext(filename)[0]+'.txt', 'r')
            for line in f:
             line = line.split()
             if int(line[0]) == 2:
                # count WBC to name the tmp images
                wbc_count += 1

                # crop image depending on yolo
                x_center, y_center = float(line[1]), float(line[2])

                cropped_img = crop_img(original_image, y_center, x_center)
                cv2.imwrite(output_dir + r'/' + os.path.splitext(filename)[0] + '.jpg', cropped_img)
                cropped_img = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_AREA)

                # mobileNet model
                cropped_img = image.img_to_array(cropped_img)
                cropped_img = np.expand_dims(cropped_img, axis=0)
                cropped_img = preprocess_input(cropped_img)

                feature = mobileNet.predict(cropped_img)
                feature = feature.flatten()
                # logistic model
                predictions = logistic_Model.predict_proba(np.atleast_2d(feature))[0]
                predictions = np.argsort(predictions)[::-1][:5]
                predictions = predictions[0]

                types_count[predictions] += 1

                # edit final output
                line[0] = predictions
                final_output.append(line)
                stop = timeit.default_timer()

                print('Time: ', stop - start) 
                print('WBC #', wbc_count, ' is ', predictions)

             else:
                continue
        f.close()

    final_count = pd.DataFrame([types_count], columns=['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL'])
    final_count[['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']].to_csv(output_dir + r'\final_count.csv',
                                                                               index=False)  

    final_output = pd.DataFrame(final_output, columns=['cell_type', 'xcenter', 'ycenter', 'width', 'height'])
    final_output[['cell_type', 'xcenter', 'ycenter', 'width', 'height']].to_csv(output_dir + r'\final_output.csv',
                                                                                index=False)


app=Flask(__name__)
@app.route("/upload",methods = ['GET', 'POST'])
def upload():
    if request.method=="POST":

        fileNamess = []
        # imagefile =request.files["image"]
        if os.path.isdir('/Users/sondos/StudioProjects/Medical/input/'):
                shutil.rmtree('/Users/sondos/StudioProjects/Medical/input/')
                os.mkdir('/Users/sondos/StudioProjects/Medical/input/')
        for item in request.files.getlist('image'):
            # data = item.read()           
            filename = werkzeug.utils.secure_filename(item.filename)
            # print(filename)         
            item.save("/Users/sondos/StudioProjects/Medical/input/"+filename)
            # print("save image")
            fileNamess.append(filename)
            
        predict(fileNamess)
        return send_file('test/final_count.csv',
                     mimetype='text/csv',
                     attachment_filename='final_count.csv',
                     as_attachment=True)
      
if __name__ =="__main__":
    app.run(debug=True,port=80)