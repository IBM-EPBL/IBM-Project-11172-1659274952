

import os
import numpy as np 
import uuid
import urllib
from flask import Flask,request,render_template
#pip install tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app=Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model=load_model(os.path.join(BASE_DIR,'ECG_Classification.h5'))

@app.route("/")
def about():
    return render_template("Home.html")
@app.route("/about")
def home():
    return render_template("Home.html")

@app.route("/types")
def types():
    return render_template("Types.html")

@app.route("/info")
def information():
    return render_template("Info.html")

@app.route("/upload")
def test():
    return render_template("predict.html")


@app.route("/predict",methods=["GET","POST"])
def upload():
    if request.method == 'POST':
        f=request.files['file'] #requesting the file
        basepath=os.path.dirname('__file__')#storing the file directory
        filepath=os.path.join(basepath,"uploads",f.filename)#storing the file in uploads folder
        f.save(filepath)#saving the file
        
        img=image.load_img(filepath,target_size=(64,64)) #load and reshaping the image
        x=image.img_to_array(img)#converting image to array
        x=np.expand_dims(x,axis=0)#changing the dimensions of the image
        
        pred=model.predict(x)#predicting classes
        y_pred = np.argmax(pred)
        print("prediction",y_pred)#printing the prediction
    
        index=['Left Bundle Branch Block','Normal','Premature Atrial Contraction',
       'Premature Ventricular Contractions', 'Right Bundle Branch Block','Ventricular Fibrillation']
        result=str(index[y_pred])

        return result
    return None

#port = int(os.getenv("PORT"))
if __name__=="__main__":
    app.run(debug=False)