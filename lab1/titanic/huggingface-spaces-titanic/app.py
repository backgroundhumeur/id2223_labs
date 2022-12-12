import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(Sex, Age, Pclass, Fare, Parch, SibSp, Embarked):
    input_list = []
    input_list.append(Sex)
    input_list.append(Age)
    input_list.append(Pclass + 1.0)
    input_list.append(Fare)
    input_list.append(Parch)
    input_list.append(SibSp)
    input_list.append(Embarked)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1))
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    pic_url = "https://raw.githubusercontent.com/backgroundhumeur/id2223_labs/main/src/titanic/assets/titanic_" + str(res[0]) + ".jpg"
    img = Image.open(requests.get(pic_url, stream=True).raw)
    return img

demo = gr.Interface(
    fn=titanic,
    title="Titanic Passenger Survival Predictive Analytics",
    description="Experiment with different characteristics of a passenger to predict whether he would have survived if he were aboard the titanic.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Dropdown(choices=["male","female"],type='index', default="male", label="Sex"),
        gr.inputs.Slider(minimum=1.0,maximum=100.0,default=28.0,step=1.0, label="Age"),
        gr.inputs.Dropdown(choices=["First", "Second","Third"],type='index', default="First", label="Ticket class"),
        gr.inputs.Number(default=14.4542, label="Fare ($)"),
        gr.inputs.Number(default=0.0, label="Number of parents/children aboard"),
        gr.inputs.Number(default=0.0, label="Number of siblings/spouses aboard"),
        gr.inputs.Dropdown(choices=["S","C", "Q"],type='index', default="S", label="Port of Embarkation")
        ],
    outputs=gr.Image(type="pil"))

demo.launch()
