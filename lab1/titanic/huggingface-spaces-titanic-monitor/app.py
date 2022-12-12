import gradio as gr
from PIL import Image
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/latest_survival.jpg")
dataset_api.download("Resources/images/actual_survival.jpg")
dataset_api.download("Resources/images/tit_df_recent.png")
dataset_api.download("Resources/images/tit_confusion_matrix.png")

with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          gr.Label("Today's Predicted Image")
          input_img = gr.Image("latest_survival.jpg", elem_id="predicted-img")
      with gr.Column():          
          gr.Label("Today's Actual Image")
          input_img = gr.Image("actual_survival.jpg", elem_id="actual-img")        
    with gr.Row():
      with gr.Column():
          gr.Label("Recent Prediction History")
          input_img = gr.Image("tit_df_recent.png", elem_id="recent-predictions")
      with gr.Column():          
          gr.Label("Confusion Maxtrix with Historical Prediction Performance")
          input_img = gr.Image("tit_confusion_matrix.png", elem_id="confusion-matrix")        

demo.launch()