import gradio as gr
from pytube import YouTube
from transformers import pipeline

class GradioInference():
  def __init__(self):
    self.transcribe_model = pipeline(model='humeur/lab2_id2223')
    self.translate_model = pipeline("translation_SV_to_EN", model="Helsinki-NLP/opus-mt-sv-en")
    self.yt = None

  def __call__(self, link):
    if self.yt is None:
      self.yt = YouTube(link)
    path = self.yt.streams.filter(only_audio=True)[0].download(filename="tmp.mp4")
    results = self.transcribe_model(path)
    results = self.translate_model(results["text"])
    return results[0]['translation_text']

  def populate_metadata(self, link):
    self.yt = YouTube(link)
    return self.yt.thumbnail_url, self.yt.title

gio = GradioInference()
title="SWED->EN Youtube Transcriber (Whisper)"
description="Speech to text transcription of Youtube videos using OpenAI's Whisper finetunned for Swedish to English translation"

block = gr.Blocks()
with block:
    gr.HTML(
        f"""
            <div style="text-align: center; max-width: 500px; margin: 0 auto;">
              <div>
                <h1>{title}</h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                {description}
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
          link = gr.Textbox(label="YouTube Link")
          title = gr.Label(label="Video Title")
          with gr.Row().style(equal_height=True):
            img = gr.Image(label="Thumbnail")
            text = gr.Textbox(label="Transcription", placeholder="Transcription Output", lines=10)
          with gr.Row().style(equal_height=True):
              btn = gr.Button("Transcribe")
          btn.click(gio, inputs=[link], outputs=[text])
          link.change(gio.populate_metadata, inputs=[link], outputs=[img, title])
block.launch()
