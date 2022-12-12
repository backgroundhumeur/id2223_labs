# id2223_labs

## Lab2

The goal of the lab was to allow users to translate Swedish videos into English
text through the use of a serverless machine learning pipeline.

In the lab2 folder, you can find the code for two notebooks that were used to fine-tune a model of OpenAI's whisper model(small) to transcribe and translate
spoken Swedish to English.

Additionnally, you can test the model on Youtube videos URLs through a huggingface 
spaces gradio app at [the following url](https://huggingface.co/spaces/humeur/Swedish-Whisper-from-Youtube).

Since the gradio app is run on the basic CPU provided by huggingface, it takes
a very long time to run one inference of the model even for small inputs(around
10mins). We can recommend some youtube videos to try the app with like [this
recent news report](https://www.youtube.com/watch?v=34QNxHTOsQU) or [an even shorter one](https://www.youtube.com/watch?v=AzlipxrzMe4).

### Explanation of the pipeline

The original notebook was split into two notebooks : one concerned with the
feature engineering that can be run exclusively with a CPU that takes around
~40mins. The other is concerned with the training of the feature data and the
frequent use of checkpoints uploaded onto a huggingface model repo to avoid
Collab interfering with prolonged use of its GPUs.

The feature engineering pipeline notebook downloads and cleans the data
downloaded from huggingface's public repos to then upload them onto a google
drive. These feature data can then be recovered either through access to the
drive by mounting it, or by downloading it through !curl commands.

The choice of GDrive was done out of practicallity, even if that entailed a need
to reduce the test set by subsampling it from 5k to 3k in order to comply with the 15gb limit.
This was reasonable though, as it matched the size of the training set to a 80/20 ratio.

The training pipeline notebook was built in order to minimize the time spent not
using the GPU. Using a separate feature pipeline, we were able to drive down the
overhead time to around only ~5mins for it to download the necessary feature dataset
used for the training.

We then make use of the training's parameters in order to force a push for every
checkpoints(1 every 1000 steps) into the model's repo to retrain from the
checkpoint in order of a Collab forceful disconnect.

### Possible improvements of the model

[Here](https://huggingface.co/humeur/lab2_id2223) is our current fine-tuned model.

* Model-centric approach

The first hyperparameter that could be changed is the maximum step count. We
didn't specify the maximum number of epochs but only trained up to 4000 steps
which resulted in 5.17/6 epochs, or in other words, pass over the whole training
set. It is clear that as we increase training time, the model would likely
improve though to more and more marginal decrease in loss and WER. This wasn't
tried out of concern of the already very long training time(~10h).

The AdamW learning rate of the training could also be lowered/increased for a possible boost in
WER depending on whether the training process has too much variance/overfits too
much the training data for a better test error.

One could also add weight decay for the AdamW optimizer in order to have the
model better generalise, but this would likely require longer training as well.

The long training time could be reduced further by making use of bigger training
and test batch sizes, but unfortunately the RAM of the GPUs collab provided
couldn't allow for bigger batche sizes.

The fp16 provides a nice boost in perfomances, but we didn't try the different
optimizers level that could possibly increasing perfomances further using
fp16_opt_level.

The final optimization that can be used and that was implemented is to reduce the test set size so that less time is spent at every evaluation. This is dangerous though as we can't reduce its size too much for fear of underestimating the error on unseen data. But a 80/20 ratio seemed like a good compromise, and this allowed us to drive the total evaluation time down 1h(=15min*4).

Another thing that can be mentioned is to change the pretrained model we use to
an even more complex one, like whisper-medium. Since it has more neurons, we
would expect better performances with the tradeoff of longer training times and
more epochs needed to arrive at a stable configuration.

* Data-centric approach

The most obvious data-centric approach to improve the model is to have a greater
training set. Initially we worked on the problem by reducing the size of the
training instead of the test set, and this lead to worse WER rates. So it is
clear that adding more data from other available data sources would be a good
way to improve our model.

Unfortunately, it is both difficult to find and make usable more potential
training data, since there are no seemingly easy sources online for Swedish
audio. One could investigate government agencies especially of the European
Union to maybe find good quality recordings usable for training.

Here is one of those EU dataset: https://live.european-language-grid.eu/catalogue/corpus/1594 

We can also cite Google's [FLEURS](https://huggingface.co/datasets/google/fleurs/viewer/sv_se/train) dataset that contains many labeled sound files
for training including a sv-SE variant.

The second problem is that the training data needs to be saved on cloud systems
and the current data already went over 15gb so any more would require to use
either extra Google Drives or Hopsworks storage space, but this would increase
download times when training and overall be more cumbersome than it already is.

---
## Lab1

This is the code for 4 serverless machine learning applications that use
2 well-known ML datasets, the Iris Flower dataset and the [Titanic](https://www.kaggle.com/competitions/titanic/data) dataset.

In the `src` folder, you can find all the code for the two applications for each
dataset.

The code for each datasets includes both the 2 pairs of feature/training python
files that need to be run locally with an Hopsworks & Modal account, and
2 folders containing the code that powers their respective huggingface spaces
online GUI, which can be found at the urls detailed below.


### 1. Iris

There are two applications for the Iris dataset : one predicts the variety of
the flower based on some parameters given through an interactive GUI, and the
other is a monitor app that runs daily predictions on new synthesised samples
and displays on an online Dashboard historical records of how well the model is
doing(accuracy & confusion matrix) along with the daily prediction.

The model used for the predictions is a binary classificator that uses KNN with
2 neighbours.

The code for this was already provided.

The resulting huggingface spaces can be found at:

1. [https://huggingface.co/spaces/humeur/Iris](https://huggingface.co/spaces/humeur/Iris)

2. [https://huggingface.co/spaces/humeur/iris-monitor](https://huggingface.co/spaces/humeur/iris-monitor)

### 2. Titanic

![Titanic sinking](https://raw.githubusercontent.com/backgroundhumeur/id2223_labs/main/src/titanic/assets/titanic_0.jpg)

There are two applications for the Titanic dataset : one predicts if a passenger of
the Titanic would have survived based on some parameters given through an
interactive GUI, and the other is a monitor app that runs daily predictions on new
synthesised samples and displays on an online Dashboard historical records of how well the model is doing(accuracy & confusion matrix) along with the daily prediction.

For the first app, you can choose many parameters and the prediction is
displayed as an image that either shows the Titanic sinking if the passenger
would have died, or Géricault's [*Le Radeau de la Méduse*](https://en.wikipedia.org/wiki/The_Raft_of_the_Medusa) if he would have survived.

For the second app, the newly synthesised passengers are generated using the
empirical data of the existing passengers coupled with uniform rolls based on
the probabilites associated with the different parameters given that they are
a survivor or not. They are then fed daily to an already running model, as
a form of monitoring.

The model used for the predictions is a binary classificator that uses XGBoost with
the standard hyperparameters.

We can see the perfomance on the test set here : 
![XGBoost confusion matrix](https://raw.githubusercontent.com/backgroundhumeur/id2223_labs/main/src/titanic/titanic_model/confusion_matrix.png)

The resulting huggingface spaces can be found at:

1. [https://huggingface.co/spaces/humeur/Titanic-Prediction](https://huggingface.co/spaces/humeur/Titanic-Prediction)

2. [https://huggingface.co/spaces/humeur/titanic-monitor](https://huggingface.co/spaces/humeur/titanic-monitor)
