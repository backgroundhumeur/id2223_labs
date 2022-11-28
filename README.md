# id2223_labs

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
