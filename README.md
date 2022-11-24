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
and displays historical records of how well the model is doing along with the
daily prediction.

The model used for the predictions is a binary classificator that uses KNN with
2 neighbours.

The code for this was already provided.

The resulting huggingface spaces can be found at:

1. [https://huggingface.co/spaces/humeur/Iris](https://huggingface.co/spaces/humeur/Iris)

2. [https://huggingface.co/spaces/humeur/iris-monitor](https://huggingface.co/spaces/humeur/iris-monitor)

### 2. Titanic

There are two applications for the Titanic dataset : one predicts if a passenger of
the Titanic would have survived based on some parameters given through an
interactive GUI, and the other is a monitor app that runs daily predictions on new
synthesised samples and displays historical records of how well the model is doing
along with the daily prediction.

For the first app, you can choose many parameters and the prediction is
displayed as an image that either shows the Titanic sinking if the passenger
would have died, and the other shows Géricault's [*Le Radeau de la Méduse*](https://en.wikipedia.org/wiki/The_Raft_of_the_Medusa) if he would
have survived.

The model used for the predictions is a binary classificator that uses KNN with
2 neighbours.

The resulting huggingface spaces can be found at:

1. [https://huggingface.co/spaces/humeur/Titanic-Prediction](https://huggingface.co/spaces/humeur/Titanic-Prediction)

2. [https://huggingface.co/spaces/humeur/titanic-monitor](https://huggingface.co/spaces/humeur/titanic-monitor)
