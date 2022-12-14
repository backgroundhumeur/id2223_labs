import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("my-custom-secret"))
   def f():
       g()

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()
    titanic_df = pd.read_csv("https://raw.githubusercontent.com/backgroundhumeur/id2223_labs/main/src/titanic/data/titanic.csv")
    #drop irrelevant columns that have either too few values or too many unique values
    titanic_df = titanic_df[['Sex','Age','Pclass','Fare','Parch','SibSp','Embarked', 'Survived']]
    #fill NAs with default values
    def_values = {'Age': titanic_df['Age'].mean(), 'Embarked': titanic_df['Embarked'].value_counts().idxmax()}
    titanic_df = titanic_df.fillna(value=def_values)
    #replace all str values with numeric values for the training
    titanic_df = titanic_df.replace({'male':0,'female':1,'S':0,'C':1,'Q':2})
    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=['Sex','Age','Pclass','Fare','Parch','SibSp','Embarked'],
        description="Titanic passengers dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
