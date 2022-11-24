import os
import modal

BACKFILL=False
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("my-custom-key"))
   def f():
       g()


def generate_passenger(survived):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    if survived:
        unif = random.uniform(0, 1)
        if unif < 109/342:
            sex = 0
        else:
            sex = 1
        if unif < 136/342:
            pclass = 1
        elif unif < 223/342:
            pclass = 2
        else:
            pclass = 3
        age = random.uniform(0.42, 80.0)
        if unif < 25/100:
            fare = random.uniform(0.0, 12.47)
        elif unif < 50/100:
            fare = random.uniform(12.47, 26.0)
        elif unif < 75/100:
            fare = random.uniform(26.0, 57.0)
        else:
            fare = random.uniform(57.0, 512.0)
        if unif < 233/342:
            parch = 0.0
        elif unif < (65+233)/342:
            parch = 1.0
        elif unif < (40+65+233)/342:
            parch = 2.0
        else:
            parch = round(random.uniform(3.0, 5.0))
        if unif < 210/342:
            sibsp = 0.0
        elif unif < (112+210)/342:
            sibsp = 1.0
        else:
            sibsp = round(random.uniform(2.0, 4.0))
        if unif < 219/342:
            embarked = 0.0
        elif unif < (93+210)/342:
            embarked = 1.0
        else:
            embarked = 2.0
    else:
        unif = random.uniform(0, 1)
        if unif < 468/549:
            sex = 0
        else:
            sex = 1
        if unif < 80/549:
            pclass = 1
        elif unif < 177/549:
            pclass = 2
        else:
            pclass = 3
        age = random.uniform(1.0, 74.0)
        if unif < 25/100:
            fare = random.uniform(0.0, 7.85)
        elif unif < 50/100:
            fare = random.uniform(7.85, 10.5)
        elif unif < 75/100:
            fare = random.uniform(10.5, 26.0)
        else:
            fare = random.uniform(26.0, 263.0)
        if unif < 445/549:
            parch = 0.0
        elif unif < (53+445)/549:
            parch = 1.0
        elif unif < (40+53+445)/549:
            parch = 2.0
        else:
            parch = round(random.uniform(3.0, 6.0))
        if unif < 398/549:
            sibsp = 0.0
        elif unif < (97+398)/549:
            sibsp = 1.0
        else:
            sibsp = round(random.uniform(2.0, 6.0))
        if unif < 427/549:
            embarked = 0.0
        elif unif < (75+427)/549:
            embarked = 1.0
        else:
            embarked = 2.0

    df = pd.DataFrame({ "sex": [sex], "age": [age], "pclass": [pclass], "fare": [fare],
                       "parch":[round(parch)], "sibsp": [round(sibsp)], "embarked": [round(embarked)]
                      })
    df['survived'] = round(survived)
    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        titanic_df = generate_passenger(1.0)
        print("Survivor added")
    else:
        titanic_df = generate_passenger(0.0)
        print("Deceased added")

    return titanic_df



def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv")
    else:
        titanic_df = get_random_passenger()

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
