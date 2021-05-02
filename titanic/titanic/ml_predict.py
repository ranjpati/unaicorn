def prediction_model(pclass,sex,age,sibSp,parch,fare,embarked,title):
    import pickle
    x = [[pclass,sex,age,sibSp,parch,fare,embarked,title]]
    randomForest = pickle.load(open("titanic_RF_Model.sav", 'rb'))
    prediction = randomForest.predict(x)
    if prediction == 0:
        prediction = 'Not Survived'
    elif prediction == 1:
        prediction = 'Survived'
    else:
        prediction = 'Error'
    return prediction
