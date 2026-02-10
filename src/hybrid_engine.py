def hybrid_predict(X, rf, iso):
    predictions = []

    for x in X:
        if rf.predict([x])[0] == 1:
            predictions.append(1)
        elif iso.predict([x])[0] == -1:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions
