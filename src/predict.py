
def predictions(models, X_test):

    predictions = []
    for name, model in models.items():
        # start_time = time.time()
        pred = model.predict(X_test)
        predictions.append({
            name: pred
        })
        #  train_time = end_time - start_time
    
    return predictions
