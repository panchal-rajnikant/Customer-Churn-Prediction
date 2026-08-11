import time

def predictions(models, X_test):

    predictions = []
    for name, model in models.items():
        start_time = time.time()
        pred = model.predict(X_test)
        end_time = time.time()
        train_time = end_time - start_time
        predictions.append({
                    name: pred,
                    # train_time: train_time
                })
    
    return predictions
