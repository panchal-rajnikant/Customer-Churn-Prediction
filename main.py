from src import data_preprocessing, train_model, predict, evaluate_model

models = {}

# data preprocessing
X_train, X_test, y_train, y_test = data_preprocessing.preprocess()

# training data
models = train_model.train(X_train, y_train)
print(models)



