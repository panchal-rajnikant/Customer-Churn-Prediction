from src import data_preprocessing, train_model, predict, evaluate_model
import joblib
from src.logger import log_event

models = {}

# data preprocessing
X_train, X_test, y_train, y_test = data_preprocessing.preprocess()

# training data
models = train_model.train(X_train, y_train)

# Predicting data
predictions = predict.predictions(models, X_test)

# Model Evaluation
results  = evaluate_model.evaluate(predictions, y_test)
print(results)
log_event("Results: ", results)

# Sort by the metric priority you want
best_row = results.sort_values(
    by=["F1 Score", "Recall", "Accuracy"],
    ascending=[False, False, False]
).iloc[0]

# model name selection
best_model_name = best_row["Model"]
best_model = models[best_model_name]

log_event("Saved best_model.pkl")
# save the best model
joblib.dump(best_model, "models/best_model.pkl")



