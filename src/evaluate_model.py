import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(predictions, y_test):
    results = []
    for model in predictions:
         for name, pred in model.items():
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1 Score": f1_score(y_test, pred),
            })

    results_df = pd.DataFrame(results)
    return results_df