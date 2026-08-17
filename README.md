# Customer Churn Prediction

Predict whether a telecom customer will churn using multiple machine learning pipelines. The project trains and compares several models, selects the best-performing pipeline by F1 score, and exposes a simple prediction API.

## Dataset

The dataset is located at `data/customer_churn.csv` and includes these columns:

- `CustomerID`
- `Age`
- `Tenure`
- `Monthly Charges`
- `Contract` (e.g. `Month-to-month`, `One year`, `Two year`)
- `Churn` (target: `Yes` / `No`)

## Technologies

- Python
- pandas
- scikit-learn
- xgboost

## Installation

Install requirements:

```bash
pip install -r requirements.txt
```

## Usage

- Train and evaluate models (from project root):

```bash
python main.py
```

- Run the FastAPI server:

```bash
uvicorn app:app --reload
```

The API provides a `/predict` POST endpoint that accepts a JSON body matching the input schema: `Age`, `Tenure`, `Monthly_Charges`, `Contract`.

### Example request

POST /predict

```json
{
    "Age": 25,
    "Tenure": 3,
    "Monthly_Charges": 90.0,
    "Contract": "Month-to-month"
}
```

### Example response

```json
{
    "prediction": 1
}
```

> Note: the exact response schema depends on the API implementation; by default the route returns only the predicted label.

## Tests

Run the unit test suite (from the project root):

```bash
pip install -r requirements.txt
pytest -q
```

Tests include schema validation checks for the API request/response and additional unit tests under `tests/`.

## Models

Implemented and compared pipelines:

- Logistic Regression
- Decision Tree
- Random Forest Classifier
- Gradient Boosting
- Extreme Gradient Boosting (XGBoost)

## Training & Outputs

- The training flow is in `main.py`. The best model is saved to `models/best_model.pkl`.
- Evaluation metrics: Accuracy, Precision, Recall, F1 score.

## Configuration

Store project-wide constants (feature lists, default paths, hyperparameters) in a dedicated module such as `src/constants.py` or `src/config.py` and import from `src` modules.

## Contributing

Contributions are welcome. Open an issue or submit a pull request with a short description of the change.

## License

Add a `LICENSE` file or include license terms here.


