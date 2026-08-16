import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.model_selection import train_test_split
from pathlib import Path


def preprocess():
    base_dir = Path(__file__).resolve().parents[1]
    csv_path = base_dir / "data" / "customer_churn.csv"
    df = pd.read_csv(csv_path)

    # check missing values
    print(df.isnull().sum())

    # convert into numeric
    df["Monthly Charges"]= pd.to_numeric(
        df["Monthly Charges"],
        errors="coerce"
    )
    # fill missing values with median
    df["Monthly Charges"]= df["Monthly Charges"].fillna(
        df["Monthly Charges"].mean()
    )

    # remove unnecessary column
    df = df.drop("CustomerID", axis=1)

    df = pd.get_dummies(df, columns=["Contract"])
    # ensure Churn is numeric 0/1
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    else:
        df["Churn"] = le.fit_transform(df["Churn"])

    X = df.drop("Churn", axis=1)
    y = df["Churn"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test