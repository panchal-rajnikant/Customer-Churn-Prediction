import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.model_selection import train_test_split

def preprocess():
    df = pd.read_csv("../data/customer_churn.csv")

    # check missing values
    print(df.isnull().sum())

    # convert into numeric
    df["Monthly Charges"]= pd.to_numeric(
        df["MonthlyCharges"],
        errors="coerce"
    )
    # fill missing values with median
    df["Monthly Charges"]= df["Monthly Charges"].fillna(
        df["Monthly Charges"].mean()
    )

    # remove unnecessary column
    df = df.drop("CustomerID", axis=1)

    df = pd.get_dummies(df, columns=["Contract"])
    df["Churn"] = le.fit_transform(df["Churn"])


    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y )

    return X_train, y_train, X_test, y_test