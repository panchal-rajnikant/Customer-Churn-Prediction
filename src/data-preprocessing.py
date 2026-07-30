import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

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

    return X, y