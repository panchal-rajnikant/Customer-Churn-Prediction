from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def train(X_train, y_train):

    # model
    lr = LogisticRegression(
        max_iter=1000 
    )
   
    dt = DecisionTreeClassifier(random_state=42)

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    gb = GradientBoostingClassifier(
        random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric="logloss"
    )

    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    models = {
        "Logistic Regression": lr,
        "Decision Tree": dt,
        "Random Forest Classifier": rf,
        "Gradient Boosting": gb,
        "Extreme Gradient Boosting": xgb
    }
    return models