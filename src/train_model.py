from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from src.constants import numeric_features
from src.logger import log_event
import pandas as pd

def train(X_train, y_train):
    log_event("model training started")
    
    # numerical data may contain missing values and may need scaling. 
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # combine the ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
    ])

    # Feature selection
    selector = SelectKBest(
        score_func=f_classif,
        k=3
    )

    # model
       
    pipeline_lr = Pipeline([
        ("preprocessor", preprocessor),
        ("selector", selector),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])
    pipeline_dt = Pipeline([
        ("preprocessor", preprocessor),
        ("selector", selector),
        ("model", DecisionTreeClassifier(random_state=42))
    ])

    pipeline_rf = Pipeline([
        ("preprocessor", preprocessor),
        ("selector", selector),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    pipeline_gb = Pipeline([
        ("preprocessor", preprocessor),
        ("selector", selector),
        ("model", GradientBoostingClassifier(random_state=42))
    ])

    pipeline_xgb = Pipeline([
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("model", XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            eval_metric="logloss"
        ))
    ])
    

    pipeline_lr.fit(X_train, y_train)
    pipeline_dt.fit(X_train, y_train)
    pipeline_rf.fit(X_train, y_train)
    pipeline_gb.fit(X_train, y_train)
    pipeline_xgb.fit(X_train, y_train)

    feature_names = pipeline_rf.named_steps["preprocessor"].get_feature_names_out()
    print("Feature Names :",feature_names)

    importance = pipeline_rf.named_steps["model"].feature_importances_
    feature_importance = pd.DataFrame({
        "feature": feature_names,
         "importance": importance
    })

    feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
    )

    print("feature_importance :", feature_importance)
    models = {
        "Logistic Regression": pipeline_lr,
        "Decision Tree": pipeline_dt,
        "Random Forest Classifier": pipeline_rf,
        "Gradient Boosting": pipeline_gb,
        "Extreme Gradient Boosting": pipeline_xgb
    }
    log_event("model training completed")
    return models