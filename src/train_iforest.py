from sklearn.ensemble import IsolationForest
import joblib

def train_iforest(X_train, y_train):
    X_normal = X_train[y_train == 0]

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42
    )
    iso.fit(X_normal)
    joblib.dump(iso, "models/iforest_model.pkl")
    return iso
