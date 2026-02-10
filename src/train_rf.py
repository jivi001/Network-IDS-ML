from sklearn.ensemble import RandomForestClassifier
import joblib

def train_rf(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/rf_model.pkl")
    return rf
