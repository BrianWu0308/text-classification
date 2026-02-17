from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_baseline(X_train, y_train, X_val, y_val):
    # TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # Model
    model = LogisticRegression(max_iter=2000, n_jobs=-1)
    model.fit(X_train_vec, y_train)

    # Eval
    pred = model.predict(X_val_vec)

    acc = accuracy_score(y_val, pred)
    report = classification_report(y_val, pred, digits=4)
    cm = confusion_matrix(y_val, pred)

    return model, vectorizer, acc, report, cm
