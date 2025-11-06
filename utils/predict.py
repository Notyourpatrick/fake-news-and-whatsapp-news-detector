import joblib

model = joblib.load('model/fake_news_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

def predict_news(text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]
    proba = model.predict_proba(text_vec)[0].max()
    return pred, round(proba * 100, 2)
