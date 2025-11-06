from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import joblib

model = joblib.load('model/fake_news_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
pipeline = make_pipeline(vectorizer, model)

explainer = LimeTextExplainer(class_names=["Real", "Fake"])

def explain_prediction(text):
    exp = explainer.explain_instance(text, pipeline.predict_proba, num_features=5)
    return exp.as_list()
