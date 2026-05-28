from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load("model/model.pkl")

app = FastAPI()

# Request schema
class Person(BaseModel):
    age: int
    cuts: int

@app.get("/")
def home():
    return {
        "message": "Emo Detector API."
    }

@app.post("/predict")
def predict(person: Person):

    data = [[
        person.age,
        person.cuts
    ]]

    prediction = model.predict(data)

    is_emo = int(prediction[0])

    if is_emo == 1:
        message = "The person is emo."
    else:
        message = "The person is not emo."

    return {
        "prediction": is_emo,
        "message": message
    }
