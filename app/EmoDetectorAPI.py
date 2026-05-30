from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel


class EmoDetectorAPI:
    ROOT = Path(__file__).resolve().parent.parent
    MODEL_PATH = ROOT / "model" / "model.pkl"

    class Person(BaseModel):
        age: int
        cuts: int

    @staticmethod
    def load_model():
        return joblib.load(EmoDetectorAPI.MODEL_PATH)

    @staticmethod
    def home() -> dict:
        return {"message": "Emo Detector API."}

    @staticmethod
    def predict(model, person: "EmoDetectorAPI.Person") -> dict:
        data = [[person.age, person.cuts]]
        is_emo = int(model.predict(data)[0])

        if is_emo == 1:
            message = "The person is emo."
        else:
            message = "The person is not emo."

        return {"prediction": is_emo, "message": message}

    @staticmethod
    def create_app() -> FastAPI:
        model = EmoDetectorAPI.load_model()
        app = FastAPI()

        @app.get("/")
        def home():
            return EmoDetectorAPI.home()

        @app.post("/predict")
        def predict_endpoint(person: EmoDetectorAPI.Person):
            return EmoDetectorAPI.predict(model, person)

        return app


app = EmoDetectorAPI.create_app()
