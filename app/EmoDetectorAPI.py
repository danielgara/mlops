from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel


# main class
class EmoDetectorAPI:
    # class attributes
    ROOT = Path(__file__).resolve().parent.parent
    MODEL_PATH = ROOT / "model" / "model.pkl"

    # inner classes
    class Person(BaseModel):
        age: int
        cuts: int

    # private methods
    @staticmethod
    def _load_model():
        return joblib.load(EmoDetectorAPI.MODEL_PATH)

    @staticmethod
    def _home() -> dict:
        return {"message": "Emo Detector API."}

    @staticmethod
    def _predict(model, person: "EmoDetectorAPI.Person") -> dict:
        data = [[person.age, person.cuts]]
        is_emo = int(model.predict(data)[0])

        if is_emo == 1:
            message = "The person is emo."
        else:
            message = "The person is not emo."

        return {"prediction": is_emo, "message": message}

    # public methods
    @staticmethod
    def create_app() -> FastAPI:
        model = EmoDetectorAPI._load_model()
        app = FastAPI()

        @app.get("/")
        def home():
            return EmoDetectorAPI._home()

        @app.post("/predict")
        def predict_endpoint(person: EmoDetectorAPI.Person):
            return EmoDetectorAPI._predict(model, person)

        return app


app = EmoDetectorAPI.create_app()
