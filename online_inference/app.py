import os
import sys
from typing import List, Optional, Union, Dict

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

from src.entities import InputData, ModelResponse
from src.validate import check_data_valid, CAT_FEATURES, NUM_FEATURES
from src.utils import load_object, setup_logging, make_binary_prediction
from src.preprocessing import extract_features

ClassificationModel = Union[LGBMClassifier, RandomForestClassifier]

MODEL_CUTOFF = 0.48
ID_COLUMN = "id"
PATH_TO_MODEL = './models/'

model: Optional[ClassificationModel] = None
transformers: Optional[Dict] = None

logger = setup_logging()
app = FastAPI()


def make_prediction(data: List[InputData], model: ClassificationModel, transformers: Dict) -> List[ModelResponse]:

    data_for_model = pd.DataFrame(instance.__dict__ for instance in data)[CAT_FEATURES + NUM_FEATURES]
    ids = [instance.__getattribute__(ID_COLUMN) for instance in data]
    
    logger.info(f"Creating features")
    X = extract_features(
        data_for_model, CAT_FEATURES, NUM_FEATURES, transformers, mode="transform"
    )
    logger.info(f"Features shape: {X.shape}")

    logger.info(f"Scoring with model")
    binary_predictions = make_binary_prediction(model, X, MODEL_CUTOFF)

    return [
        ModelResponse(id=instance_id, target=prediction)
        for instance_id, prediction in zip(ids, binary_predictions)
    ]


@app.get("/")
def main():
    return "Check the '/docs' endpoint to see how to get predictions on your data"


@app.on_event("startup")
def load_model():

    logger.info(f"Loading model components from path: {PATH_TO_MODEL}")
    global model
    global transformers
    model = load_object(PATH_TO_MODEL + "classifier", verbose=False)
    transformers = load_object(PATH_TO_MODEL + "transformers", verbose=False)


@app.get("/status")
def status() -> bool:
    return f"Model is{' not ' if model is None or transformers is None else ' '}ready"


@app.api_route("/predict", response_model=List[ModelResponse], methods=["POST"])
def predict(request: List[InputData]):

    for data in request:
        logger.info(f"Checking dataframe for model consistency")
        is_valid, error_message = check_data_valid(data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
    return make_prediction(request, model, transformers)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
