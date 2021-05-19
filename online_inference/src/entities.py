from pydantic import BaseModel

class InputData(BaseModel):
    id: int
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class ModelResponse(BaseModel):
    id: int
    target: int
