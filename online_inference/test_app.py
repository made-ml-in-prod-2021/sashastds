import json

import pytest
from fastapi.testclient import TestClient

from app import app, load_model
from src.entities import InputData, ModelResponse


@pytest.fixture(scope="session", autouse=True)
def get_model():
    load_model()


@pytest.fixture()
def get_test_request_data():
    data = [
        InputData(
            id=8,
            sex=1,
            fbs=1,
            restecg=2,
            exang=1,
            slope=2,
            age=100,
            cp=3,
            trestbps=210,
            chol=600,
            thalach=210,
            oldpeak=7,
            ca=4,
            thal=3,
        ),
        InputData(
            id=88,
            sex=0,
            fbs=0,
            restecg=0,
            exang=0,
            slope=0,
            age=18,
            cp=0,
            trestbps=90,
            chol=110,
            thalach=65,
            oldpeak=0,
            ca=0,
            thal=0,
        ),
    ]
    return data


def test_can_get_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert 200 == response.status_code


def test_can_get_status_endpoint():
    with TestClient(app) as client:
        expected_text = "Model is ready"
        expected_status = 200
        response = client.get("/status")
        assert expected_status == response.status_code
        assert expected_text == response.json()


def test_predict_endpoint_works_correctly(get_test_request_data):
    expected_status = 200
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            data=json.dumps([item.__dict__ for item in get_test_request_data]),
        )
        responce_content = response.json()
        assert expected_status == response.status_code
        assert len(responce_content) == len(get_test_request_data)
        for idx, item in enumerate(get_test_request_data):
            assert responce_content[idx]["id"] == item.__getattribute__("id")
            assert (
                responce_content[idx]["target"] >= 0
                and responce_content[idx]["target"] <= 1
            )


def test_predict_endpoint_corrupted_data_type(get_test_request_data):
    with TestClient(app) as client:
        corrupted_data = get_test_request_data[0]
        corrupted_data.sex = "MALE"
        response = client.post("/predict", data=json.dumps([corrupted_data.__dict__]))
        expected_text = "value is not a valid integer"
        expected_status = 422
        assert expected_status == response.status_code
        assert expected_text == response.json()["detail"][0]["msg"]


def test_predict_endpoint_extreme_data_value(get_test_request_data):
    with TestClient(app) as client:
        corrupted_data = get_test_request_data[0]
        corrupted_data.age = -1
        response = client.post("/predict", data=json.dumps([corrupted_data.__dict__]))
        expected_status = 400
        expected_text = "value -1 in 'age' is out of [18, 100] interval"
        assert expected_status == response.status_code
        assert expected_text == response.json()["detail"]
