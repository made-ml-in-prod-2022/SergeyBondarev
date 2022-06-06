from fastapi.testclient import TestClient
from src.webapp.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200 or response.status_code == 425


def test_predict():
    response = client.post("/predict", json={"data": [66,178,228,165,1.0,0,3,0,1,1,1,2,2]})
    print(response.text)
    assert response.status_code == 200
    assert response.json() == {"predictions": [0]} or response.json() == {"predictions": [1]}


def test_batch_predict():
    data = [[66,178,228,165,1.0,0,3,0,1,1,1,2,2], [59,140,177,162,0.0,1,3,0,1,0,0,1,2]]
    response = client.post("/predict", json={"data": data})
    print(response.text)
    assert response.status_code == 200
    assert response.json() == {"predictions": [0, 0]} or response.json() == {"predictions": [0, 1]} \
           or response.json() == {"predictions": [1, 0]} or response.json() == {"predictions": [1, 1]}
