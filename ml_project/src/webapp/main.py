import os
import joblib
import numpy as np

from src.configs import configs
from fastapi import FastAPI, status, HTTPException, Request

app = FastAPI()


@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(data: Request):
    model = joblib.load(configs.output_model_path)

    data = await data.json()
    data = np.array(data['data'])
    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    predicted = model.predict(data)
    return {"predictions": predicted.tolist()}


@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    ok = os.path.exists(configs.output_model_path)
    if not ok:
        raise HTTPException(status_code=425, detail="Model not trained yet")
    return {"status": "ok"}
