from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from joblib import load

from predict import extract_structured_data
from config import MODEL_PATH

# PRELOAD model at startup
recipeModel = load(MODEL_PATH)

app = FastAPI()

@app.post("/predict")
async def predict(html: UploadFile = File(...)):
    html_text = (await html.read()).decode("utf-8")
    result = extract_structured_data(html_text, recipeModel)
    return JSONResponse(result)

@app.get("/ping")
async def ping():
    return {"status": "ok"}