from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from joblib import load

from predict import extract_structured_data  # <-- import your real function
from config import MODEL_PATH

# PRELOAD model at startup (example: set global variable)
recipeModel = load(MODEL_PATH)

def extract_structured_data_with_model(html_text: str) -> dict:
    # If your extract_structured_data takes a model, pass it here; else, refactor to use the preloaded model.
    return extract_structured_data(html_text, recipeModel)  # Or whatever your function requires

app = FastAPI()

@app.post("/predict")
async def predict(html: UploadFile = File(...)):
    html_text = (await html.read()).decode("utf-8")
    result = extract_structured_data_with_model(html_text)
    return JSONResponse(result)

@app.get("/ping")
async def ping():
    return {"status": "ok"}