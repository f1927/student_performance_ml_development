numpy
pandas
scikit-learn
flask
fastapi
uvicorn
gradio
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("student_model.pkl", "rb"))

app = FastAPI(title="Student Performance Prediction API")

# Input schema
class StudentData(BaseModel):
    age: int
    study_hours: float
    attendance: float
    prev_score: float
    internet_hours: float

@app.get("/")
def home():
    return {"message": "Student Performance Prediction API is running"}

@app.post("/predict")
def predict(data: StudentData):
    features = np.array([[ 
        data.age,
        data.study_hours,
        data.attendance,
        data.prev_score,
        data.internet_hours
    ]])

    prediction = model.predict(features)[0]

    result = "PASS" if prediction == 1 else "FAIL"

    return {"prediction": result}
    http://127.0.0.1:5000
