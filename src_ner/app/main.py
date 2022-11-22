from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from ner_predict import nerModel
import os
import pandas as pd
# from model import predict, convert
# sentence_model = SentenceTransformer("all-mpnet-base-v2")

app = FastAPI()

# pydantic models
class sentIn(BaseModel):
    sentence: str

class sentOut(BaseModel):
    ner_tags: dict
    
data_path = os.path.join(os.getcwd(), "asset")
ner_obj = nerModel(data_path)

@app.post("/ner/predict", response_model=sentOut, status_code=200)
def get_prediction(payload:sentIn):
    sent_txt = payload.sentence
    ner_result = ner_obj.do_pridict(sent_txt)
    response_object = {
    "ner_tags": ner_result
    }

    return response_object
