from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from doc_embed import documentEmbedding
from doc_clustering import documentClustering
from topic_model import topicModel
import os
import pandas as pd
# from model import predict, convert
# sentence_model = SentenceTransformer("all-mpnet-base-v2")

app = FastAPI()

# pydantic models
class docIn(BaseModel):
    document: dict

class topicOut(BaseModel):
    topics: dict
    
data_path = os.getcwd()
doc_emb = documentEmbedding(data_path)
doc_cluster = documentClustering(data_path)
topic_obj = topicModel(data_path)

@app.post("/topic/predict", response_model=topicOut, status_code=200)
def get_prediction(payload:docIn):
    doc_dic = payload.document
    df = pd.DataFrame(doc_dic.values(), columns=['article'])
    # Generate embedding
    emb_vectors = doc_emb.embedding_main(df)
    # Apply Clustering algorithm
    df['class_label'] = doc_cluster.test_cluster(emb_vectors)
    # Topic modeling
    df['preprocessed_article'] =  df['article'].apply(topic_obj.preprocess_article_text)
    df['result_topic'] = df.apply(lambda x: topic_obj.do_pridict(df['preprocessed_article'], df['class_label']), axis=1)

    response_object = {
    "topics": df['result_topic'].to_dict()}

    return response_object