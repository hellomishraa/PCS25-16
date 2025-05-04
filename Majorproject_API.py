#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI, File, UploadFile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import pandas as pd
import io

app = FastAPI()

# Load CSV and preprocess once
data = pd.read_csv("E:/final_data.csv")
data_segments = data['Text'].dropna().tolist()

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + ' '
    return text.strip()

def check_plagiarism(input_text, csv_segments):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform([input_text] + csv_segments)
    similarity_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:])
    plagiarism_scores = similarity_matrix[0]
    plag_percent = (sum(plagiarism_scores > 0.0) / len(csv_segments)) * 100
    return plag_percent

@app.post("/check-plagiarism/")
async def check_plagiarism_api(file: UploadFile = File(...)):
    pdf_content = await file.read()
    input_text = extract_text_from_pdf(io.BytesIO(pdf_content))
    result = check_plagiarism(input_text, data_segments)
    return {"plagiarism_percent": round(result, 2)}

