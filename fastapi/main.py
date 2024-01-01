from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

# Load the BERT tokenizer and model
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("CustomModel")
    return tokenizer, model

tokenizer, model = get_model()

# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def read_item():
    return FileResponse("static/index.html")

# Endpoint for emotion prediction
@app.post("/predict")
async def predict_emotion(request: Request):
    data = await request.json()
    print(data)
    
    if 'text' in data:
        user_input = data['text']
        test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')
        output = model(**test_sample)
        predictions = torch.nn.functional.softmax(output.logits, dim=-1)
        predictions = predictions.cpu().detach().numpy()
        class_names = ['Negative emotion', 'Neutral emotion', 'Positive emotion'] 
        predicted_class_indices = np.argmax(predictions, axis=1)
        predicted_class_names = [class_names[idx] for idx in predicted_class_indices] 
        response = {"Received Text": user_input, "Prediction": predicted_class_names}
    else:
        response = {"Received Text": "No Text Found"}

    return response

if __name__ == "__main__":
    uvicorn.run("main:app", host='localhost', port=8080, reload=True, debug=True)
