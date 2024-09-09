from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import BiLSTMWithAttention  # Your model
from prometheus_client import start_http_server, Summary

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict/")
async def predict(input: InputText):
    # Process input
    sequences = tokenizer.texts_to_sequences([input.text])
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    inputs = torch.LongTensor(padded_sequences)

    # Get model prediction
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    
    return {"prediction": label_encoder.inverse_transform(predicted.numpy())[0]}


@app.post("/predict/")
@REQUEST_TIME.time()
async def predict(input: InputText):
    # Your prediction code

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
