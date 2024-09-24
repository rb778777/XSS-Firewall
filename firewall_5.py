from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import os

app = Flask(__name__)

class XSSDetector(nn.Module):
    def __init__(self, n_classes):
        super(XSSDetector, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 128)  
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(128, n_classes)  

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = bert_output.last_hidden_state[:, 0, :]  
        output = self.drop(output)
        output = self.fc1(output)  
        output = self.drop(output)
        return self.out(output)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


model = XSSDetector(n_classes=2)


checkpoint_path = 'xss_detection_model.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))  
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

model.eval()


def preprocess(sentence):
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=min(512, len(sentence)),  
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    return encoding['input_ids'], encoding['attention_mask']


def predict(sentence):
    input_ids, attention_mask = preprocess(sentence)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probs, dim=1)
        confidence = probs[0][prediction].item()

    
    threshold = 0.73  
    if confidence >= threshold:
        return prediction.item(), confidence
    else:
        return 0, confidence  


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.form.get('data')
    if data is None or data.strip() == '':
        return jsonify({"error": "No data provided"}), 400

    
    try:
        prediction, confidence = predict(data)
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {str(e)}"}), 500

    
    if prediction == 1:
        return jsonify({"result": "XSS detected", "confidence": confidence}), 403
    else:
        return jsonify({"result": "No XSS detected", "confidence": confidence}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

