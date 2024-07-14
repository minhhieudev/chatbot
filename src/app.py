import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import random
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import json

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "http://localhost:8081"}})

# Load all intent files from the data/intents directory
intents = []
intents_dir = '../data/'
for filename in os.listdir(intents_dir):
    if filename.endswith('.json'):
        with open(os.path.join(intents_dir, filename), 'r', encoding='utf-8') as file:
            intents.extend(json.load(file)['intents'])

# Load trained model
FILE = "../data.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"

def chatbot_response(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "Tôi không hiểu..."

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = chatbot_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Sử dụng debug=False để không chạy trong chế độ debug
