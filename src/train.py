import numpy as np
import random
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Tải tất cả các tệp intent từ thư mục data/intents
intents = []
intents_dir = '../data/'
for filename in os.listdir(intents_dir):
    if filename.endswith('.json'):
        with open(os.path.join(intents_dir, filename), 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                if 'intents' not in data:
                    print(f"LỖI: Tệp '{filename}' thiếu khóa 'intents'.")
                else:
                    intents.extend(data['intents'])
            except json.JSONDecodeError:
                print(f"LỖI: Tệp '{filename}' chứa dữ liệu JSON không hợp lệ.")

all_words = []
tags = []
xy = []

# Lặp qua mỗi câu trong mẫu intent
for intent in intents:
    tag = intent['tag']
    # Thêm vào danh sách tag
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokenize từng từ trong câu
        w = tokenize(pattern)
        # Thêm vào danh sách từ của chúng ta
        all_words.extend(w)
        # Thêm vào cặp xy
        xy.append((w, tag))

# Stem và chuyển tất cả các từ thành chữ thường
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Loại bỏ bản sao và sắp xếp
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "mẫu")
print(len(tags), "tags:", tags)
print(len(all_words), "từ duy nhất đã được stem:", all_words)

# Tạo dữ liệu huấn luyện
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: túi từ cho mỗi pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss chỉ cần nhãn lớp, không cần mã hóa one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Siêu tham số
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Huấn luyện mô hình
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Truyền thuận
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Truyền ngược và tối ưu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Loss cuối cùng: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "../data.pth"
torch.save(data, FILE)

print(f'huấn luyện hoàn thành. file đã được lưu vào {FILE}')
