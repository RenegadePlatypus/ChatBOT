import random
import json 
import torch
from model import NeuralNet
from nltk_utils import bag_of_words,tokenize
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
queries = []
bot_name = "Nex_Gen"
print("___ is a product which is able to automate a task. Using this prductivity tool will save many hours of your valuable time. Currently, the product is under development. If you want to gain early access, please type yes")
no_count = 0
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break
    if sentence in ["no","nah","never mind","not interested"]:
        no_count += 1
    if no_count == 3:
        break
    if sentence not in ["Yes","yes","yea","Yeah","Okay","OK","okay","yeah"]:
        queries.append(sentence)
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")

output = pd.DataFrame(queries)
output.to_csv('output.csv',index=False)