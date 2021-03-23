from flask import Flask, request, jsonify

#from script import getPrediction

import sys
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import wget
import json

# th architecture to use
arch = 'resnet18'
# load the pre-trained weights
model_file = './resnet18_places365.pth.tar'

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = './categories_places365.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

def getPrediction(img_url):
    img_name = wget.download(img_url)
    img = Image.open(img_name).convert('RGB')
    input_img = V(centre_crop(img).unsqueeze(0))
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    # probs, idx = h_x
    # print(h_x)
    h_x_trans = h_x.detach().numpy()
    # print(h_x_trans)
    # for h in h_x_trans:
    #     print('****** ',h)

    # output = []
    # for i in range(0, 365):
    #     mp = {}
    #     mp[classes[idx[i]]] = probs[i].item()
    #     output.append(mp)

    path = os.path.join('./',img_name)
    if os.path.exists(path):
        os.remove(path)

    return h_x_trans

# output = getPrediction(sys.argv[1])
# print('output: ', output)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict')
def predict():
    print("we are here *******************************")
    url = request.args.get('url')
    print("HERE",url)

    output = getPrediction(url)
    return_output = ' '.join(output.astype('str'))
    return return_output

app.run(host='localhost', port=5000)
if __name__=="__main__":
    app.run(debug=True)