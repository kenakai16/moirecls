import os
import numpy as np
from PIL import Image

from flask import Flask, render_template, request, send_from_directory
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import math

print(torch.__version__)
plt.ion()   # interactive mode

app = Flask(__name__)

STATIC_FOLDER = 'static'
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'
MODEL_FOLDER = STATIC_FOLDER + '/models'

torch.manual_seed(0)
CHECK_POINT_PATH = STATIC_FOLDER + '/models/best_model.pkl'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def apply_test_transforms(inp):
    out = transforms.functional.resize(inp, [224,224])
#     print(np.shape(out))
    out = transforms.functional.to_tensor(out)
#     print(np.shape(out))
    out = transforms.functional.normalize(out, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    out = out.to(device)
    return out


def restore_net():
    # restore entire net1 to net2
    global net
    net = torch.load(CHECK_POINT_PATH)
    return net

def predict(path):
    print("path:",path)
    im = Image.open(path)
    im = im.convert('RGB')
    print('img shape is ' + str(np.array(im).shape))
#     plt.imshow(im)

    im_as_tensor = apply_test_transforms(im)
    print(im_as_tensor.size())
    minibatch = torch.stack([im_as_tensor])
    print(minibatch.size())
    net(minibatch)

    softMax = nn.Softmax(dim = 1)
    preds = softMax(net(minibatch))
    preds = list([preds[0,0].item(),preds[0,1].item()])

    predict_proba = max(preds)
    predict_label = "moire" if preds.index(predict_proba)==0 else "normal"

    print("label:",predict_label)
    print("proba:",predict_proba)
    
    return predict_label,predict_proba


# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Process file and predict his label
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)

        img = Image.open(file)
        img = img.convert('RGB')
        img.save(fullname, quality=90)


        label,proba = predict(file)

        return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=proba)


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def create_app():
    restore_net()
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host="0.0.0.0",debug=True)
