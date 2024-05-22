import torch
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from sklearn import preprocessing
import torch.nn.functional as F
from torch import nn
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from defisheye import Defisheye
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

#Scanning image
print("Scanning image\n")
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
rawCapture = PiRGBArray(camera)
# allow the camera to warmup
time.sleep(0.1)
# grab an image from the camera
camera.capture(rawCapture, format="bgr")
image = rawCapture.array

#Image defisheye

print("Converting image")

dtype = 'linear'
format = 'fullframe'
fov = 180
pfov = 90

obj = Defisheye(image, dtype=dtype, format=format, fov=fov, pfov=pfov)

# To use the converted image in memory

new_image = obj.convert()

#Image resizing
print("resizing image")

res = cv2.resize(new_image, dsize=(8, 16), interpolation=cv2.INTER_AREA)
gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) 

#Convert to tensor
print("Convert to tensor adn threshold")

inp = torch.tensor(gray_image)
inp = inp.to(torch.float32).reshape((1,1,16,8))

inp[inp<225.] = 1
inp[inp>=225.] = 0

#Go through model

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        
        self.fc1 = torch.nn.Linear(512, 26)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        #print('x_shape:', x.shape)
        x = x.view(-1, self.num_flat_features(x))
        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
model = Model().to(device)

model = torch.load('model.pth')

inp = read_image("i.jpg", mode = ImageReadMode.GRAY)
inp = inp.to(torch.float32).reshape((1,1,16,8))

inp[inp<=40.] = 1
inp[inp>40.] = 0

pred = model(inp)
pred_probab = nn.Softmax(dim=1)(pred)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {chr(y_pred[0]+97)}")