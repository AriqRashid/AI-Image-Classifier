import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as transforms
import numpy as np
from ConvNet import ConvNet

def softmax(a, b):
    values = np.array([a, b])
    e_values = np.exp(values - np.max(values))  
    return e_values / np.sum(e_values)


PATH = "models\e10_b32_lr01_91pct.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ConvNet().to(device)
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

img_path  =  sys.argv[1]
transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
image = Image.open(img_path)
image = transform(image)


with torch.no_grad():
    outputs = model(image)

#  _, predicted = torch.max(outputs.data, 1)
# print(_)
# print(predicted)

values = outputs.cpu().detach().numpy()

if values[0][0] > abs(values[0][1]):
    print("\nThe model predicts this image is AI GENERATED")

else:
    print("\nThe model predicts this image is AUTHENTIC")


fake, real = softmax(values[0][0], values[0][1])
print(f'{round(fake*100, 2)} % it is AI generated.\n{round(real*100, 2)} % it is authentic.\n')