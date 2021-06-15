from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates

from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
import torch.nn.functional as F
from api.dog_breed_classification_model import DogBreedClassification

templates = Jinja2Templates(directory="./templates/")

dog_breeds = {0:'beagle', 1:'chihuahua', 2:'doberman',3:'french_bulldog', 4:'golden_retriever', \
                      5:'malamute', 6:'pug',7:'saint_bernard', 8:'scottish_deerhound',9:'tibetan_mastiff'}



class DogBreedImageDataset:
    def __init__(self,img,transform = None):
        """
        Args:
            img : img file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img = img
        self.transform = transform
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.transform(self.img)
        return img

#image transformations 
transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],inplace=True)
            ])

#load the model
model = DogBreedClassification(len(dog_breeds))
model.load_state_dict(torch.load("./model/dog_breed_classification.pth"))
model.eval()

app = FastAPI()

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def fetch_predictions(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(file.file.read()))
    img = DogBreedImageDataset(img,transform=transform)
    pred = torch.max(F.softmax(model.forward(torch.unsqueeze(img[0],dim=0))),dim=1)
    score = pred.values.tolist()[0]
    idx = pred.indices.tolist()[0]
    return {"breed":dog_breeds[idx],"prob":score} 

