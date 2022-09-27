import os
from glob import glob
from torchtoolbox.transform import Cutout
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
imgClasses = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']


class NIH_Dataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file = self.data.iloc[:,0].iloc[idx]
        img = Image.open(img_file).convert('RGB')
        label = np.array(self.data.iloc[:,1].iloc[idx]).astype(float)

        if self.transform:
            img = self.transform(img)
        return img,label


train_data_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=(-20, +20)),
    T.RandomAffine(translate=(0.15, 0.15), degrees=0),
    T.RandomGrayscale(),
    T.Resize(size=(300, 300)),
    Cutout(),
    T.RandomResizedCrop(256),
    T.ToTensor()
])
valid_data_transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor()
])


INPUT_DIR = 'D:/data/NIH_Chest_X_rays/'
imgClasses = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
deletClasses = ['Consolidation','Emphysema','Hernia','Fibrosis','Pleural_Thickening','Edema']
def processDF(df, topLabels=imgClasses):
    # create a field which houses full path to the images
    allImagesGlob = glob(f'{INPUT_DIR}images*/images/*.png')
    allImagesPathDict = {os.path.basename(x): x for x in allImagesGlob}
    df['path'] = df['Image Index'].map(allImagesPathDict.get)

    for label in topLabels:
        df[label] = df['Finding Labels'].map(lambda x: 1.0 if label in x else 0.0)
    df['finalLabel'] = df.apply(lambda x: [x[topLabels].values], 1).map(lambda x: x[0])

    # topLables
    tempdf = df['Finding Labels']

    # drop not req columns
    dropLabels = ['Finding Labels', 'Image Index', 'Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender',
                  'View Position', 'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]', 'Unnamed: 11']
    df.drop(columns=dropLabels, inplace=True)
    df.drop(columns=topLabels, inplace=True)
    # add
    df = pd.concat([df, tempdf], axis=1)

    return df



