import pandas as pd
import torch
from tqdm import tqdm

#INPUT_DIR = 'D:/data/NIH_Chest_X_rays/'
imgClasses = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
#deletClasses = ['Consolidation','Emphysema','Hernia','Fibrosis','Pleural_Thickening','Edema']
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def class_accuracy(dataloader, model):
    per_class_accuracy = [0 for i in range(len(imgClasses))]
    total = 0.0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            ps = model(images.to(device))
            labels = labels.to(device)
            ps =torch.sigmoid(ps)
            ps = (ps >= 0.5).float()

            for i in range(ps.shape[1]):
                x1 = ps[:, i:i+1]
                x2 = labels[:, i:i+1]
                per_class_accuracy[i] += int((x1 == x2).sum())

        per_class_accuracy = [(i/len(dataloader.dataset))*100.0 for i in per_class_accuracy]

    return per_class_accuracy


def get_acc_data(class_names, acc_list, outputpath):
    df = pd.DataFrame(list(zip(class_names, acc_list)), columns=['Labels', 'Acc'])
    df.to_csv(outputpath, sep=',', index=False, header=True)
    print("Class Accuracy Save")
    print(df)
    return df





