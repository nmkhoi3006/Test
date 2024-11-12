
from DataSet import MyDataSet
from Model import CNNModel
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize
from sklearn.metrics import accuracy_score
from tqdm.autonotebook import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    log_path = "tensorboard"
    transforms = Compose([
        ToTensor(),
        Resize(size=(32, 32))
    ])
    data_train = MyDataSet(root="my_data", train=True, transformer=transforms)
    data_test = MyDataSet(root="my_data", train=False, transformer=transforms)


    dataloader_train = DataLoader(
        dataset=data_train,
        batch_size=2,
        num_workers=0,
        drop_last=False
    )
    dataloader_test = DataLoader(
        dataset=data_test,
        batch_size=2,
        num_workers=0,
        drop_last=False
    )

    model = CNNModel(num_class=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2)
    num_epochs = 100
    num_iter = int(len(data_train)/32)
    for epoch in range(1, num_epochs+1):
        #TRAIN
        # progress_bar = tqdm(dataloader_train, colour="blue")
        print("TRAIN")
        model.train()
        for iter, (img, label) in enumerate(dataloader_train):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            prediction = model(img)
            loss = criterion(prediction, label)

            print(loss.item())
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # progress_bar.set_description(f"Epoch: {epoch}/{num_epochs} | Loss: {loss:.4f}")

        #VAL
        all_losses = []
        all_labels = []
        all_predictions = []
        print("VALIDATION")
        model.eval()
        with torch.inference_mode():
            for img, label in (dataloader_test):
                img = img.to(device)
                label = label.to(device)
                prediction = model(img)
                _, index = torch.max(prediction, dim=1)
                all_predictions.extend(index.tolist())
                all_labels.extend(label.tolist())

                loss = criterion(prediction, label)
                all_losses.append(loss)

        acc = accuracy_score(all_labels, all_predictions)
        loss = np.mean(all_losses)
        print(f"Accuracy_Validation: {acc} | Loss_Validation: {loss}")




if __name__ == '__main__':
    print(device)
    train()