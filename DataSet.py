from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image

class MyDataSet(Dataset):
    def __init__(self, root, train, transformer=None):
        super().__init__()
        self.data_path = ""
        self.transformer = transformer
        if train:
            self.data_path = os.path.join(root, "train")    
        else:
            self.data_path = os.path.join(root, "test")

        self.class_names = ["pizza", "steak", "sushi"]
        self.img = []
        self.label = []
        for iter, category in enumerate(self.class_names):
            category_path = os.path.join(self.data_path, category)
            for item in os.listdir(category_path):
                img_path = os.path.join(category_path, item)

                self.img.append(img_path)
                self.label.append(iter)

        pass

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img_path = self.img[item]
        img = Image.open(img_path).convert("RGB")
        if self.transformer:
            img = self.transformer(img)
        label = self.label[item]
        return img, label

if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((64, 64))
    ])
    data = MyDataSet(root="my_data", train=True, transformer=transform)
    # img, label = data[50]
    # print(type(img))

    data_loader = DataLoader(
        dataset=data,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )
    for images, labels in data_loader:
        print(images.shape)