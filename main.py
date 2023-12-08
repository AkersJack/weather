import torch 
from torch import nn 
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.models import resnet50, resnet101
import torch.optim as optim 
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd
import os 
from skimage import io, transform 
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
from torchvision import transforms, utils
from torchvision import models
from torch.nn import functional as F
from skimage.transform import resize
import json
import csv 

def show_images(image):
    plt.imshow(image)




# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         super().__init__()
#         self.root_dir = root_dir
#         self.data = pd.read_csv(csv_file)
#         self.transform = transform
        

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.data.iloc[idx, 5])
#         # print(f"Image name")
#         image = io.imread(img_name)
#         data = self.data.iloc[idx, 0:5]
#         data = np.array([data], dtype=float).reshape(-1, 5)
#         # print(f"Data shape: {data.shape}") 
#         sample = {'image':image, 'data':data}
#         # print(f"Sample: {sample}")
#         # labels = self.data[]
#         # sample = {"image":image, ""}
#         if self.transform: 
#             sample = self.transform(sample)
        
#         return sample 
#         # return image, data
    


#     def __len__(self):
#         return len(self.data)


# class ToTensor(object): 
    
#     def __call__(self, sample): 
#         image, data = sample['image'], sample['data']

#         image = image.transpose((2, 0, 1))

#         return {'image': torch.from_numpy(image), 
#                 'data': torch.from_numpy(data)}


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_path, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.labels = pd.read_csv(csv_file) 

    
    def __len__(self): 
        return len(self.labels)
    
    def __getitem__(self, index): 
        location = "./Data_tree_full.csv"
        tree = pd.read_csv(location)
        with open("data_new.json", 'r') as f: 
            data = json.load(f)
        tree["time"][index]
        for x in tree["time"]:
            if x in data: 
                if data[x]["picture"] != None:
                    image_path = os.path.join("./weather_used/", data[x]["picture"])
        


        # print(f"image_path")
        image_path = os.path.join(self.data_path, self.labels.iloc[index, 5])
        image = io.imread(image_path)
        # new_width = int(image.shape[1] * 0.39) # 228
        # new_height = int(image.shape[0] * 0.39) # 184
        # new_width = 224
        # new_height = 224
        # print(f"New width: {new_width}")
        # print(f"New height: {new_height}")
        # image = resize(image, (new_height, new_width), anti_aliasing=True)
        # image = transform.resize(image, (224, 224))
        # image = transform.resize(image, (new_height, new_width), antialias=True)
        # image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        image = image.transpose((2, 0, 1)) # Go from H, W, C to C, H, W which the network expects

        # image = Image.open(image_path)

        if self.transform: 
            image = self.transform(image)
        
        label = self.labels.iloc[index, 0:5]
        # label = np.array(label, dtype=float).reshape(-1, 5)
        
        
        # Convert image and label to tensors 
        image = torch.from_numpy(image).float()
        original_w, original_h = image.shape[1], image.shape[2]
        aspect_ratio = original_w / original_h
        target_size = (224, 224)
        if target_size[0] / target_size[1] > aspect_ratio:
            # Width is larger than the desired aspect ratio
            padded_width = target_size[0]
            padded_height = int(padded_width / aspect_ratio)
        else:
            # Height is larger than the desired aspect ratio
            padded_height = target_size[1]
            padded_width = int(padded_height * aspect_ratio)

        h_padding = (padded_width - original_w) // 2
        v_padding = (padded_height - original_h) // 2
        padding = (h_padding, v_padding, h_padding, v_padding)
        padded_img_tensor = transforms.Pad(padding, padding_mode='reflect')(image)
        image = transforms.Resize(target_size, antialias=True)(padded_img_tensor)

        label = torch.tensor(label).float()
        # image = transforms.Normalize(mean=[0.0081, 0.0132, 0.0119], std=[0.0081, 0.0132, 0.0119])(image)
        # image = transforms.Normalize(mean=[0.0163, 0.0265, 0.0239], std=[0.0163, 0.0265, 0.0239])(image)
        #image = transforms.Normalize(mean=[4.2011, 6.8036, 6.2111], std=[4.2011, 6.8036, 6.2111])(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
        # label = F.one_hot(label, num_classes=5)
        # print(f"Label: {label}")

        return image, label


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # According to the PyTorch Documentation:
        # "if a nn.Conv2d layer is directly followed by a nn.BatchNorm2d layer, then the bias 
        # in the convolution is not needed. The first step of BatchNorm subtracts the mean, which
        # effectively cancels out the bias effect"
        # "This applies to 1d and 3d convolutions as long as BatchNorm (or other normalization layer)
        # normalizes on the same dimension as convolution's bias"
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.LazyLinear(out_features=256)
        self.fc2 = nn.LazyLinear(out_features=128)
        self.fc3 = nn.LazyLinear(out_features=5)
        self.flat = nn.Flatten()



    def forward(self, x): 
        x = self.pool(self.bn1(torch.relu(self.conv1(x)))) 
        x = self.pool(self.bn2(torch.relu(self.conv2(x)))) 
        x = self.pool(self.bn3(torch.relu(self.conv3(x)))) 
        
        x = self.pool(self.bn4(torch.relu(self.conv4(x))))
        x = self.flat(x)
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x)) 
        x = self.fc3(x) 
        return x

model = resnet50(weights=None)
# # model = resnet50()
# model = resnet101(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        

        img = transform.resize(image, (new_h, new_w))
        # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        # h and w are swapped for data because for images,
        # x and y axes are axis 1 and 0 respectively
        
        return img



def image(image_path):

        image = io.imread(image_path)
        # new_width = int(image.shape[1] * 0.39) # 228
        # new_height = int(image.shape[0] * 0.39) # 184
        # new_width = 224
        # new_height = 224
        # print(f"New width: {new_width}")
        # print(f"New height: {new_height}")
        # image = resize(image, (new_height, new_width), anti_aliasing=True)
        # image = transform.resize(image, (224, 224))
        # image = transform.resize(image, (new_height, new_width), antialias=True)
        # image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        image = image.transpose((2, 0, 1)) # Go from H, W, C to C, H, W which the network expects

        # image = Image.open(image_path)

        # label = np.array(label, dtype=float).reshape(-1, 5)
        
        
        # Convert image and label to tensors 
        image = torch.from_numpy(image).float()
        original_w, original_h = image.shape[1], image.shape[2]
        aspect_ratio = original_w / original_h
        target_size = (224, 224)
        if target_size[0] / target_size[1] > aspect_ratio:
            # Width is larger than the desired aspect ratio
            padded_width = target_size[0]
            padded_height = int(padded_width / aspect_ratio)
        else:
            # Height is larger than the desired aspect ratio
            padded_height = target_size[1]
            padded_width = int(padded_height * aspect_ratio)

        h_padding = (padded_width - original_w) // 2
        v_padding = (padded_height - original_h) // 2
        padding = (h_padding, v_padding, h_padding, v_padding)
        padded_img_tensor = transforms.Pad(padding, padding_mode='reflect')(image)
        image = transforms.Resize(target_size, antialias=True)(padded_img_tensor)

        # image = transforms.Normalize(mean=[0.0081, 0.0132, 0.0119], std=[0.0081, 0.0132, 0.0119])(image)
        # image = transforms.Normalize(mean=[0.0163, 0.0265, 0.0239], std=[0.0163, 0.0265, 0.0239])(image)
        #image = transforms.Normalize(mean=[4.2011, 6.8036, 6.2111], std=[4.2011, 6.8036, 6.2111])(image)
        # image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        # image = transforms.Normalize(mean=[0.0081, 0.0132, 0.0119], std=[0.0081, 0.0132, 0.0119])(image)
        # image = transforms.Normalize(mean=[0.0163, 0.0265, 0.0239], std=[0.0163, 0.0265, 0.0239])(image)
        #image = transforms.Normalize(mean=[4.2011, 6.8036, 6.2111], std=[4.2011, 6.8036, 6.2111])(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        image = image.unsqueeze(0)

        return image


# 1424 images train
# 357 images validate

def getData():
    mean_sum = 0.0
    std_sum = 0.0
    total_samples = 0

    # composed = transforms.Compose([
    # ])
    data = MyDataset("/home/jack/Documents/UMBC/cmsc478/project/Data.csv", "./weather_used/", transform=None)
    # Train on 80% of the dataset
    train_size = int(0.8 * len(data)) 

    # Validate on the remaining 20% of the dataset
    val_size = len(data) - train_size

    # Randomly split the dataset into training and testing data

    # all_data = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=4)


    # for  batch, (images, labels) in enumerate(all_data):
    #     print(f"Image shape: ", images.shape) #[32, 3, 244, 244]

    #     mean_sum += torch.mean(images, dim=(0, 2, 3)) # 0, 2, 3
    #     std_sum += torch.mean(images, dim=(0, 2, 3))
    #     total_samples += images.size(0)



    # overall_mean = mean_sum / total_samples
    # overall_std = std_sum / total_samples

    # print(f"Calculated Mean: ", overall_mean)
    # print(f"Calculated Standard Deviation: ", overall_std)
        
    data_full = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=4)
        
    
    train_dataset, val_dataset = random_split(data, [train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # print(f"Datset Size: {len(data)}")
    # print(f"training Data Size: {len(train_loader.dataset)}")
    # print(f"Validation Data Size: {len(val_loader.dataset)}")

    # return train_loader, val_loader
    # return train_dataset, val_dataset
    return data_full


def test(): 
    model.eval()
    n_correct = 0 
    n_incorrect = 0
    test_loss = 0
    with torch.no_grad():
        for images, labels in val_loader: 
            images, labels = images.to(device), labels.to(device)
            
            # Forward Pass
            output = model(images)
            
            test_loss += lossFunc_test(output, labels).item()


            output.data = torch.sigmoid(output) # Convert output data to between 0 and 1

            print(f"Predictions: {np.round(output.numpy(), decimals=3)}")

            # predictions = (output > 0.9).int()
            predictions = (output > 0.5).int()
            
            # print(f"Predictions: {predictions}")
            # print(f"True labels: {labels}")

            n_correct += (predictions == labels).all(dim=1).sum()
            n_incorrect += (predictions != labels).all(dim=1).sum()
             


    # output.data = torch.mul(100, output) # conver the data / confidence into %
    test_loss /= len(val_loader.dataset)
    print("\nTest set: Avg. loss: {:.4f} Accuracy: {}/{} ({:.2f}%)\n".format(
        test_loss, n_correct, len(val_loader.dataset), 
        100. * (n_correct / len(val_loader.dataset))))
            




if __name__ == '__main__':
    num_epochs = 1000
    batch_size = 1
    learning_rate = 0.001
    log_interval = 128
    num_classes = 5

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(f"Using device: {device}")
    
    
    # train_loader, val_loader = getData()
    # data = getData()
    # data
    
    # show_images(io.imread(os.path.join("D:/weather/", "2021-11-08.02_55_59.png")))
    # plt.show()1


    # data = MyDataset("D:/project/data.csv", "D:/weather/", transform=composed)



    # for x in range(data.__len__())  :
    #     print(f"X: {x} Data size (imge): ", (data.__getitem__(x))['image'].size()[2])
    #     item = data.__getitem__(x)["image"].size()
    #     if item[2] != 631: 
    #         print(f"Item: {item}")
        # print(f"X: {x} Data size: (data)", (data.__getitem__(x))['data'].size())

    # criterion = nn.CrossEntropyLoss()
    #model = Net() # Testing with custom CNN (normally using resnet 50)

    
    
    # criterion = nn.MultiLabelSoftMarginLoss()
    lossFunc_test = nn.MultiLabelSoftMarginLoss(reduction="sum")
    # criterion = nn.BCELoss()
    # lossFunc_test = nn.BCELoss(reduction="sum")

    # criterion = nn.BCEWithLogitsLoss()
    # lossFunc_test = nn.BCEWithLogitsLoss(reduction="sum")

    
    
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
    #                                                 max_lr = 0.01, 
    #                                                 steps_per_epoch=len(train_loader),
    #                                                 epochs=num_epochs
    #                                                 )

    
    # location = "./Data_tree_full.csv"
    # tree = pd.read_csv(location)
    # with open("data_new.json", 'r') as f: 
    #     data = json.load(f)
    

    # count = 0
    # for x in tree["time"]: 
    #     if x in data: 
    #         # print(data[x]["picture"])
    #         if data[x]["picture"] != None:
    #             loc = os.path.join("./weather_used/", data[x]["picture"])
    #             count += 1 
    #             img = image(loc)
    #             output = model([img])
    #             print(output)
    
    # print(count)
        
    
    


    # output = model()
                                                    
                                                    
                                                    
    
    # scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1, patience=3, verbose=True)
                                  

                                  


    # print("Initial Test: ")
    # test()
    # for epoch in range(num_epochs):
    #     running_loss = 0.0
    #     model.train()
    #     # for inputs, sample in enumerate(data_loader):
    #         # print(i, sample['image'].size(), sample['data'].size()) 
    #     for batch, (images, labels) in enumerate(train_loader):
    #         images, labels = images.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         # print(f"Image shape: {images.shape}")
    #         # print(f"label shape: {labels.shape}")
    #         # print(f"label: {label}")
    #         # print(f"Type of sample: {data_input.dtype}")
    #         outputs = model(images)
    #         # loss = criterion(outputs, sample['data'])
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
    #         running_loss += loss.item()

    #         # print(f"Len images: {len(images)}")
    #         # print(f"len trainloader: {len(train_loader.dataset)}")
    #         # print(f"Batch: {batch}")

    #         if batch % log_interval == 0: 
    #             print('Epoch [{}/{}], Step [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             epoch + 1, num_epochs, (batch * len(images)), len(train_loader.dataset),
    #             100. * (batch / len(train_loader)), loss.item()))
            
        
    #     torch.save(model.state_dict(), './model.pth')
    #     torch.save(optimizer.state_dict(), './optimizer.pth')
    #     average_loss = running_loss / len(train_loader)
    #     # scheduler.step(average_loss)
    #     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")
    #     if epoch % 5 == 0: 
    #         test()
    
    # test()
    # print("Training Finished")
    # torch.save(model, "./model_FULL.pth")

    

    model = torch.load("model_FULL_28.pth")
    model.to(device)
    model.eval()
    location = "./Data_tree_full.csv"
    tree = pd.read_csv(location)
    with open("data_new.json", 'r') as f: 
        data = json.load(f)
    

    count = 0
    row = []
    # predictions = open("Predict", "w")
    for i, x in enumerate(tree["time"]): 
        if x in data: 

            # print(data[x]["picture"])
            if data[x]["picture"] != None:
                loc = os.path.join("./weather_used/", data[x]["picture"])
                count += 1 
                img = image(loc)
                # print(img)
                img = img.to(device)
                output = model(img)

                output.data = torch.sigmoid(output) # Convert output data to between 0 and 1
                out = output.cpu()
                prediction = np.round(out.detach().numpy(),decimals=3)[0]
                # print(prediction)
                # predictions.write(str(prediction.tolist()) + "\n")
                # predictions.write(str(prediction.tolist()))
                # predictions.write("\n")
                # print(f"Predictions: {prediction}")
                # cpu = output.to('cpu')
                # row.append(out)

            else: 
                print(tree["precipitation"][i])
                s = pd.Series(tree["precipitation"][i])
                # predictions.write(str(s.tolist()[0]) + "\n")
                # row.append(tree["precipitation"][i])

    # tree.insert((len(tree.columns), "cnn"), row)
    # predictions.close()
    print(row)

        
    # output.data = torch.sigmoid(output) # Convert output data to between 0 and 1
