import os
import copy
import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from network import model_modify

root_dir = './traffic-sign/'
TRAIN = root_dir + 'train'
TSET = root_dir + 'test'
train_csv_file = root_dir + 'train_label.csv'
test_csv_file = root_dir + 'test_label.csv'
NUM_EPOCHS = 50

class MyDataset():

    def __init__(self, root_dir, data_folder, data_csv_file, transform = None):
        self.root_dir = root_dir
        self.data_folder = data_folder
        self.data_csv_file = data_csv_file
        self.transform = transform
        if not os.path.exists(data_folder):
            print(self.data_folder + ' does not exist!')
        self.classes_sample_counts = []
        for sub_folder in os.listdir(self.data_folder):
            sub_folder_path = os.path.join(self.data_folder, sub_folder)
            if not os.path.exists(sub_folder_path):
                print(sub_folder_path + ' does not exist!')
            num = len(os.listdir(sub_folder_path))
            self.classes_sample_counts.append(num)
        if not os.path.isfile(data_csv_file):
            print(self.data_csv_file + " dose not exist!")
        self.file_info = pd.read_csv(data_csv_file, index_col = 0)
        self.classes_labels = self.file_info['class_id'].tolist()

    def __len__(self):
        return len(self.file_info)

    def __getitem__(self, idx):
        image_path = self.file_info['image_location'][idx]
        if not os.path.isfile(image_path):
            print(image_path + ' does not exit!')
            return None
        
        image = Image.open(image_path).convert('RGB')
        image_label = self.file_info['class_id'][idx]

        sample = {'image': image, 'class': image_label}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample
    
    def get_classes_sample_counts(self):
        return self.classes_sample_counts

    def get_weight(self):
        return 1./torch.tensor(self.classes_sample_counts, dtype = torch.float)
    
    def get_classes_labels(self):
        return self.classes_labels
    
    def get_samples_weights(self):
        return self.get_weight()[self.classes_labels]

train_transforms = transforms.Compose([transforms.Resize([200,200]),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()
                                    ])
train_dataset = MyDataset(root_dir, TRAIN, train_csv_file, transform = train_transforms)
train_sampler = WeightedRandomSampler(weights = train_dataset.get_samples_weights(), num_samples = len(train_dataset), replacement=True)
train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = False, sampler = train_sampler)

test_transforms = transforms.Compose([transforms.Resize([200,200]),
                                       transforms.ToTensor()
                                    ])
test_dataset = MyDataset(root_dir, TSET, test_csv_file, test_transforms)
test_loader = DataLoader(dataset = test_dataset)
data_loader = {"train":train_loader, "val":test_loader}

device = torch.device("cpu" if not torch.cuda.is_available() else 'gpu:0')
print(device)

def visualize_dataset():
    print(len(train_dataset))
    idx = random.randint(0, len(train_dataset))
    sample = train_loader.dataset[idx]
    print(idx, sample['image'].shape, sample['class'])
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()
visualize_dataset()

def train_model(model, crierion, optimizer, scheduler, num_epochs = 50):
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_classes = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}/{num_epochs-1}')
        print('*'*10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.eval()
            else:
                model.eval()
            
            running_loss = 0.0
            corrects_classes = 0

            for idx, data in enumerate(data_loader[phase]):
                print(phase + f' processing: {idx} th batch')
                inputs = data['image'].to(device)
                labels_classes = data['class'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    x_classes = model(inputs)

                    x_classes = x_classes.view(-1,62)

                    _, preds_classes = torch.max(x_classes, 1)

                    loss = crierion(x_classes, labels_classes)
                    if phase == 'trian':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                
                corrects_classes += torch.sum(labels_classes == preds_classes)
            
            epoch_loss = running_loss/len(data_loader[phase].dataset)
            Loss_list[phase].append(epoch_loss)

            epoch_acc = corrects_classes.double()/len(data_loader[phase].dataset)
            
            Accuracy_list_classes[phase].append(100 * epoch_acc)

            print(f'{phase} Loss : {epoch_loss:.3f} Acc: {epoch_acc}')

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'Best val acc : {best_acc:.2f}')
        scheduler.step()
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts.stete_dict(), 'best_model.pt')
    return model, Loss_list, Accuracy_list_classes    

model = model_modify()
model.to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma=0.1)
model, Loss_list, Accuracy_list = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

x = range(0, NUM_EPOCHS)
y1 = Loss_list['train']
y2 = Loss_list['val']
plt.plot(x, y1, c = 'r', linestyle = '-', marker = '*', linewidth = 1, lable = 'train')
plt.plot(x, y2, c = 'b', linestyle = '-', marker = '*', linewidth = 1, lable = 'val')
plt.title('trian and val loss vs epochs')
plt.ylabel('loss')
plt.savefig('train and val loss vs epochs')
plt.close(None)

y3 = Loss_list['train']
y4 = Loss_list['val']
plt.plot(x, y3, c = 'r', linestyle = '-', marker = '*', linewidth = 1, lable = 'train')
plt.plot(x, y4, c = 'b', linestyle = '-', marker = '*', linewidth = 1, lable = 'val')
plt.title('trian and val acc vs epochs')
plt.ylabel('acc')
plt.savefig('train and val acc vs epochs')
plt.close(None)

