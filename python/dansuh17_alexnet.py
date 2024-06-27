import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

import torchvision.datasets as datasets
import torchvision.transforms as transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227
NUM_CLASSES = 1000
DEVICE_IDS = [0, 1, 2, 3]
INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
OUTPUT_DIR = 'alexnext_data_out'
CHECKPOINT_DIR = OUTPUT_DIR + '/models'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU()
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net(12).bias, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(0, 256 * 6 * 6)
        return self.classifier(x)
    

seed = torch.initial_seed()
print(f"[INFO] Current seed : {seed}.")

alexnet = AlexNet(num_classes=NUM_CLASSES).to(DEVICE)
alextnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)

print(f"[INFO] AlexNet created : {alexnet}.")

dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
    transforms.CenterCrop(IMAGE_DIM),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
]))
print("[INFO] Dataset created.")

dataloader = data.DataLoader(
    dataset,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True,
    batch_size=BATCH_SIZE
)
print("[INFO] Dataloader created.")

optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
print("[INFO] Optimizer created.")

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
print("[INFO] LR Scheduler created.")

print("[INFO] Training start...")

total_steps = 1
for epoch in range(NUM_EPOCHS):
    lr_scheduler.step()
    for imgs, classes in dataloader:
        imgs, classes = imgs.to(DEVICE), classes.to(DEVICE)

        # Calculate the loss
        output = alexnet(imgs)
        loss = F.cross_entropy(output, classes)

        # Update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print out gradient values and parameter average values
        if total_steps % 100 == 0:
            with torch.no_grad():
                print("*" * 10)
                for name, parameter in alexnet.named_parameters():
                    if parameter.grad is not None:
                        avg_grad = torch.mean(parameter.grad)
                        print(f"\t{name} - grad_avg: {avg_grad}")
                    if parameter.data is not None:
                        avg_weight = torch.mean(parameter.data)
                        print(f"\t{name} - param_avg: {avg_weight}")
        total_steps += 1

    # Save checkpoints
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"alexnet_states_e{epoch + 1}.pkl")
    state = {
        "epoch": epoch,
        "total_steps": total_steps,
        "optimizer" : optimizer,
        "model": alexnet.state_dict(),
        "seed": seed
    }
    torch.save(state, checkpoint_path)