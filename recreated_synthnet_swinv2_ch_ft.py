import os
import shutil
import copy
import random
from glob import glob
import sys

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.swin_transformer import Swin_V2_T_Weights, Swin_V2_B_Weights
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.init as init
from torchvision.transforms import ToTensor, ToPILImage

from transformers import set_seed, AutoImageProcessor, AutoModelForImageClassification, get_cosine_schedule_with_warmup
from PIL import Image

from tqdm import tqdm

import numpy as np

import pytorch_lightning as pl


# Specify Seed, Device and Transformer Cache

os.environ["TRANSFORMERS_CACHE"] = "/mnt/data/transformers_cache"

pl.seed_everything(42, workers=True)

DEVICE = torch.device("cuda:0")


# Helper Functions

def get_processor_transforms_batch_size_ch():
    #processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
    #transform_train = None
    #transform_val = None
    processor = None
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=192, scale=(0.7, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    transform_val = transforms.Compose([
        transforms.Resize(size=(192,192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    batch_size = 32
    return processor, transform_train, transform_val, batch_size


def get_processor_transforms_batch_size_ft():
    #processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
    #transform_train = None
    #transform_val = None
    processor = None
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=192, scale=(0.7, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.AugMix(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    transform_val = transforms.Compose([
        transforms.Resize(size=(192,192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    batch_size = 32
    return processor, transform_train, transform_val, batch_size


class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None, processor=None):
        self.folder_path = folder_path
        self.transform = transform
        self.processor = processor

        self.image_paths = []
        self.labels = []
        self.label_mapping = {}
        self.reverse_label_mapping = {}

        class_dirs = sorted([d for d in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, d))])
        for label, class_dir in enumerate(class_dirs):
            self.label_mapping[class_dir] = label
            self.reverse_label_mapping[label] = class_dir

            class_path = os.path.join(self.folder_path, class_dir)
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    image_path = os.path.join(class_path, filename)
                    self.image_paths.append(image_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.processor:
            image = self.processor(images=image, return_tensors="pt", do_normalize=True, do_rescale=True, do_resize=True)
            inputs = {'pixel_values': image['pixel_values'][0], 'labels': torch.tensor(label)}
            return inputs
        inputs = {'pixel_values': image, 'labels': torch.tensor(label)}
        return inputs

    def get_class_name(self, label):
        return self.reverse_label_mapping[label]


def get_hyperparameters_ch(model):
    num_epochs = 20
    loss_function = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-3
    momentum = 0.9
    weight_decay = 0.0
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = None
    return num_epochs, loss_function, optimizer, scheduler


def get_hyperparameters_ft(model):
    num_epochs = 20
    loss_function = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-5
    weight_decay = 0.01
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_epochs*0.1, num_training_steps=20)
    #total_steps = len(train_loader_rgb) * num_epochs
    #scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_epochs*0.1, num_training_steps=total_steps)
    return num_epochs, loss_function, optimizer, scheduler


def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_classifier(model):
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def unfreeze_stage(model, index):
    stage = model.swinv2.encoder.layers[index]
    for param in stage.parameters():
        param.requires_grad = True
    return model


def training(model, model_name, num_epochs, loss_function, optimizer, scheduler, train_dataloader, val_dataloader):
    train_loss_array = []
    train_acc_array = []
    val_loss_array = []
    val_acc_array = []
    lowest_val_loss = np.inf
    best_model = None

    for epoch in tqdm(range(num_epochs), file=sys.stdout, dynamic_ncols=True):

        if scheduler:
            print('Epoch: {} | Learning rate: {}'.format(epoch + 1, scheduler.get_last_lr()), flush=True)
        else:
            print('Epoch: {}'.format(epoch + 1), flush=True)

        for phase in ['train', 'val']:

            epoch_loss = 0
            epoch_correct_items = 0
            epoch_items = 0

            if phase == 'train':
                model.train()
                with torch.enable_grad():
                    for batch in train_dataloader:
                        inputs = batch['pixel_values'].to(DEVICE)
                        targets = batch['labels'].to(DEVICE)

                        optimizer.zero_grad()
                        outputs = model(inputs)
                        logits = outputs.logits
                        loss = loss_function(logits, targets)

                        loss.backward()
                        optimizer.step()

                        if scheduler:
                            scheduler.step()

                        preds = logits.argmax(dim=1)
                        correct_items = (preds == targets).sum().item()

                        epoch_loss += loss.item()
                        epoch_correct_items += correct_items
                        epoch_items += len(targets)

                train_loss_array.append(epoch_loss / epoch_items)
                train_acc_array.append(epoch_correct_items / epoch_items)


            elif phase == 'val':
                model.eval()
                with torch.no_grad():
                    for batch in val_dataloader:
                        inputs = batch['pixel_values'].to(DEVICE)
                        targets = batch['labels'].to(DEVICE)

                        outputs = model(inputs)
                        logits = outputs.logits
                        loss = loss_function(logits, targets)

                        preds = logits.argmax(dim=1)
                        correct_items = (preds == targets).sum().item()

                        epoch_loss += loss.item()
                        epoch_correct_items += correct_items
                        epoch_items += len(targets)

                val_loss_array.append(epoch_loss / epoch_items)
                val_acc_array.append(epoch_correct_items / epoch_items)

                if epoch_loss / epoch_items < lowest_val_loss:
                    lowest_val_loss = epoch_loss / epoch_items
                    torch.save(model.state_dict(), '{}_weights.pth'.format(model_name))
                    best_model = copy.deepcopy(model)
                    print("\t| New lowest val loss for {}: {}".format(model_name, lowest_val_loss), flush=True)

                print("\t| Val accuracy for {}: {:.4f}".format(model_name, epoch_correct_items / epoch_items), flush=True)

    return best_model, train_loss_array, train_acc_array, val_loss_array, val_acc_array


def test_model(model, test_dataloader):
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch['pixel_values'].to(DEVICE)
            targets = batch['labels'].to(DEVICE)

            outputs = model(inputs)
            logits = outputs.logits

            all_outputs.append(logits)
            all_labels.append(targets)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_outputs, all_labels


def compute_accuracy(outputs, labels):
    _, predictions = outputs.max(1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

def print_train_test_results(val_loss_array, val_acc_array, test_accuracy, run):
    min_loss = min(val_loss_array)
    min_loss_epoch = val_loss_array.index(min_loss)
    min_loss_accuracy = val_acc_array[min_loss_epoch]
    print(f"\nResults ({run}):")
    print("\tMin val loss {:.4f} was achieved during epoch #{}".format(min_loss, min_loss_epoch + 1))
    print("\tVal Accuracy during min val loss is {:.4f}".format(min_loss_accuracy))
    print(f"Test Accuracy: {test_accuracy:.2f}%")


# Format data for processeing (CH)

processor, transform_train, transform_val, batch_size = get_processor_transforms_batch_size_ch()

train_dataset_rgb = CustomImageDataset(folder_path='/mnt/data/data_rgb/train', transform=transform_train, processor=processor)
val_dataset_rgb = CustomImageDataset(folder_path='/mnt/data/data_rgb/val', transform=transform_val, processor=processor)
test_dataset_rgb = CustomImageDataset(folder_path='/mnt/data/data_rgb/test', transform=transform_val, processor=processor)

train_loader_rgb = DataLoader(train_dataset_rgb, batch_size=batch_size, shuffle=True)
val_loader_rgb = DataLoader(val_dataset_rgb, batch_size=batch_size, shuffle=False)
test_loader_rgb = DataLoader(test_dataset_rgb, batch_size=batch_size, shuffle=False)


# Load Model

swin2_standard_model_original = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k", num_labels=102, ignore_mismatched_sizes=True)


# Freeze all Layers except Classification Head

swin2_standard_model_original = freeze_all_layers(swin2_standard_model_original)

swin2_standard_model_original = unfreeze_classifier(swin2_standard_model_original)

swin2_standard_model_original = swin2_standard_model_original.to(DEVICE)


# Get Hyperparameters for Classification Head Training

num_epochs_ch, loss_function_ch, optimizer_ch, scheduler_ch = get_hyperparameters_ch(swin2_standard_model_original)


# Train Model (CH)

print("*"*30)
print("Classification Head Training:")

swin2_standard_model_ch_training_results = training(model=swin2_standard_model_original,
                                        model_name='swin2_standard_model_ch',
                                        num_epochs=num_epochs_ch,
                                        loss_function=loss_function_ch,
                                        optimizer=optimizer_ch,
                                        scheduler=scheduler_ch,
                                        train_dataloader=train_loader_rgb,
                                        val_dataloader=val_loader_rgb)


# Results (CH)

swin2_standard_model_ch, train_loss_array, train_acc_array, val_loss_array, val_acc_array = swin2_standard_model_ch_training_results

all_outputs, all_labels = test_model(swin2_standard_model_ch, test_loader_rgb)

test_accuracy = compute_accuracy(all_outputs, all_labels)

print_train_test_results(val_loss_array, val_acc_array, test_accuracy, "CH")


# Format data for processeing (FT)

processor, transform_train, transform_val, batch_size = get_processor_transforms_batch_size_ft()

train_dataset_rgb = CustomImageDataset(folder_path='/mnt/data/data_rgb/train', transform=transform_train, processor=processor)
val_dataset_rgb = CustomImageDataset(folder_path='/mnt/data/data_rgb/val', transform=transform_val, processor=processor)
test_dataset_rgb = CustomImageDataset(folder_path='/mnt/data/data_rgb/test', transform=transform_val, processor=processor)

train_loader_rgb = DataLoader(train_dataset_rgb, batch_size=batch_size, shuffle=True)
val_loader_rgb = DataLoader(val_dataset_rgb, batch_size=batch_size, shuffle=False)
test_loader_rgb = DataLoader(test_dataset_rgb, batch_size=batch_size, shuffle=False)


# Unfreeze all 4 stages of the SwinV2 Model

swin2_standard_model_ft = copy.deepcopy(swin2_standard_model_ch)

swin2_standard_model_ft = freeze_all_layers(swin2_standard_model_ft)

swin2_standard_model_ft = unfreeze_classifier(swin2_standard_model_ft)

swin2_standard_model_ft = unfreeze_stage(swin2_standard_model_ft, 3)

swin2_standard_model_ft = unfreeze_stage(swin2_standard_model_ft, 2)

swin2_standard_model_ft = unfreeze_stage(swin2_standard_model_ft, 1)

swin2_standard_model_ft = unfreeze_stage(swin2_standard_model_ft, 0)

swin2_standard_model_ft = swin2_standard_model_ft.to(DEVICE)


# Get Hyperparameters for Full Training

num_epochs_ft, loss_function_ft, optimizer_ft, scheduler_ft = get_hyperparameters_ft(swin2_standard_model_ft)


# Train Model (FT)

print("*"*30)
print("Full Model Training")

swin2_standard_model_ft_training_results = training(model=swin2_standard_model_ft,
                                        model_name='swin2_standard_model_ft',
                                        num_epochs=num_epochs_ft,
                                        loss_function=loss_function_ft,
                                        optimizer=optimizer_ft,
                                        scheduler=scheduler_ft,
                                        train_dataloader=train_loader_rgb,
                                        val_dataloader=val_loader_rgb)


# Results (FT)

swin2_standard_model_ft, train_loss_array, train_acc_array, val_loss_array, val_acc_array = swin2_standard_model_ft_training_results

all_outputs, all_labels = test_model(swin2_standard_model_ft, test_loader_rgb)

test_accuracy = compute_accuracy(all_outputs, all_labels)

print_train_test_results(val_loss_array, val_acc_array, test_accuracy, "FT")


# Delete Transformers Cache

shutil.rmtree("/mnt/data/transformers_cache", ignore_errors=True)
