################################################################################################
#################################### LATE-FUSION GRIDSEARCH ####################################
################################################################################################

"""

"""

################################################################################################
########################################### IMPORTS ############################################
################################################################################################

import os
import shutil
import copy
import random
from glob import glob
import sys
import itertools

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
import torchvision.transforms.functional as TF

from transformers import set_seed, AutoImageProcessor, AutoModelForImageClassification, get_cosine_schedule_with_warmup
from PIL import Image, ImageOps, ImageEnhance

from tqdm import tqdm

import numpy as np

import pytorch_lightning as pl

import inspect


################################################################################################
########################################### SETTINGS ###########################################
################################################################################################

os.environ["TRANSFORMERS_CACHE"] = "/mnt/data/transformers_cache"

pl.seed_everything(42, workers=True)

DEVICE = torch.device("cuda:0")

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
    #total_steps = len(train_loader) * num_epochs
    #scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_epochs*0.1, num_training_steps=total_steps)
    return num_epochs, loss_function, optimizer, scheduler


################################################################################################
####################################### UTILITY FUNCTIONS ######################################
################################################################################################

class CustomTransforms:
    def __init__(self):
    	self.augmentation_list = [
    		self.identity,
    		self.autocontrast,
    		self.equalize,
    		self.rotate,
    		self.solarize,
    		self.color,
    		self.contrast,
    		self.brightness,
    		self.sharpness
    	]

    def int_parameter(self, level, maxval):
        return int(level * maxval / 10)
	
    def float_parameter(self, level, maxval):
        return float(level * maxval / 10)
	
    def identity(self, img, severity=1):
        return img
	
    def autocontrast(self, img, severity=1):
        return ImageOps.autocontrast(img)
	
    def equalize(self, img, severity=1):
        return ImageOps.equalize(img)
	
    def rotate(self, img, severity=1):
        degrees = self.int_parameter(severity, 30)
        if random.random() < 0.5:
            degrees = -degrees
        return img.rotate(degrees, resample=Image.BILINEAR)
	
    def solarize(self, img, severity=1):
        threshold = 256 - self.int_parameter(severity, 128)
        return ImageOps.solarize(img, threshold)
	
    def color(self, img, severity=1):
        factor = self.float_parameter(severity, 1.8) + 0.1
        return ImageEnhance.Color(img).enhance(factor)
	
    def contrast(self, img, severity=1):
        factor = self.float_parameter(severity, 1.8) + 0.1
        return ImageEnhance.Contrast(img).enhance(factor)
	
    def brightness(self, img, severity=1):
        factor = self.float_parameter(severity, 1.8) + 0.1
        return ImageEnhance.Brightness(img).enhance(factor)
	
    def sharpness(self, img, severity=1):
        factor = self.float_parameter(severity, 1.8) + 0.1
        return ImageEnhance.Sharpness(img).enhance(factor)
	
    def synchronized_augmix(self, rgb_image, depth_image, severity=3, width=3, depth=-1, alpha=1.0):
        ws = np.float32(np.random.dirichlet([alpha] * width))
        m = np.float32(np.random.beta(alpha, alpha))

        rgb_mix = np.zeros_like(np.array(rgb_image), dtype=np.float32)
        depth_mix = np.zeros_like(np.array(depth_image), dtype=np.float32)
	
        for i in range(width):
            rgb_image_aug = rgb_image.copy()
            depth_image_aug = depth_image.copy()

            depth_chain = depth if depth > 0 else np.random.randint(1, 4)
            for _ in range(depth_chain):
                op = np.random.choice(self.augmentation_list)
                rgb_image_aug = op(rgb_image_aug, severity)
                depth_image_aug = op(depth_image_aug, severity)
	
            rgb_mix += ws[i] * np.array(rgb_image_aug, dtype=np.float32)
            depth_mix += ws[i] * np.array(depth_image_aug, dtype=np.float32)
	
        rgb_mixed = (1 - m) * np.array(rgb_image, dtype=np.float32) + m * rgb_mix
        depth_mixed = (1 - m) * np.array(depth_image, dtype=np.float32) + m * depth_mix

        return Image.fromarray(np.uint8(rgb_mixed)), Image.fromarray(np.uint8(depth_mixed))

    def random_resized_crop(self, rgb_image, depth_image, size):
        i, j, h, w = transforms.RandomResizedCrop.get_params(rgb_image, scale=(0.7, 1.0), ratio=(1.0, 1.0))
        rgb_image = TF.resized_crop(rgb_image, i, j, h, w, size=size)
        depth_image = TF.resized_crop(depth_image, i, j, h, w, size=size)
        return rgb_image, depth_image	

    def random_horizontal_flip(self, rgb_image, depth_image):
        if random.random() > 0.5:
            rgb_image = TF.hflip(rgb_image)
            depth_image = TF.hflip(depth_image)
        return rgb_image, depth_image

    def to_tensor_and_normalize(self, rgb_image, depth_image):
        rgb_image = TF.to_tensor(rgb_image)
        depth_image = TF.to_tensor(depth_image)
        rgb_image = TF.normalize(rgb_image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        depth_image = TF.normalize(depth_image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return rgb_image, depth_image

    def random_greyscale(self, rgb_image, depth_image):
        if random.random() > 0.5:
            rgb_image = TF.to_grayscale(rgb_image, num_output_channels=3)
            depth_image = TF.to_grayscale(depth_image, num_output_channels=3)
        return rgb_image, depth_image

    def synchronized_transform_train_ch(self, rgb_image, depth_image):
        rgb_image, depth_image = self.random_resized_crop(rgb_image, depth_image, size=192)
        rgb_image, depth_image = self.random_horizontal_flip(rgb_image, depth_image)
        # ColorJitter (only for RGB)
        color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        rgb_image = color_jitter(rgb_image)
        rgb_image, depth_image = self.random_greyscale(rgb_image, depth_image)
        return self.to_tensor_and_normalize(rgb_image, depth_image)

    def synchronized_transform_train_ft(self, rgb_image, depth_image):
        rgb_image, depth_image = self.random_resized_crop(rgb_image, depth_image, size=192)
        rgb_image, depth_image = self.random_horizontal_flip(rgb_image, depth_image)
        rgb_image, depth_image = self.synchronized_augmix(rgb_image, depth_image)
        return self.to_tensor_and_normalize(rgb_image, depth_image)

    def synchronized_transform_val(self, rgb_image, depth_image):
        rgb_image = TF.resize(rgb_image, size=(192, 192))
        depth_image = TF.resize(depth_image, size=(192, 192))
        return self.to_tensor_and_normalize(rgb_image, depth_image)


class CustomImageDataset(Dataset):
    def __init__(self, rgb_folder_path, depth_folder_path, processor=None, transform=None):
        self.rgb_folder_path = rgb_folder_path
        self.depth_folder_path = depth_folder_path
        self.processor = processor
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.label_mapping = {}
        self.reverse_label_mapping = {}

        class_dirs = sorted([d for d in os.listdir(self.rgb_folder_path) if os.path.isdir(os.path.join(self.rgb_folder_path, d))])
        for label, class_dir in enumerate(class_dirs):
            self.label_mapping[class_dir] = label
            self.reverse_label_mapping[label] = class_dir

            rgb_class_path = os.path.join(self.rgb_folder_path, class_dir)
            depth_class_path = os.path.join(self.depth_folder_path, class_dir)

            valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
            for rgb_filename, depth_filename in zip(os.listdir(rgb_class_path), os.listdir(depth_class_path)):
                if rgb_filename.lower().endswith(valid_extensions) and depth_filename.lower().endswith(valid_extensions):
                    rgb_image_path = os.path.join(rgb_class_path, rgb_filename)
                    depth_image_path = os.path.join(depth_class_path, depth_filename)
                    self.image_paths.append((rgb_image_path, depth_image_path))
                    self.labels.append(label)

            #for rgb_filename in os.listdir(rgb_class_path):
            #    if rgb_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            #        rgb_image_path = os.path.join(rgb_class_path, rgb_filename)
            #        depth_image_path = os.path.join(depth_class_path, rgb_filename)
            #        self.image_paths.append((rgb_image_path, depth_image_path))
            #        self.labels.append(label)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rgb_image_path, depth_image_path = self.image_paths[idx]
        rgb_inputs = Image.open(rgb_image_path).convert('RGB')
        #depth_inputs = Image.open(depth_image_path).convert('RGB')
        depth_inputs_prelim = Image.open(depth_image_path).convert('L')
        depth_inputs = Image.new('RGB', depth_inputs_prelim.size)
        depth_inputs.paste(depth_inputs_prelim, (0, 0, depth_inputs_prelim.width, depth_inputs_prelim.height))
        label = self.labels[idx]
        if self.transform:
            rgb_inputs, depth_inputs = self.transform(rgb_inputs, depth_inputs)
        if self.processor:
            rgb_inputs = self.processor(images=rgb_inputs, return_tensors="pt", do_normalize=True, do_rescale=True, do_resize=True)
            depth_inputs = self.processor(images=depth_inputs, return_tensors="pt", do_normalize=True, do_rescale=True, do_resize=True)
            rgb_inputs = {'pixel_values': rgb_inputs['pixel_values'][0], 'labels': torch.tensor(label)}
            depth_inputs = {'pixel_values': depth_inputs['pixel_values'][0], 'labels': torch.tensor(label)}
            return rgb_inputs, depth_inputs
        rgb_inputs = {'pixel_values': rgb_inputs, 'labels': torch.tensor(label)}
        depth_inputs = {'pixel_values': depth_inputs, 'labels': torch.tensor(label)}
        return rgb_inputs, depth_inputs

    def get_class_name(self, label):
        return self.reverse_label_mapping[label]


class EnsembleModel(nn.Module):
    def __init__(self, model_rgb, model_depth, model_rgbd_classifier):
        super(EnsembleModel, self).__init__()

        self.model_rgb = model_rgb
        self.model_depth = model_depth
        self.model_rgbd_classifier = model_rgbd_classifier
        self.model_rgbd_classifier.classifier = nn.Linear(in_features=1024*2, out_features=102, bias=True)
        self.model_rgbd_classifier = list(self.model_rgbd_classifier.children())[-1]
        self.hook_output_rgb = None
        self.hook_output_depth = None
        self.model_rgb.swinv2.pooler.register_forward_hook(self.hook_fn_rgb)
        self.model_depth.swinv2.pooler.register_forward_hook(self.hook_fn_depth)

    def hook_fn_rgb(self, module, input, output):
        self.hook_output_rgb = output

    def hook_fn_depth(self, module, input, output):
        self.hook_output_depth = output

    def forward(self, pixel_values_rgb, pixel_values_depth):
        _ = self.model_rgb(pixel_values=pixel_values_rgb)
        _ = self.model_depth(pixel_values=pixel_values_depth)
        x1 = self.hook_output_rgb.squeeze(-1)
        x2 = self.hook_output_depth.squeeze(-1)
        x = torch.cat((x1, x2), dim=1)
        out = self.model_rgbd_classifier(x)
        return out


################################################################################################
####################################### HELPER FUNCTIONS #######################################
################################################################################################


def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def unfreeze_classifier(model):
    for param in model.model_rgbd_classifier.parameters():
        param.requires_grad = True
    return model

def unfreeze_global_average_pooling_layer(model):
    for param in model.model_rgb.swinv2.layernorm.parameters():
        param.requires_grad = True
    for param in model.model_depth.swinv2.layernorm.parameters():
        param.requires_grad = True
    return model

def unfreeze_stage_3(model):
    stage_rgb = model.model_rgb.swinv2.encoder.layers[3]
    for param in stage_rgb.parameters():
        param.requires_grad = True
    stage_depth = model.model_depth.swinv2.encoder.layers[3]
    for param in stage_depth.parameters():
        param.requires_grad = True    
    return model

def unfreeze_stage_2(model):
    stage_rgb = model.model_rgb.swinv2.encoder.layers[2]
    for param in stage_rgb.parameters():
        param.requires_grad = True
    stage_depth = model.model_depth.swinv2.encoder.layers[2]
    for param in stage_depth.parameters():
        param.requires_grad = True    
    return model

def unfreeze_stage_1(model):
    stage_rgb = model.model_rgb.swinv2.encoder.layers[1]
    for param in stage_rgb.parameters():
        param.requires_grad = True
    stage_depth = model.model_depth.swinv2.encoder.layers[1]
    for param in stage_depth.parameters():
        param.requires_grad = True    
    return model

def unfreeze_stage_0(model):
    stage_rgb = model.model_rgb.swinv2.encoder.layers[0]
    for param in stage_rgb.parameters():
        param.requires_grad = True
    stage_depth = model.model_depth.swinv2.encoder.layers[0]
    for param in stage_depth.parameters():
        param.requires_grad = True    
    return model

def unfreeze_patch_partition_and_linear_embedding_layers(model):
    for param in model.model_rgb.swinv2.embeddings.parameters():
        param.requires_grad = True
    for param in model.model_depth.swinv2.embeddings.parameters():
        param.requires_grad = True
    return model

unfreeze_layer_config_functions = {
	'classifier': unfreeze_classifier,
    'global average pooling': unfreeze_global_average_pooling_layer,
    'unfreeze stage 3': unfreeze_stage_3,
    'unfreeze stage 2': unfreeze_stage_2,
    'unfreeze stage 1': unfreeze_stage_1,
    'unfreeze stage 0': unfreeze_stage_0,
    'patch partition and linear embedding': unfreeze_patch_partition_and_linear_embedding_layers, 
    }

def freeze_specified_layers(model, unfreeze_layers_options):
    for config_name in unfreeze_layers_options:
        if config_name in unfreeze_layer_config_functions:
            unfreeze_function = unfreeze_layer_config_functions[config_name]
            model = unfreeze_function(model)
        else:
            raise ValueError(f"Configuration '{config_name}' is not defined.")
    return model


def print_train_test_results(val_loss_array, val_acc_array, test_accuracy, run):
    min_loss = min(val_loss_array)
    min_loss_epoch = val_loss_array.index(min_loss)
    min_loss_accuracy = val_acc_array[min_loss_epoch]
    print(f"\nResults ({run}):", flush=True)
    print("\tMin val loss {:.4f} was achieved during epoch #{}".format(min_loss, min_loss_epoch + 1), flush=True)
    print("\tVal Accuracy during min val loss is {:.4f}".format(min_loss_accuracy), flush=True)
    print(f"Test Accuracy: {test_accuracy:.2f}%", flush=True)


################################################################################################
################################## TRAINING/TESTING FUNCTIONS ##################################
################################################################################################

def training(model, model_name, num_epochs, loss_function, optimizer, scheduler, train_dataloader, val_dataloader, initial_val_loss=np.inf):

    train_loss_array = []
    train_acc_array = []
    val_loss_array = []
    val_acc_array = []
    lowest_val_loss = initial_val_loss
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
                    for (batch_rgb, batch_depth) in train_dataloader:
                        inputs_rgb = batch_rgb['pixel_values'].to(DEVICE)
                        inputs_depth = batch_depth['pixel_values'].to(DEVICE)
                        targets = batch_rgb['labels'].to(DEVICE)

                        optimizer.zero_grad()
                        outputs = model(pixel_values_rgb=inputs_rgb, pixel_values_depth=inputs_depth)
                        loss = loss_function(outputs, targets)

                        loss.backward()
                        optimizer.step()

                        if scheduler:
                            scheduler.step()

                        preds = outputs.argmax(dim=1)
                        correct_items = (preds == targets).float().sum()

                        epoch_loss += loss.item()
                        epoch_correct_items += correct_items.item()
                        epoch_items += len(targets)

                train_loss_array.append(epoch_loss / epoch_items)
                train_acc_array.append(epoch_correct_items / epoch_items)

            elif phase == 'val':
                model.eval()
                with torch.no_grad():
                    for (batch_rgb, batch_depth) in val_dataloader:
                        inputs_rgb = batch_rgb['pixel_values'].to(DEVICE)
                        inputs_depth = batch_depth['pixel_values'].to(DEVICE)
                        targets = batch_rgb['labels'].to(DEVICE)

                        outputs = model(pixel_values_rgb=inputs_rgb, pixel_values_depth=inputs_depth)
                        loss = loss_function(outputs, targets)

                        preds = outputs.argmax(dim=1)
                        correct_items = (preds == targets).float().sum()

                        epoch_loss += loss.item()
                        epoch_correct_items += correct_items.item()
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
        for (batch_rgb, batch_depth) in test_dataloader:
            inputs_rgb = batch_rgb['pixel_values'].to(DEVICE)
            inputs_depth = batch_depth['pixel_values'].to(DEVICE)
            targets = batch_rgb['labels'].to(DEVICE)

            outputs = model(pixel_values_rgb=inputs_rgb, pixel_values_depth=inputs_depth)

            all_outputs.append(outputs)
            all_labels.append(targets)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    _, predictions = all_outputs.max(1)
    correct = (predictions == all_labels).sum().item()
    total = all_labels.size(0)
    accuracy = 100 * correct / total

    return all_outputs, all_labels, accuracy


#def compute_accuracy(outputs, labels):
#    _, predictions = outputs.max(1)
#    correct = (predictions == labels).sum().item()
#    total = labels.size(0)
#    accuracy = 100 * correct / total
#    return accuracy


################################################################################################
###################################### PIPELINE FUNCTIONS ######################################
################################################################################################


def run_ch_training(processor=None, transforms_train=None, transform_val=None, batch_size=32):
    
    print("*"*30, flush=True)
    print("Classification Head Training Configurations", flush=True)
    print("Transforms: Hugging Face Processor", flush=True) if processor else print("Transforms: Synthnet Transforms", flush=True)
    print(f"Batch Size: {batch_size}", flush=True)
    print("*"*30, flush=True)
	
    train_dataset = CustomImageDataset(rgb_folder_path='/mnt/data/data_rgb/train', depth_folder_path='/mnt/data/data_depth/train', processor=processor, transform=transforms_train)
    val_dataset = CustomImageDataset(rgb_folder_path='/mnt/data/data_rgb/val', depth_folder_path='/mnt/data/data_depth/val', processor=processor, transform=transform_val)
    test_dataset = CustomImageDataset(rgb_folder_path='/mnt/data/data_rgb/test', depth_folder_path='/mnt/data/data_depth/test', processor=processor, transform=transform_val)
	
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
		
    base_model_swin_rgb = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k", num_labels=102, ignore_mismatched_sizes=True)
    base_model_swin_depth = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k", num_labels=102, ignore_mismatched_sizes=True)
    classifier_model_swin_rgbd = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k", num_labels=102, ignore_mismatched_sizes=True)
	
    swinv2_late_fusion_model_ch_training = EnsembleModel(base_model_swin_rgb, base_model_swin_depth, classifier_model_swin_rgbd)
		
    swinv2_late_fusion_model_ch_training = freeze_all_layers(swinv2_late_fusion_model_ch_training)

    swinv2_late_fusion_model_ch_training = unfreeze_classifier(swinv2_late_fusion_model_ch_training)
	
    swinv2_late_fusion_model_ch_training = swinv2_late_fusion_model_ch_training.to(DEVICE)
		
    num_epochs_ch, loss_function_ch, optimizer_ch, scheduler_ch = get_hyperparameters_ch(swinv2_late_fusion_model_ch_training)
	    	
    swinv2_late_fusion_model_ch_training_results = training(model=swinv2_late_fusion_model_ch_training,
	                                     model_name='swinv2_late_fusion_model_ch',
	                                     num_epochs=num_epochs_ch,
	                                     loss_function = loss_function_ch,
	                                     optimizer = optimizer_ch,
	                                     scheduler = scheduler_ch,
	                                     train_dataloader=train_loader,
	                                     val_dataloader=val_loader)

    swinv2_late_fusion_model_ch, train_loss_array, train_acc_array, val_loss_array, val_acc_array = swinv2_late_fusion_model_ch_training_results
	
    #all_outputs, all_labels = test_model(swinv2_late_fusion_model_ch, test_loader)
	
    #test_accuracy = compute_accuracy(all_outputs, all_labels)

    all_outputs, all_labels, test_accuracy = test_model(swinv2_late_fusion_model_ch, test_loader)
	
    print_train_test_results(val_loss_array, val_acc_array, test_accuracy, "CH")

    return swinv2_late_fusion_model_ch


def run_ch_ft_training_with_config(config):

    ch_model, transform_train, unfreeze_layers, batch_size = config

    print("*"*30, flush=True)
    print("Full Training Configuration:", flush=True)
    model_name = getattr(ch_model, 'model_name', 'Unknown Model')
    print(f"Classification Head Training Transforms: {model_name}", flush=True)
    transform_name = "Synthnet Transforms" if transform_train else "Hugging Face Processor"
    print(f"Full Training Transforms: {transform_name}", flush=True)
    print("Freezing Schedule:", flush=True)
    print(unfreeze_layers, flush=True)
    print(f"Batch Size {batch_size}", flush=True)
    print("*"*30, flush=True)

    processor = None if transform_train else AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
    transform_val = CustomTransforms().synchronized_transform_val if transform_train else None

    train_dataset = CustomImageDataset(rgb_folder_path='/mnt/data/data_rgb/train', depth_folder_path='/mnt/data/data_depth/train', processor=processor, transform=transform_train)
    val_dataset = CustomImageDataset(rgb_folder_path='/mnt/data/data_rgb/val', depth_folder_path='/mnt/data/data_depth/val', processor=processor, transform=transform_val)
    test_dataset = CustomImageDataset(rgb_folder_path='/mnt/data/data_rgb/test', depth_folder_path='/mnt/data/data_depth/test', processor=processor, transform=transform_val)
	
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	
    ft_model_training = copy.deepcopy(ch_model)
    for phase, layers_unfrozen in enumerate(unfreeze_layers):
        print(f"Phase: {phase+1}\nUnfrozen Layers: {layers_unfrozen}", flush=True)
        ft_model_training = freeze_all_layers(ft_model_training)
        ft_model_training = freeze_specified_layers(ft_model_training, layers_unfrozen)
        ft_model_training = ft_model_training.to(DEVICE)
		
        num_epochs_ft, loss_function_ft, optimizer_ft, scheduler_ft = get_hyperparameters_ft(ft_model_training)
		
        swinv2_late_fusion_model_ch_ft_training_results = training(model=ft_model_training,
	                                     model_name='swinv2_late_fusion_model_ch_ft',
	                                     num_epochs=num_epochs_ft,
	                                     loss_function = loss_function_ft,
	                                     optimizer = optimizer_ft,
	                                     scheduler = scheduler_ft,
	                                     train_dataloader=train_loader,
	                                     val_dataloader=val_loader,)
	                                     #initial_val_loss=min(val_loss_array))
		
        ft_model_training, train_loss_array, train_acc_array, val_loss_array, val_acc_array = swinv2_late_fusion_model_ch_ft_training_results
	
        #all_outputs, all_labels = test_model(ft_model_training, test_loader)
	
        #test_accuracy = compute_accuracy(all_outputs, all_labels)

        all_outputs, all_labels, test_accuracy = test_model(ft_model_training, test_loader)
	
        print_train_test_results(val_loss_array, val_acc_array, test_accuracy, f"CH-FT ({phase+1})")

    return None


###################################################################################################
####################################### ORCHESTRATION CODE ########################################
###################################################################################################


############################ Classification Head Training ###########################

swinv2_late_fusion_model_ch_huggingface_processor = run_ch_training(processor=AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k"),
																	transforms_train=None,
																	transform_val=None,
																	batch_size=32
																	)
swinv2_late_fusion_model_ch_huggingface_processor.model_name = "Hugging Face Processor"

swinv2_late_fusion_model_ch_synchronized_tranforms = run_ch_training(processor=None,
																	transforms_train=CustomTransforms().synchronized_transform_train_ch,
																	transform_val=CustomTransforms().synchronized_transform_val,
																	batch_size=32
																	)
swinv2_late_fusion_model_ch_synchronized_tranforms.model_name = "Synthnet Transforms"


#################### GridSearch Classification Head / Full Training ####################

ch_model_options = [swinv2_late_fusion_model_ch_huggingface_processor, swinv2_late_fusion_model_ch_synchronized_tranforms]

transform_options = [None, CustomTransforms().synchronized_transform_train_ft]

unfreeze_layers_options = [ 
    [['classifier', 'unfreeze stage 3', 'unfreeze stage 2', 'unfreeze stage 1', 'unfreeze stage 0']],

    [['classifier', 'global average pooling', 'unfreeze stage 3', 'unfreeze stage 2', 'unfreeze stage 1', 'unfreeze stage 0', 'patch partition and linear embedding']],

    [['classifier', 'unfreeze stage 3','unfreeze stage 2'], 
     ['classifier', 'unfreeze stage 1', 'unfreeze stage 0',]],

    [['classifier', 'global average pooling', 'unfreeze stage 3','unfreeze stage 2'], 
     ['classifier', 'unfreeze stage 1', 'unfreeze stage 0', 'patch partition and linear embedding']],
    
    [['classifier', 'unfreeze stage 3'],
     ['classifier', 'unfreeze stage 2'],
     ['classifier', 'unfreeze stage 1'],
     ['classifier', 'unfreeze stage 0']],

    [['classifier', 'global average pooling', 'unfreeze stage 3', 'patch partition and linear embedding'],
     ['classifier', 'global average pooling', 'unfreeze stage 2', 'patch partition and linear embedding'],
     ['classifier', 'global average pooling', 'unfreeze stage 1', 'patch partition and linear embedding'],
     ['classifier', 'global average pooling', 'unfreeze stage 0', 'patch partition and linear embedding']],

    [['classifier', 'unfreeze stage 3',],
     ['classifier', 'unfreeze stage 3', 'unfreeze stage 2',],
     ['classifier', 'unfreeze stage 3', 'unfreeze stage 2', 'unfreeze stage 1',],
     ['classifier', 'unfreeze stage 3', 'unfreeze stage 2', 'unfreeze stage 1', 'unfreeze stage 0',]],

    [['classifier', 'unfreeze stage 0',],
     ['classifier', 'unfreeze stage 0', 'unfreeze stage 1',],
     ['classifier', 'unfreeze stage 0', 'unfreeze stage 1', 'unfreeze stage 2',],
     ['classifier', 'unfreeze stage 0', 'unfreeze stage 1', 'unfreeze stage 2', 'unfreeze stage 3',]],

    [['classifier', 'global average pooling', 'unfreeze stage 3', 'patch partition and linear embedding'],
    ['classifier', 'global average pooling', 'unfreeze stage 3', 'unfreeze stage 2', 'patch partition and linear embedding'],
    ['classifier', 'global average pooling', 'unfreeze stage 3', 'unfreeze stage 2', 'unfreeze stage 1', 'patch partition and linear embedding'],
    ['classifier', 'global average pooling', 'unfreeze stage 3', 'unfreeze stage 2', 'unfreeze stage 1', 'unfreeze stage 0', 'patch partition and linear embedding'],],

    [['classifier', 'global average pooling', 'unfreeze stage 0', 'patch partition and linear embedding'],
    ['classifier', 'global average pooling', 'unfreeze stage 0', 'unfreeze stage 1', 'patch partition and linear embedding'],
    ['classifier', 'global average pooling', 'unfreeze stage 0', 'unfreeze stage 1', 'unfreeze stage 2', 'patch partition and linear embedding'],
    ['classifier', 'global average pooling', 'unfreeze stage 0', 'unfreeze stage 1', 'unfreeze stage 2', 'unfreeze stage 3', 'patch partition and linear embedding'],],
]

batch_size_options = [32]

all_configs = list(itertools.product(ch_model_options, transform_options, unfreeze_layers_options, batch_size_options))

for config in all_configs:
    run_ch_ft_training_with_config(config)


print("###################################################################################################")
print("############################### GRIDSEARCH CH-FT TRAINING COMPLETED ###############################")
print("###################################################################################################")


# Delete Transformers Cache

shutil.rmtree("/mnt/data/transformers_cache", ignore_errors=True)