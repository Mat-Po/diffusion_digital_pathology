import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader,Subset,ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import numpy as np
import random
from PIL import ImageEnhance


class DPpublicdata(LightningDataModule):
    def __init__(self, data_dir,test_data_dir,batch_size = 32,num_workers = 4,per = 0.7,color = True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.per = per
        self.test_data_dir = test_data_dir
        if color:
            self.train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.3),
            transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 5)),
            transforms.Normalize(mean=[0.7358, 0.4814, 0.6079], 
                            std= [0.1314, 0.1863, 0.1465]
                            ),
            transforms.RandomInvert(p=0.3),
            transforms.RandomErasing()
            ])
            self.val_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.7358, 0.4814, 0.6079], 
                            std= [0.1314, 0.1863, 0.1465]
                            )
                ])
        else:
            print('USING GRAYSCALE')
            self.train_transforms = transforms.Compose([
            transforms.RandomCrop(336),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels = 3),
            transforms.Normalize(mean=[0.5896, 0.5896, 0.5896], 
                            std= [0.1549, 0.1549, 0.1549] 
                            )
            ])
            self.val_transforms = transforms.Compose([
                transforms.CenterCrop(336),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels = 3),
                transforms.Normalize(mean=[0.5896, 0.5896, 0.5896], 
                            std= [0.1549, 0.1549, 0.1549]  
                            )
                ])


    def prepare_data(self):
        # download
      pass  

    def setup(self, stage=None):
        self.data_test = ImageFolder(root=self.test_data_dir,transform=self.val_transforms)
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset = ImageFolder(root= self.data_dir,transform= self.train_transforms)
            dataset_2 = ImageFolder(root= self.data_dir,transform= self.val_transforms)
            labels = dataset.targets
            num_classes = len(set(labels))
            train_indices = []
            val_indices = []
            for i in range(num_classes):
                # Get indices of samples corresponding to current class
                class_indices = np.where(np.array(labels) == i)[0]
                class_samples = [dataset.samples[idx][0] for idx in class_indices]
                # Use sklearn's train_test_split to split class_indices into train and validation sets
                #train_size = int(self.per*len(class_indices))
                #train_class_indices, val_class_indices = train_test_split(class_indices, train_size=train_size)

                ## if u want to leave slides out and not tiles
                sample_names = [sample.split('slidename-')[-1] for sample in class_samples]
                slide_names = ['-'.join(slide.split('-')[:3]) for slide in sample_names]
                unique_slides = list(set(slide_names))
                train_size = int(self.per*len(unique_slides))
                train_slides = random.sample(unique_slides,train_size)
                train_class_indices = [index for idx,index in enumerate(class_indices) if slide_names[idx] in train_slides]
                train_class_indices = random.sample(train_class_indices,3001)
                val_class_indices = [index for idx,index in enumerate(class_indices) if not(slide_names[idx] in train_slides)]
                # Add train and validation indices to train_indices and val_indices lists
                train_indices.extend(train_class_indices)
                val_indices.extend(val_class_indices)
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset_2, val_indices)
            self.data_train = train_dataset
            self.data_val = val_dataset


    def train_dataloader(self):
        return DataLoader(self.data_train,shuffle=True, batch_size=self.batch_size,num_workers = self.num_workers,persistent_workers=True,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size,num_workers = self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size,num_workers = self.num_workers)
    


class DPmix(LightningDataModule):
    def __init__(self, data_dir_real,data_dir_fake,batch_size = 32,num_workers = 4,per = 0.7,color = True):
        super().__init__()
        self.data_dir_real = data_dir_real
        self.data_dir_fake = data_dir_fake
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.per = per

        
        self.train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.3),
            transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 5)),
            transforms.Normalize(mean=[0.7358, 0.4814, 0.6079], 
                            std= [0.1314, 0.1863, 0.1465]
                            ),
            transforms.RandomInvert(p=0.3),
            transforms.RandomErasing()
            ])
        self.val_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.7358, 0.4814, 0.6079], 
                            std= [0.1314, 0.1863, 0.1465]
                            )
                ])
        
    def prepare_data(self):
        # download
      pass  

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset = ImageFolder(root= self.data_dir_real,transform= self.train_transforms)
            dataset_2 = ImageFolder(root= self.data_dir_real,transform= self.val_transforms)
            labels = dataset.targets
            num_classes = len(set(labels))
            train_indices = []
            val_indices = []
            for i in range(num_classes):
                # Get indices of samples corresponding to current class
                class_indices = np.where(np.array(labels) == i)[0]
                class_samples = [dataset.samples[idx][0] for idx in class_indices]
                # Use sklearn's train_test_split to split class_indices into train and validation sets
                #train_size = int(self.per*len(class_indices))
                #train_class_indices, val_class_indices = train_test_split(class_indices, train_size=train_size)

                ## if u want to leave slides out and not tiles
                sample_names = [sample.split('slidename-')[-1] for sample in class_samples]
                slide_names = ['-'.join(slide.split('-')[:3]) for slide in sample_names]
                unique_slides = list(set(slide_names))
                train_size = int(self.per*len(unique_slides))
                train_slides = random.sample(unique_slides,train_size)
                train_class_indices = [index for idx,index in enumerate(class_indices) if slide_names[idx] in train_slides]
                train_class_indices = random.sample(train_class_indices,3001)
                val_class_indices = [index for idx,index in enumerate(class_indices) if not(slide_names[idx] in train_slides)]
                # Add train and validation indices to train_indices and val_indices lists
                train_indices.extend(train_class_indices)
                val_indices.extend(val_class_indices)
            train_dataset_real = Subset(dataset, train_indices)
            val_dataset_real = Subset(dataset_2, val_indices)

            dataset = ImageFolder(root= self.data_dir_fake,transform= self.train_transforms)
            dataset_2 = ImageFolder(root= self.data_dir_fake,transform= self.val_transforms)
            labels = dataset.targets
            num_classes = len(set(labels))
            train_indices = []
            val_indices = []
            for i in range(num_classes):
                # Get indices of samples corresponding to current class
                class_indices = np.where(np.array(labels) == i)[0]
                # Use sklearn's train_test_split to split class_indices into train and validation sets
                train_size = int(self.per*len(class_indices))
                train_class_indices, val_class_indices = train_test_split(class_indices, train_size=train_size)

                # Add train and validation indices to train_indices and val_indices lists
                train_indices.extend(train_class_indices)
                val_indices.extend(val_class_indices)
            train_dataset_fake = Subset(dataset, train_indices)
            val_dataset_fake= Subset(dataset_2, val_indices)


            train_dataset = ConcatDataset([train_dataset_fake, train_dataset_real])
            val_dataset = ConcatDataset([val_dataset_fake,val_dataset_real])
            self.data_train = train_dataset
            self.data_val = val_dataset


    def train_dataloader(self):
        return DataLoader(self.data_train,shuffle=True, batch_size=self.batch_size,num_workers = self.num_workers,persistent_workers=True,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size,num_workers = self.num_workers)


class DPsyntheticdata(LightningDataModule):
    def __init__(self, data_dir,batch_size = 32,num_workers = 4,per = 0.7):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.per = per
        self.train_transforms = transforms.Compose([
            transforms.Lambda(lambda img: ImageEnhance.Sharpness(img).enhance(3)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.3),
            transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 5)),
            transforms.Normalize(mean=[0.6895, 0.4152, 0.5501], 
                            std= [0.1044, 0.1355, 0.1108]
                            ),
            transforms.RandomInvert(p=0.3),
            transforms.RandomErasing()
            ])
        self.val_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.6895, 0.4152, 0.5501], 
                            std= [0.1044, 0.1355, 0.1108]
                            )
                ])
    
    def prepare_data(self):
        # download
      pass  

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset = ImageFolder(root= self.data_dir,transform= self.train_transforms)
            dataset_2 = ImageFolder(root= self.data_dir,transform= self.val_transforms)
            labels = dataset.targets
            num_classes = len(set(labels))
            train_indices = []
            val_indices = []
            for i in range(num_classes):
                # Get indices of samples corresponding to current class
                class_indices = np.where(np.array(labels) == i)[0]
                # Use sklearn's train_test_split to split class_indices into train and validation sets
                train_size = int(self.per*len(class_indices))
                train_class_indices, val_class_indices = train_test_split(class_indices, train_size=train_size)

                # Add train and validation indices to train_indices and val_indices lists
                train_indices.extend(train_class_indices)
                val_indices.extend(val_class_indices)
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset_2, val_indices)
            self.data_train = train_dataset
            self.data_val = val_dataset


    def train_dataloader(self):
        return DataLoader(self.data_train,shuffle=True, batch_size=self.batch_size,num_workers = self.num_workers,persistent_workers=True,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size,num_workers = self.num_workers)
    





class DPreal_vs_fake(LightningDataModule):
    def __init__(self, data_dir,batch_size = 32,num_workers = 4,per = 0.7):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.per = per
        self.train_transforms = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.ToTensor(),
            #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.3),
            #transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 5)),
            transforms.Normalize(mean=[0.7158, 0.4506, 0.5808], 
                            std= [0.1166, 0.1610, 0.1286]
                            ),
            transforms.RandomInvert(p=0.3),
            transforms.RandomErasing()
            ])
        self.val_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.7158, 0.4506, 0.5808], 
                            std= [0.1167, 0.1610, 0.1286]
                            )
                ])
    
    def prepare_data(self):
        # download
      pass  

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset = ImageFolder(root= self.data_dir,transform= self.train_transforms)
            dataset_2 = ImageFolder(root= self.data_dir,transform= self.val_transforms)
            labels = dataset.targets
            num_classes = len(set(labels))
            train_indices = []
            val_indices = []
            for i in range(num_classes):
                # Get indices of samples corresponding to current class
                class_indices = np.where(np.array(labels) == i)[0]
                # Use sklearn's train_test_split to split class_indices into train and validation sets
                train_size = int(self.per*len(class_indices))
                train_class_indices, val_class_indices = train_test_split(class_indices, train_size=train_size)
                # Add train and validation indices to train_indices and val_indices lists
                train_indices.extend(train_class_indices)
                val_indices.extend(val_class_indices)
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset_2, val_indices)
            self.data_train = train_dataset
            self.data_val = val_dataset


    def train_dataloader(self):
        return DataLoader(self.data_train,shuffle=True, batch_size=self.batch_size,num_workers = self.num_workers,persistent_workers=True,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size,num_workers = self.num_workers)
    







