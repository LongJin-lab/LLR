import torch
import torchvision.datasets 
import torchvision.transforms
import numpy as np

from utils.configurable import configurable
from data.build import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class CIFAR10_base:
    @configurable
    def __init__(self, datadir,val_type,val_rat) -> None:
        self.datadir = datadir
        self.val_type = val_type
        self.val_rat = val_rat
        self.n_classes = 10
        self.mean = np.array([125.3, 123.0, 113.9]) / 255.0
        self.std = np.array([63.0, 62.1, 66.7]) / 255.0
    
    @classmethod
    def from_config(cls, args):
        return {
            "datadir": args.datadir+'/cifar10',
            "val_type": args.val_type,
            "val_rat": args.val_rat,
        }

    # def get_data(self):
    #     train_data = torchvision.datasets.CIFAR10(root=self.datadir, train=True, transform=self._train_transform(), download=True)
    #     test_data = torchvision.datasets.CIFAR10(root=self.datadir, train=False, transform=self._test_transform(), download=True)
    #     return train_data, test_data

    def get_data(self, args):
        train_data = torchvision.datasets.CIFAR10(root=self.datadir, train=True, transform=self._train_transform(), download=True)
        
        valset_size = int(len(train_data) * args.val_rat)
        trainset_size = len(train_data) - valset_size
        # if not valset_size==0:
        train_data, val_data = torch.utils.data.random_split(train_data, [trainset_size, valset_size])
        # TODO: it seems that val_data can only use the '_train_transform'
        if val_data:
            val_data.dataset.transform = self._val_transform()
        # else:
        #     val_data = None
        test_data = torchvision.datasets.CIFAR10(root=self.datadir, train=False, transform=self._test_transform(), download=True)
        return train_data, val_data, test_data
    
    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            # Cutout()
        ])
        return train_transform

    def _val_transform(self):
        if self.val_type == 'train':
            val_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),            
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std)
            ])
        elif self.val_type == 'test':
            val_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
            ])
        return val_transform
    
    def _test_transform(self):
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        return test_transform

@DATASET_REGISTRY.register()
class CIFAR10_cutout(CIFAR10_base):
    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            Cutout(size=16, p=0.5),
        ])
        return train_transform
    def _val_transform(self):
        if self.val_type == 'train':
            val_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
                Cutout(size=16, p=0.5),
            ])
        elif self.val_type == 'test':
            val_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
            ])
        return val_transform    

@DATASET_REGISTRY.register()
class CIFAR10_auto(CIFAR10_base):
    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            # torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10),            
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return train_transform
    def _val_transform(self):
        if self.val_type == 'train':
            val_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10),            
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
            ])
        elif self.val_type == 'test':
            val_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
            ])
        
        return val_transform      

@DATASET_REGISTRY.register()
class CIFAR100_base:
    @configurable
    def __init__(self, datadir,val_type,val_rat) -> None:
        self.datadir = datadir
        self.val_type = val_type
        self.val_rat = val_rat

        self.n_classes = 100
        self.mean = np.array([125.3, 123.0, 113.9]) / 255.0
        self.std = np.array([63.0, 62.1, 66.7]) / 255.0
    
    @classmethod
    def from_config(cls, args):
        return {
            "datadir": args.datadir+'/cifar100',
            "val_type": args.val_type,
            "val_rat": args.val_rat,

        }
    
    def get_data(self, args):
        train_data = torchvision.datasets.CIFAR100(root=self.datadir, train=True, transform=self._train_transform(), download=True)
        # print('args',args)
        valset_size = int(len(train_data) * args.val_rat)
        trainset_size = len(train_data) - valset_size
        train_data, val_data = torch.utils.data.random_split(train_data, [trainset_size, valset_size])
        
        test_data = torchvision.datasets.CIFAR100(root=self.datadir, train=False, transform=self._test_transform(), download=True)
        # return train_data, test_data
        return train_data, val_data, test_data

    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            # Cutout()
        ])
        return train_transform

    def _val_transform(self):
        if self.val_type == 'train':
            val_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
            ])
        elif self.val_type == 'test':
            val_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
            ])        

        return val_transform
    
    def _test_transform(self):
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        return test_transform

@DATASET_REGISTRY.register()
class CIFAR100_cutout(CIFAR100_base):
    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            Cutout(size=16, p=0.5),
        ])
        return train_transform
    def _val_transform(self):
        if self.val_type == 'train':
            val_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
                Cutout(size=16, p=0.5),
            ])
        elif self.val_type == 'test':
            val_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
            ])        

        return val_transform    

@DATASET_REGISTRY.register()
class CIFAR100_auto(CIFAR100_base):
    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10),            
            # torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return train_transform
    def _val_transform(self):
        if self.val_type == 'train':
            val_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10),            
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
            ])
        elif self.val_type == 'test':
            val_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
            ])        

        return val_transform      
    
@DATASET_REGISTRY.register()
class ImageNet_base:
    @configurable
    def __init__(self, datadir,val_type,val_rat) -> None:
        self.datadir = datadir
        self.val_type = val_type
        self.val_rat = val_rat

        self.n_classes = 1000
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    @classmethod
    def from_config(cls, args):
        return {
            "datadir": args.datadir,
            "val_type": args.val_type,
            "val_rat": args.val_rat,

        }
    
    def get_data(self, args):
        train_dataset = torchvision.datasets.ImageFolder(root=self.datadir + '/train', transform=self._train_transform())

        valset_size = int(len(train_dataset) * args.val_rat)
        trainset_size = len(train_dataset) - valset_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [trainset_size, valset_size])
        
        test_dataset = torchvision.datasets.ImageFolder(root=self.datadir + '/val', transform=self._test_transform())
        return train_dataset, val_dataset, test_dataset

    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            # Cutout()
        ])
        return train_transform

    def _test_transform(self):
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return test_transform

    def _val_transform(self):
        if self.val_type == 'train':
            val_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
            ])
        elif self.val_type == 'test':
            val_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
            ])        

        return val_transform


@DATASET_REGISTRY.register()
class SVHN_base:
    @configurable
    def __init__(self, datadir,val_type,val_rat) -> None:
        self.datadir = datadir
        self.val_type = val_type
        self.val_rat = val_rat

        self.n_classes = 100
        self.mean = np.array([125.3, 123.0, 113.9]) / 255.0
        self.std = np.array([63.0, 62.1, 66.7]) / 255.0
    
    @classmethod
    def from_config(cls, args):
        return {
            "datadir": args.datadir+'/svhn',
            "val_type": args.val_type,
            "val_rat": args.val_rat,

        }
    
    def get_data(self, args):
        train_data = torchvision.datasets.SVHN(root=self.datadir, split='train', transform=self._train_transform(), download=True)
        # print('args',args)
        valset_size = int(len(train_data) * args.val_rat)
        trainset_size = len(train_data) - valset_size
        train_data, val_data = torch.utils.data.random_split(train_data, [trainset_size, valset_size])
        
        test_data = torchvision.datasets.SVHN(root=self.datadir, split='test', transform=self._test_transform(), download=True)
        # return train_data, test_data
        return train_data, val_data, test_data

    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            # Cutout()
        ])
        return train_transform

    def _val_transform(self):
        if self.val_type == 'train':
            val_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
            ])
        elif self.val_type == 'test':
            val_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
            ])        

        return val_transform
    
    def _test_transform(self):
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        return test_transform

@DATASET_REGISTRY.register()
class SVHN_cutout(SVHN_base):
    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            Cutout(size=16, p=0.5),
        ])
        return train_transform
    def _val_transform(self):
        if self.val_type == 'train':
            val_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
                Cutout(size=16, p=0.5),
            ])
        elif self.val_type == 'test':
            val_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
            ])        

        return val_transform

@DATASET_REGISTRY.register()
class SVHN_auto(SVHN_base):
    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10),            
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            Cutout(size=16, p=0.5),
        ])
        return train_transform
    def _val_transform(self):
        if self.val_type == 'train':
            val_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10),            
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
                Cutout(size=16, p=0.5),
            ])
        elif self.val_type == 'test':
            val_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
            ])        

        return val_transform
    
class Cutout(object):
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image
        
        h, w = image.size(1), image.size(2)
        mask = np.ones((h,w), np.float32)

        x = np.random.randint(w)
        y = np.random.randint(h)

        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image = image * mask
        return image