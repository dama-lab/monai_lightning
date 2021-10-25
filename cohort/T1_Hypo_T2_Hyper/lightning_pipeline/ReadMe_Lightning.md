# Pytorch Lightning Tutorial
[TOC]
- Lightning training module:
> Ref: https://www.geeksforgeeks.org/training-neural-networks-using-pytorch-lightning/
- Lightning Data module:
> Ref: https://www.geeksforgeeks.org/understanding-pytorch-lightning-datamodules/
# Introduction:

PyTorch Lightning is a library that provides a high-level interface for PyTorch. Problem with PyTorch is that every time you start a project you have to rewrite those training and testing loop. PyTorch Lightning fixes the problem by not only reducing boilerplate code but also providing added functionality that might come handy while training your neural networks. One of the things I love about Lightning is that the code is very organized and reusable, and not only that but it reduces the training and testing loop while retain the flexibility that PyTorch is known for. And once you learn how to use it you’ll see how similar the code is to that of PyTorch.
# PyTorch Lightning Model Format:
```python
import pytorch-lightning as pl

class model(pl.LightningModule):
    def __init__(self):
        # Define Model Here
        
    def forward(self, x):
        # Define Forward Pass Here
    
    def configure_optimizers(self):
       # Define Optimizer Here
       
    def training_step(self, train_batch, batch_idx):
        # Define Training loop steps here
        
    def validation_step(self, valid_batch, batch_idx):
        # Define Validation loop steps here
```
> Note: The names of the above functions should be exactly the same.
> 
## Training Neural Network:
### Loading by creating DataLoader:
```python
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor()
])

train = datasets.MNIST('',train = True, download = True, transform=transform)
test = datasets.MNIST('',train = False, download = True, transform=transform)

trainloader = DataLoader(train, batch_size= 32, shuffle=True)
testloader = DataLoader(test, batch_size= 32, shuffle=True)
```
### Loading Data by Creating LightningDataModule
```python
import pytorch-lightning as pl
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

class Data(pl.LightningDataModule):
    def prepare_data(self):
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
      
        self.train_data = datasets.MNIST('', train=True, download=True, transform=transform)
        self.test_data = datasets.MNIST('', train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size= 32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size= 32, shuffle=True)
```
> Note: The names of the above functions should be exactly the same.

### Defining Neural Network
Defining the model in PyTorch lighting is pretty much the same as that in PyTorch except now we are clubbing everything inside our model class.
```python
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import SGD

class model(pl.LightningModule):
    def __init__(self):
        super(model,self).__init__()
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,128)
        self.out = nn.Linear(128,10)
        self.lr = 0.01
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self,x):
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
    
    def configure_optimizers(self):
        return SGD(self.parameters(),lr = self.lr)
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.loss(logits,y)
        return loss
    
    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        logits = self.forward(x)
        loss = self.loss(logits,y)
```
### Train model in Lightning:
```python
# Create Model Object
clf = model()
# Create Data Module Object
mnist = Data()
# Create Trainer Object
trainer = pl.Trainer(gpus=1,accelerator='dp',max_epochs=5)
trainer.fit(clf,mnist)
```
> Note: `dp` is DataParallel (split batch among GPUs of same machine).
> Note: If you have loaded data by creating dataloaders you can fit trainer by trainer.fit(clf,trainloader,testloader).

## Difference Between PyTorch Model and Lightning Model
### class
 the first difference between PyTorch and lightning model is the class that the model class inherits:-

  - PyTorch
    ```python
    class model(nn.Module):
    ```
  - PyTorch-Lightning
    ```python
    class model(pl.LightningModule):
    __init__() method
    ```
### `__init__()` method
In both Pytorch and and Lightning Model we use the __init__() method to define our layers, since in lightning we club everything together we can also define other hyper parameters like learning rate for optimizer and the loss function.

  - PyTorch
    ```python
    def __init__(self):
        super(model,self).__init__()
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,128)
        self.out = nn.Linear(128,10)
    ```
  - Pytorch-Lightning
    ```python
    def __init__(self):
        super(model,self).__init__()
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,128)
        self.out = nn.Linear(128,10)
        self.lr = 0.01
        self.loss = nn.CrossEntropyLoss()
    ```
### `forward()` method:
In both Pytorch and Lightning Model we use the forward() method to define our forward pass, hence it is same for both.

  - PyTorch and PyTorch-Lightning
    ```python
    def forward(self,x):
      batch_size, _, _, _ = x.size()
      x = x.view(batch_size,-1)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      return self.out(x)
    ```
### Defining Optimizer:

In PyTorch, we usually define our optimizers by directly creating their object but in PyTorch-lightning we define our optimizers under configure_optimizers() method. Another thing to note is that in PyTorch we pass model object parameters as the arguments for optimizer but in lightning, we pass self.parameters() since the class is the model itself.
  - pytorch
    ```PYTHON
    from torch.optim import SGD
    clf = model()    # Pytorch Model Object
    optimizer = SGD(clf.parameters(),lr=0.01)
    PyTorch-Lightning
    def configure_optimizers(self):
        return SGD(self.parameters(),lr = self.lr)
    ```
  - PyTorch-Lightning
    ```python
    def configure_optimizers(self):
      return SGD(self.parameters(),lr = self.lr)
    ```
> Note: You can create multiple optimizers in lightning too.

### Training Loop(Step):

It won’t be wrong to say that this is what makes Lightning stand out from PyTorch. In PyTorch we define the full training loop while in lightning we use the Trainer() to do the job. But we still define the steps that are going to be executed while training.

  - PyTorch
    ```python
    epochs = 5

    for i in range(epochs):
        train_loss = 0.0
        for data,label in trainloader:
            if is_gpu:
                data, label = data.cuda(), label.cuda()
            output = model(data)
            optimizer.zero_grad()
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
        print(f'Epoch: {i+1} / {epochs} \t\t\t Training Loss:{train_loss/len(trainloader)}')
    ```
  - PyTorch-Lightning
    ```python
    def training_step(self, train_batch, batch_idx):
      x, y = train_batch
      logits = self.forward(x)
      loss = self.loss(logits,y)
      return loss
    ```
See how in training steps we just write the steps necessary(bolded). 

## Code
```python
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
  
# training model
class model(pl.LightningModule):
    def __init__(self):
        super(model, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 10)
        self.lr = 0.01
        self.loss = nn.CrossEntropyLoss()
  
    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
  
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
  
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        return loss
  
    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
  
# data module
class Data(pl.LightningDataModule):
    def prepare_data(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
  
        self.train_data = datasets.MNIST(
            '', train=True, download=True, transform=transform)
        self.test_data = datasets.MNIST(
            '', train=False, download=True, transform=transform)
  
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=32, shuffle=True)
  
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=32, shuffle=True)
  
# Train
clf = model()
mnist = Data()
trainer = pl.Trainer(gpus=1, accelerator='dp', max_epochs=5)
trainer.fit(clf, mnist)
```

# Pytorch Lightning DataModule Format
PyTorch Lightning aims to make PyTorch code more structured and readable and that not just limited to the PyTorch Model but also the data itself. In PyTorch we use DataLoaders to train or test our model. While we can use DataLoaders in PyTorch Lightning to train the model too, PyTorch Lightning also provides us with a better approach called DataModules. DataModule is a reusable and shareable class that encapsulates the DataLoaders along with the steps required to process data. Creating dataloaders can get messy that’s why it’s better to club the dataset in the form of DataModule. Its recommended that you know how to define a neural network using PyTorch Lightning.

To define a Lightning DataModule we follow the following format:
```python
import pytorch-lightning as pl
from torch.utils.data import random_split, DataLoader

class DataModuleClass(pl.LightningDataModule):
    def __init__(self):
        #Define required parameters here
    
    def prepare_data(self):
        # Define steps that should be done
    # on only one GPU, like getting data.
    
    def setup(self, stage=None):
        # Define steps that should be done on 
    # every GPU, like splitting data, applying
    # transform etc.
    
    def train_dataloader(self):
        # Return DataLoader for Training Data here
    
    def val_dataloader(self):
        # Return DataLoader for Validation Data here
    
    def test_dataloader(self):
        # Return DataLoader for Testing Data here
```
> Note: The names of the above functions should be exactly the same.

## Understanding the DataModule Class
For this article, I’ll be using MNIST data as an example. As we can see, the first requirement to create a Lightning DataModule is to inherit the LightningDataModule class in pytorch-lightning:
```python
import pytorch-lightning as pl
from torch.utils.data import random_split, DataLoader

class DataModuleMNIST(pl.LightningDataModule):
```
### `__init__()` method:
It is used to store information regarding **batch size, transforms, etc**. 
```python
def __init__(self):
    super().__init__()
    self.download_dir = ''
    self.batch_size = 32
    self.transform = transforms.Compose([
        transforms.ToTensor()
    ])
```
### `prepare_data()` method:
This method is used to define the processes that are meant to be performed by only one GPU. **It’s usually used to handle the task of downloading the data**. 
```python
def prepare_data(self):
    datasets.MNIST(self.download_dir,
           train=True, download=True)
           
    datasets.MNIST(self.download_dir, train=False,        
           download=True)
```
### `setup()` method:
This method is used to define the process that is meant to be performed by all the available GPU. **It’s usually used to handle the task of loading the data**. 
```python
def setup(self, stage=None):
    data = datasets.MNIST(self.download_dir,
             train=True, transform=self.transform)
             
    self.train_data, self.valid_data = random_split(data, [55000, 5000])
        
    self.test_data = datasets.MNIST(self.download_dir,
                        train=False, transform=self.transform)
```

### `train_dataloader()` method:
This method is used to create a training data dataloader. **In this function, you usually just return the dataloader of training data.**
```python
def train_dataloader(self):
    return DataLoader(self.train_data, batch_size=self.batch_size)
```

### `val_dataloader()` method:
This method is used to create a validation data dataloader. In this function, you usually just return the dataloader of validation data.
```python
def val_dataloader(self):
   return DataLoader(self.valid_data, batch_size=self.batch_size)
```

### `test_dataloader()` method:
This method is used to create a testing data dataloader. In this function, you usually just return the dataloader of testing data.
```python 
def test_dataloader(self):
   return DataLoader(self.test_data, batch_size=self.batch_size)
```

## Training Pytorch Lightning Model Using DataModule:
In Pytorch Lighting, we use Trainer() to train our model and in this, we can pass the data as DataLoader or DataModule. Let’s use the model I defined in this article here as an example:
```python
class model(pl.LightningModule): 
    def __init__(self): 
        super(model, self).__init__() 
        self.fc1 = nn.Linear(28*28, 256) 
        self.fc2 = nn.Linear(256, 128) 
        self.out = nn.Linear(128, 10) 
        self.lr = 0.01
        self.loss = nn.CrossEntropyLoss() 
  
    def forward(self, x): 
        batch_size, _, _, _ = x.size() 
        x = x.view(batch_size, -1) 
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        return self.out(x) 
  
    def configure_optimizers(self): 
        return torch.optim.SGD(self.parameters(), lr=self.lr) 
  
    def training_step(self, train_batch, batch_idx): 
        x, y = train_batch 
        logits = self.forward(x) 
        loss = self.loss(logits, y) 
        return loss 
  
    def validation_step(self, valid_batch, batch_idx): 
        x, y = valid_batch 
        logits = self.forward(x) 
        loss = self.loss(logits, y)
```
Now to train this model we’ll create a Trainer() object and fit() it by passing our model and datamodules as parameters.
```python
clf = model() 
mnist = DataModuleMNIST() 
trainer = pl.Trainer(gpus=1) 
trainer.fit(clf, mnist)
```

Below is the full implementation:

```python

# import module
import torch 
  
# To get the layers and losses for our model
from torch import nn 
import pytorch_lightning as pl 
  
# To get the activation function for our model
import torch.nn.functional as F 
  
# To get MNIST data and transforms
from torchvision import datasets, transforms
  
# To get the optimizer for our model
from torch.optim import SGD 
  
# To get random_split to split training
# data into training and validation data
# and DataLoader to create dataloaders for train, 
# valid and test data to be returned
# by our data module
from torch.utils.data import random_split, DataLoader 
  
class model(pl.LightningModule): 
    def __init__(self): 
        super(model, self).__init__() 
          
        # Defining our model architecture
        self.fc1 = nn.Linear(28*28, 256) 
        self.fc2 = nn.Linear(256, 128) 
        self.out = nn.Linear(128, 10) 
          
        # Defining learning rate
        self.lr = 0.01
          
        # Defining loss 
        self.loss = nn.CrossEntropyLoss() 
    
    def forward(self, x):
        
          # Defining the forward pass of the model
        batch_size, _, _, _ = x.size() 
        x = x.view(batch_size, -1) 
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        return self.out(x) 
    
    def configure_optimizers(self):
        
          # Defining and returning the optimizer for our model
        # with the defines parameters
        return torch.optim.SGD(self.parameters(), lr = self.lr) 
    
    def training_step(self, train_batch, batch_idx): 
        
          # Defining training steps for our model
        x, y = train_batch 
        logits = self.forward(x) 
        loss = self.loss(logits, y) 
        return loss 
    
    def validation_step(self, valid_batch, batch_idx): 
        
        # Defining validation steps for our model
        x, y = valid_batch 
        logits = self.forward(x) 
        loss = self.loss(logits, y)
  
class DataModuleMNIST(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
          
        # Directory to store MNIST Data
        self.download_dir = ''
          
        # Defining batch size of our data
        self.batch_size = 32
          
        # Defining transforms to be applied on the data
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
  
    def prepare_data(self):
        
          # Downloading our data
        datasets.MNIST(self.download_dir, 
                       train = True, download = True)
          
        datasets.MNIST(self.download_dir,
                       train = False, download = True)
  
    def setup(self, stage=None):
        
          # Loading our data after applying the transforms
        data = datasets.MNIST(self.download_dir,
                              train = True, 
                              transform = self.transform)
          
        self.train_data, self.valid_data = random_split(data,
                                                        [55000, 5000])
  
        self.test_data = datasets.MNIST(self.download_dir,
                                        train = False,
                                        transform = self.transform)
  
    def train_dataloader(self):
        
          # Generating train_dataloader
        return DataLoader(self.train_data, 
                          batch_size = self.batch_size)
  
    def val_dataloader(self):
        
          # Generating val_dataloader
        return DataLoader(self.valid_data,
                          batch_size = self.batch_size)
  
    def test_dataloader(self):
        
        # Generating test_dataloader
        return DataLoader(self.test_data,
                          batch_size = self.batch_size)
  
clf = model() 
mnist = DataModuleMNIST() 
trainer = pl.Trainer()
trainer.fit(clf, mnist) 
```

Output:
![Output](https://media.geeksforgeeks.org/wp-content/uploads/20201201184016/Screenshotfrom20201201183750-660x278.png)