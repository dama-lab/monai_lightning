#%% following the torchio tutorial
# > https://colab.research.google.com/github/fepegar/torchio-notebooks/blob/main/notebooks/TorchIO_tutorial.ipynb#scrollTo=b0NdJiFW3Uy7
%load_ext autoreload
%autoreload 2
#%% Config
seed = 42  # for reproducibility
training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
num_epochs = 5

# If the following values are False, the models will be downloaded and not computed
compute_histograms = False
train_whole_images = False 
train_patches = False

#%% import libraries
import enum
import time
import random
import multiprocessing
from pathlib import Path

import torch
import torchvision
import torchio as tio
import torch.nn.functional as F

import numpy as np
from unet import UNet
from scipy import stats
import matplotlib.pyplot as plt

from IPython import display
from tqdm.notebook import tqdm

#%%
random.seed(seed)
torch.manual_seed(seed)
num_workers = multiprocessing.cpu_count()
%config InlineBackend.figure_format = 'retina'
plt.rcParams['figure.figsize'] = 12, 6

print('TorchIO version:', tio.__version__)
#%% defining data
root_dir = Path("/project/6003102/dma73/Data/Brain_MRI/T1_Hypo_T2_Hyper/RAW_DATA/UBCMIXDEM_WMHT1T2relationships/")
processed_dir = root_dir/"PROCESSED_DIR"
processed_dir.mkdir(exist_ok=True, parents=True)
histogram_landmarks_path = f'{processed_dir}/landmarks.npy'
images_dir = Path(f"{root_dir}/RAW_DATA/")
image_paths = sorted(images_dir.glob("*_T1W.nii.gz"))
# label_paths = sorted(images_dir.glob("*_T2HYPERWMSAinT1W.nii.gz"))
label_paths = sorted(images_dir.glob("*_T1HYPOWMSAinT1W.nii.gz"))

assert len(image_paths) == len(label_paths)

#%% define subject object
subjects = []
for (image_path, label_path) in zip(image_paths, label_paths):
    subject = tio.Subject(
        mri = tio.ScalarImage(image_path),
        brain = tio.LabelMap(label_path),
    )
    subjects.append(subject)

dataset = tio.SubjectsDataset(subjects)
print('Dataset size:', len(dataset), 'subjects')

#%% Look at one subject
one_subject = dataset[0]
one_subject.plot()
#%% check images
print(one_subject)
print(one_subject.mri)
print(one_subject.brain)

#%% transform preprocessing pipeline
def plot_histogram(axis, tensor, num_positions=100, label=None, alpha=0.05, color=None):
    values = tensor.numpy().ravel()
    kernel = stats.gaussian_kde(values)
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    histogram = kernel(positions)
    kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
    if label is not None:
        kwargs['label'] = label
    axis.plot(positions, histogram, **kwargs)
#%% plot histogram equalization
paths = image_paths
color = None
if compute_histograms:
    fig, ax = plt.subplots(dpi=100)
    for path in tqdm(paths):
        tensor = tio.ScalarImage(path).data
        if 'HH' in path.name: color = 'red'
        elif 'Guys' in path.name: color = 'green'
        elif 'IOP' in path.name: color = 'blue'
        plot_histogram(ax, tensor, color=color)
    ax.set_xlim(-100, 2000)
    ax.set_ylim(0, 0.004);
    ax.set_title('Original histograms of all samples')
    ax.set_xlabel('Intensity')
    ax.grid()
    graph = None
else:
    graph = display.Image(url='https://www.dropbox.com/s/daqsg3udk61v65i/hist_original.png?dl=1')
graph
#%% histogram equalization
landmarks = tio.HistogramStandardization.train(
    image_paths,
    output_path=histogram_landmarks_path,
)
np.set_printoptions(suppress=True, precision=3)
print('\nTrained landmarks:', landmarks)

#%% compute_histograms=False
landmarks_dict = {'mri': landmarks}
histogram_transform = tio.HistogramStandardization(landmarks_dict)

if compute_histograms:
    fig, ax = plt.subplots(dpi=100)
    for i ,sample in enumerate(tqdm(dataset)):
        standard = histogram_transform(sample)
        tensor = standard.mri.data
        path = str(sample.mri.path)
        if 'HH' in path: color = 'red'
        elif 'Guys' in path: color = 'green'
        elif 'IOP' in path: color = 'blue'
        plot_histogram(ax, tensor, color=color)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 0.02)
    ax.set_title('Intensity values of all samples after histogram standardization')
    ax.set_xlabel('Intensity')
    ax.grid()
    graph = None
else:
    graph = display.Image(url='https://www.dropbox.com/s/dqqaf78c86mrsgn/hist_standard.png?dl=1')

# %% visualize before z-normalization
sample = dataset[0]
fig, ax = plt.subplots(dpi=100)
plot_histogram(ax, sample.mri.data, label='Z-normed', alpha=1)
ax.set_title('Intensity values of one sample before z-normalization')
ax.set_xlabel('Intensity')
ax.grid()

#%% znorm
sample = dataset[0]
znorm_transform = tio.ZNormalization(masking_method=tio.ZNormalization.mean)

transform = tio.Compose([histogram_transform, znorm_transform])
znormed = transform(sample)

fig, ax = plt.subplots(dpi=100)
plot_histogram(ax, znormed.mri.data, label='Z-normed', alpha=1)
ax.set_title('Intensity values of one sample after z-normalization')
ax.set_xlabel('Intensity')
ax.grid()

# %% Prepare dataset/dataloader to train a network
num_subjects = len(dataset)
num_training_subjects = int(training_split_ratio * num_subjects)
num_validation_subjects = num_subjects - num_training_subjects

num_split_subjects = num_training_subjects, num_validation_subjects
training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

#%%
training_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(4),
    tio.CropOrPad((48, 60, 48)),
    tio.RandomMotion(p=0.2),
    tio.HistogramStandardization({'mri': landmarks}),
    tio.RandomBiasField(p=0.3),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.RandomNoise(p=0.5),
    tio.RandomFlip(),
    tio.OneOf({
        tio.RandomAffine(): 0.8,
        tio.RandomElasticDeformation(): 0.2,
    }),
    tio.OneHot(num_classes=2),
])

validation_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(4),
    tio.CropOrPad((48, 60, 48)),
    tio.HistogramStandardization({'mri': landmarks}),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.OneHot(num_classes=2),
])

training_set = tio.SubjectsDataset(
    training_subjects, transform=training_transform)

validation_set = tio.SubjectsDataset(
    validation_subjects, transform=validation_transform)

print('Training set:', len(training_set), 'subjects')
print('Validation set:', len(validation_set), 'subjects')

# %% visualize training
training_instance = training_set[14]  # transform is applied inside SubjectsDataset
training_instance.plot()
# %% visualize validation
validation_instance = validation_set[1]
validation_instance.plot()

#%% prepare Training using the whole image
training_batch_size = 2
validation_batch_size = 2 # training_batch_size * 2

training_loader = torch.utils.data.DataLoader(
    training_set,
    batch_size=training_batch_size,
    shuffle=True,
    num_workers=num_workers,
)

validation_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=validation_batch_size,
    num_workers=num_workers,
)
#%% check image size
for i,one_batch in enumerate(iter(training_loader)):
    print(i, one_batch['mri']['data'].shape,one_batch['brain']['data'].shape)
    # break
#%% Test batch/channel size
one_batch = next(iter(training_loader))
print(one_batch['mri']['data'].shape,one_batch['brain']['data'].shape)

# %%
k = one_batch['mri'][tio.DATA].shape[-1]//2 # mid-slice
batch_mri = one_batch['mri'][tio.DATA][..., k]
batch_label = one_batch['brain'][tio.DATA][:, 1:, ..., k]
slices = torch.cat((batch_mri, batch_label))
image_path = f'{processed_dir}/batch_whole_images.png'
torchvision.utils.save_image(
    slices,
    image_path,
    nrow=training_batch_size//2,
    normalize=True,
    scale_each=True,
    padding=0,
)
display.Image(image_path)


#%% Deep learning stuff
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'

def prepare_batch(batch, device):
    inputs = batch['mri'][tio.DATA].to(device)
    targets = batch['brain'][tio.DATA].to(device)
    return inputs, targets

def get_dice_score(output, target, epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score

def get_dice_loss(output, target):
    return 1 - get_dice_score(output, target)

def get_model_and_optimizer(device):
    model = UNet(
        in_channels=1,
        out_classes=2,
        dimensions=3,
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        normalization='batch',
        upsampling_type='linear',
        padding=True,
        activation='PReLU',
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer

def run_epoch(epoch_idx, action, loader, model, optimizer):
    is_training = action == Action.TRAIN
    epoch_losses = []
    times = []
    model.train(is_training)
    for batch_idx, batch in enumerate(tqdm(loader)):
        inputs, targets = prepare_batch(batch, device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            logits = model(inputs)
            probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
            batch_losses = get_dice_loss(probabilities, targets)
            batch_loss = batch_losses.mean()
            if is_training:
                batch_loss.backward()
                optimizer.step()
            times.append(time.time())
            epoch_losses.append(batch_loss.item())
    epoch_losses = np.array(epoch_losses)
    print(f'{action.value} mean loss: {epoch_losses.mean():0.3f}')
    return times, epoch_losses

def train(num_epochs, training_loader, validation_loader, model, optimizer, weights_stem):
    train_losses = []
    val_losses = []
    val_losses.append(run_epoch(0, Action.VALIDATE, validation_loader, model, optimizer))
    for epoch_idx in range(1, num_epochs + 1):
        print('Starting epoch', epoch_idx)
        train_losses.append(run_epoch(epoch_idx, Action.TRAIN, training_loader, model, optimizer))
        val_losses.append(run_epoch(epoch_idx, Action.VALIDATE, validation_loader, model, optimizer))
        torch.save(model.state_dict(), f'{weights_stem}_epoch_{epoch_idx}.pth')
    return np.array(train_losses), np.array(val_losses)

# %% Train
model, optimizer = get_model_and_optimizer(device)
weights_path = 'whole_image_state_dict.pth'
if train_whole_images:
    weights_stem = 'whole_images'
    train_losses, val_losses = train(num_epochs, training_loader, validation_loader, model, optimizer, weights_stem)
    checkpoint = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'weights': model.state_dict(),
    }
    torch.save(checkpoint, weights_path)
else:
    weights_path = 'whole_image_state_dict.pth'
    weights_url = 'https://github.com/fepegar/torchio-data/raw/master/models/whole_images_epoch_5.pth'
    !curl --location --silent --output {weights_path} {weights_url}
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['weights'])
    train_losses, val_losses = checkpoint['train_losses'], checkpoint['val_losses']

def plot_times(axis, losses, label):
    from datetime import datetime
    times, losses = losses.transpose(1, 0, 2)
    times = [datetime.fromtimestamp(x) for x in times.flatten()]
    axis.plot(times, losses.flatten(), label=label)
    
fig, ax = plt.subplots()
plot_times(ax, train_losses, 'Training')
plot_times(ax, val_losses, 'Validation')
ax.grid()
ax.set_xlabel('Time')
ax.set_ylabel('Dice loss')
ax.set_title('Training with whole images')
ax.legend()
fig.autofmt_xdate()

# %% Test
batch = next(iter(validation_loader))
model.eval();
inputs, targets = prepare_batch(batch, device)
FIRST = 0
FOREGROUND = 1
with torch.no_grad():
    probabilities = model(inputs).softmax(dim=1)[:, FOREGROUND:].cpu()
affine = batch['mri'][tio.AFFINE][0].numpy()
subject = tio.Subject(
    mri=tio.ScalarImage(tensor=batch['mri'][tio.DATA][FIRST], affine=affine),
    label=tio.LabelMap(tensor=batch['brain'][tio.DATA][FIRST], affine=affine),
    predicted=tio.ScalarImage(tensor=probabilities[FIRST], affine=affine),
)
subject.plot(figsize=(9, 8), cmap_dict={'predicted': 'RdBu_r'})

# %%
with torch.no_grad():
    # probabilities = model(inputs).softmax(dim=1)[:, FOREGROUND:].cpu()
    seg = model(inputs).max(dim=1,keepdim=True)[1].cpu()
affine = batch['mri'][tio.AFFINE][0].numpy()
subject = tio.Subject(
    mri=tio.ScalarImage(tensor=batch['mri'][tio.DATA][FIRST], affine=affine),
    label=tio.LabelMap(tensor=batch['brain'][tio.DATA][FIRST], affine=affine),
    predicted=tio.ScalarImage(tensor=seg[FIRST], affine=affine),
)
subject.plot(figsize=(9, 8), cmap_dict={'predicted': 'RdBu_r'})



# %%%%%% =========== Patch-based ==========

training_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(4),
    # tio.CropOrPad((48, 60, 48)),
    tio.RandomMotion(p=0.2),
    tio.HistogramStandardization({'mri': landmarks}),
    tio.RandomBiasField(p=0.3),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.RandomNoise(p=0.5),
    tio.RandomFlip(),
    tio.OneOf({
        tio.RandomAffine(): 0.8,
        tio.RandomElasticDeformation(): 0.2,
    }),
    tio.OneHot(num_classes=2),
])

validation_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(4),
    # tio.CropOrPad((48, 60, 48)),
    tio.HistogramStandardization({'mri': landmarks}),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.OneHot(num_classes=2),
])

num_subjects = len(dataset)
num_training_subjects = int(training_split_ratio * num_subjects)
num_validation_subjects = num_subjects - num_training_subjects

num_split_subjects = num_training_subjects, num_validation_subjects
training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

training_set = tio.SubjectsDataset(
    training_subjects, transform=training_transform)

validation_set = tio.SubjectsDataset(
    validation_subjects, transform=validation_transform)

print('Training set:', len(training_set), 'subjects')
print('Validation set:', len(validation_set), 'subjects')



# %% setup patch sampler
num_workers = multiprocessing.cpu_count()
patch_size = 24
samples_per_volume = 5
max_queue_length = 300
# sampler = tio.data.UniformSampler(patch_size)
sampler = tio.data.LabelSampler(patch_size=patch_size,
                                label_name='brain',
                                label_probabilities={0:0.5, 1:0.5})

training_batch_size = 16
validation_batch_size = 2 # * training_batch_size

# Queue: Ref: https://torchio.readthedocs.io/data/patch_training.html#queue
patches_training_set = tio.Queue(
    subjects_dataset=training_set,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=num_workers,
    shuffle_subjects=True,
    shuffle_patches=True,
)

patches_validation_set = tio.Queue(
    subjects_dataset=validation_set,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=num_workers,
    shuffle_subjects=False,
    shuffle_patches=False,
)

training_loader_patches = torch.utils.data.DataLoader(
    patches_training_set, batch_size=training_batch_size)

validation_loader_patches = torch.utils.data.DataLoader(
    patches_validation_set, batch_size=validation_batch_size)

#%% Test one batch/channel size
one_batch = next(iter(training_loader_patches))
print(one_batch['mri']['data'].shape,one_batch['brain']['data'].shape)

#%% check all batch/channel size
for i,one_batch in enumerate(iter(training_loader_patches)):
    print(i, one_batch['mri']['data'].shape,one_batch['brain']['data'].shape)
    # break

# %%
one_batch = next(iter(training_loader_patches))
k = int(patch_size // 4)
batch_mri = one_batch['mri'][tio.DATA][..., k]
batch_label = one_batch['brain'][tio.DATA][:, 1:, ..., k]
print(batch_mri.shape, batch_label.shape)
slices = torch.cat((batch_mri, batch_label))
image_path = 'batch_patches.png'
torchvision.utils.save_image(
    slices,
    image_path,
    nrow=training_batch_size,
    normalize=True,
    scale_each=True,
)
display.Image(image_path)
# %% Train
model, optimizer = get_model_and_optimizer(device)
weights_path = f'{processed_dir}/patches_state_dict.pth'
train_patches = True
num_epochs = 500

if train_patches:
    weights_stem = 'patches'
    train_losses, val_losses = train(
        num_epochs,
        training_loader_patches,
        validation_loader_patches,
        model,
        optimizer,
        weights_stem,
    )
    checkpoint = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'weights': model.state_dict(),
    }
    torch.save(checkpoint, weights_path)
else:
    weights_url = 'https://github.com/fepegar/torchio-data/raw/master/models/patches_epoch_5.pth'
    !curl --location --silent --output {weights_path} {weights_url}
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['weights'])
    train_losses, val_losses = checkpoint['train_losses'], checkpoint['val_losses']

#%% plot
def plot_times(axis, losses, label):
    from datetime import datetime
    times, losses = losses.transpose(1, 0, 2)
    times = [datetime.fromtimestamp(x) for x in times.flatten()]
    axis.plot(times, losses.flatten(), label=label)

fig, ax = plt.subplots()
plot_times(ax, train_losses, 'Training')
plot_times(ax, val_losses, 'Validation')
ax.grid()
ax.set_xlabel('Time')
ax.set_ylabel('Dice loss')
ax.set_title('Training with patches (subvolumes)')
ax.legend()
fig.autofmt_xdate()
# %% test - visualize probability
subject = random.choice(validation_set)
input_tensor = sample.mri.data[0]
patch_size = 48, 48, 48  # we can user larger patches for inference
patch_overlap = 4, 4, 4
grid_sampler = tio.inference.GridSampler(
    subject,
    patch_size,
    patch_overlap,
)
patch_loader = torch.utils.data.DataLoader(
    grid_sampler, batch_size=validation_batch_size)
aggregator = tio.inference.GridAggregator(grid_sampler)

m = model.eval();
with torch.no_grad():
    for patches_batch in patch_loader:
        inputs = patches_batch['mri'][tio.DATA].to(device)
        locations = patches_batch[tio.LOCATION]
        probabilities = model(inputs).softmax(dim=CHANNELS_DIMENSION)
        aggregator.add_batch(probabilities, locations)

foreground = aggregator.get_output_tensor()
affine = subject.mri.affine
prediction = tio.ScalarImage(tensor=foreground, affine=affine)
subject.add_image(prediction, 'prediction')
subject.plot(figsize=(9, 8), cmap_dict={'prediction': 'RdBu_r'})

# %% test - visualize seg
subject = random.choice(validation_set)
input_tensor = sample.mri.data[0]
patch_size = 48, 48, 48  # we can user larger patches for inference
patch_overlap = 4, 4, 4
grid_sampler = tio.inference.GridSampler(
    subject,
    patch_size,
    patch_overlap,
)
patch_loader = torch.utils.data.DataLoader(
    grid_sampler, batch_size=validation_batch_size)
aggregator = tio.inference.GridAggregator(grid_sampler)

m = model.eval();
with torch.no_grad():
    for patches_batch in patch_loader:
        inputs = patches_batch['mri'][tio.DATA].to(device)
        locations = patches_batch[tio.LOCATION]
        # probabilities = model(inputs).softmax(dim=CHANNELS_DIMENSION)
        seg = model(inputs).max(dim=CHANNELS_DIMENSION, keepdim=True)[1]
        aggregator.add_batch(seg, locations)

foreground = aggregator.get_output_tensor()
affine = subject.mri.affine
prediction = tio.ScalarImage(tensor=foreground, affine=affine)
subject.add_image(prediction, 'prediction')
subject.plot(figsize=(9, 8), cmap_dict={'prediction': 'RdBu_r'})

# %% change location to full path using `os.path.expandvars`
# ref https://stackoverflow.com/questions/52412297/how-to-replace-environment-variable-value-in-yaml-file-to-be-parsed-using-python
# os.path.abspath()
# https://www.geeksforgeeks.org/python-os-path-abspath-method-with-example/

for key, value in yaml_dict.items():
  if key == 'root_dir':
    print(f"{key}: {os.path.expandvars(value)}")
  else:
    print(f"{key}: {value}")

#%% test file copying speed:
rsync -r --progress --info=progress2

# %% SSHFS (not working properly)
# https://docs.pyfilesystem.org/en/latest/openers.html
# https://pypi.org/project/fs.sshfs/
import fs
cedar = fs.open_fs('ssh://cedar/')
from fs.sshfs import SSHFS
cedar = SSHFS(host="cedar")