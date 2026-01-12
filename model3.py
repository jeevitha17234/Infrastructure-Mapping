# Setup
import os

import tensorflow as tf
from keras import layers, callbacks, metrics, utils, optimizers
import pandas as pd
import numpy as np
import cv2

tf.random.set_seed(123)

# Define the paths and filenames
annotation_folder = "/dataset/"
train_path = "train/indoors"
val_path = "val/indoors"

# Download and extract validation dataset
if not os.path.exists(os.path.abspath(".") + annotation_folder):
    os.makedirs(os.path.abspath(".") + annotation_folder)
    annotation_zip = utils.get_file(
        "train.tar.gz",
        cache_subdir=os.path.abspath(".") + annotation_folder,
        origin="http://diode-dataset.s3.amazonaws.com/train.tar.gz",
        extract=True,
    )
    annotation_zip = utils.get_file(
        "val.tar.gz",
        cache_subdir=os.path.abspath(".") + annotation_folder,
        origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
        extract=True,
    )

train_filelist = []
for root, dirs, files in os.walk(train_path):
    for file in files:
        train_filelist.append(os.path.join(root, file))
train_filelist.sort()

val_filelist = []
for root, dirs, files in os.walk(val_path):
    for file in files:
        val_filelist.append(os.path.join(root, file))
val_filelist.sort()

print("Number of training images:", len(train_filelist))    
print("Number of validation images:", len(val_filelist))

# Create DataFrames for training and validation datasets
train_data = {
    "image": [x for x in train_filelist if x.endswith(".png")],
    "depth": [x for x in train_filelist if x.endswith("_depth.npy")],
    "mask": [x for x in train_filelist if x.endswith("_depth_mask.npy")],
}

val_data = {
    "image": [x for x in val_filelist if x.endswith(".png")],
    "depth": [x for x in val_filelist if x.endswith("_depth.npy")],
    "mask": [x for x in val_filelist if x.endswith("_depth_mask.npy")],
}

# Create DataFrames
tdf = pd.DataFrame(train_data)
vdf = pd.DataFrame(val_data)

# Optionally shuffle the data
tdf = tdf.sample(frac=1, random_state=42)
vdf = vdf.sample(frac=1, random_state=42)

print("Training DataFrame shape:", tdf.shape)
print("Validation DataFrame shape:", vdf.shape)

# Hyperparameters
HEIGHT = 256
WIDTH = 256
LR = 0.0002
EPOCHS = 50
BATCH_SIZE = 4

# Data Generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=6, dim=(768, 1024), n_channels=3, shuffle=True):
        self.data = data
        self.indices = self.data.index.tolist()
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_depth = 0.1
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
        index = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        x, y = self.data_generation(batch)

        return x, y

    def on_epoch_end(self):
        
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def load(self, image_path, depth_map, mask):

        image_ = cv2.imread(image_path)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image_ = cv2.resize(image_, self.dim)
        image_ = tf.image.convert_image_dtype(image_, tf.float32)

        depth_map = np.load(depth_map).squeeze()

        mask = np.load(mask)
        mask = mask > 0

        max_depth = min(300, np.percentile(depth_map, 99))
        depth_map = np.clip(depth_map, self.min_depth, max_depth)
        depth_map = np.log(depth_map, where=mask)

        depth_map = np.ma.masked_where(~mask, depth_map)

        depth_map = np.clip(depth_map, 0.1, np.log(max_depth))
        depth_map = cv2.resize(depth_map, self.dim)
        depth_map = np.expand_dims(depth_map, axis=2)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

        return image_, depth_map

    def data_generation(self, batch):
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        for i, batch_id in enumerate(batch):
            x[i,], y[i,] = self.load(
                self.data["image"][batch_id],
                self.data["depth"][batch_id],
                self.data["mask"][batch_id],
            )

        return x, y

# Building the model
class DownscaleBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2a = layers.BatchNormalization()
        self.bn2b = layers.BatchNormalization()

        self.pool = layers.MaxPool2D((2, 2), (2, 2))

    def call(self, input_tensor):
        d = self.convA(input_tensor)
        x = self.bn2a(d)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        x += d
        p = self.pool(x)
        return x, p

class UpscaleBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.us = layers.UpSampling2D((2, 2))
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2a = layers.BatchNormalization()
        self.bn2b = layers.BatchNormalization()
        self.conc = layers.Concatenate()

    def call(self, x, skip):
        x = self.us(x)
        concat = self.conc([x, skip])
        x = self.convA(concat)
        x = self.bn2a(x)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        return x

class BottleNeckBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        x = self.convA(x)
        x = self.reluA(x)
        x = self.convB(x)
        x = self.reluB(x)
        return x

# Defining the loss
def ssim_loss(target, pred):
    return tf.reduce_mean(1 - tf.image.ssim(target, pred, max_val=WIDTH))

def l1_loss(target, pred):
    return tf.reduce_mean(tf.abs(target - pred))

def edge_loss(target, pred):
    dy_true, dx_true = tf.image.image_gradients(target)
    dy_pred, dx_pred = tf.image.image_gradients(pred)
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y
    depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(
        abs(smoothness_y)
    )
    return depth_smoothness_loss

def total_loss(target, pred):
    ssim = ssim_loss(target, pred)
    l1 = l1_loss(target, pred)
    edge = edge_loss(target, pred)
    return 0.85 * ssim + 0.1 * l1 + 0.9 * edge

# Building the model
class DepthEstimationModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        f = [16, 32, 64, 128, 256]
        self.downscale_blocks = [
            DownscaleBlock(f[0]),
            DownscaleBlock(f[1]),
            DownscaleBlock(f[2]),
            DownscaleBlock(f[3]),
        ]
        self.bottle_neck_block = BottleNeckBlock(f[4])
        self.upscale_blocks = [
            UpscaleBlock(f[3]),
            UpscaleBlock(f[2]),
            UpscaleBlock(f[1]),
            UpscaleBlock(f[0]),
        ]
        self.conv_layer = layers.Conv2D(1, (1, 1), padding="same", activation="tanh")

    def train_step(self, batch_data):
        input, target = batch_data
        with tf.GradientTape() as tape:
            pred = self(input, training=True)
            loss = total_loss(target, pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, batch_data):
        input, target = batch_data
        pred = self(input, training=False)
        loss = total_loss(target, pred)
        return {"loss": loss}

    def call(self, x):
        c1, p1 = self.downscale_blocks[0](x)
        c2, p2 = self.downscale_blocks[1](p1)
        c3, p3 = self.downscale_blocks[2](p2)
        c4, p4 = self.downscale_blocks[3](p3)

        bn = self.bottle_neck_block(p4)

        u1 = self.upscale_blocks[0](bn, c4)
        u2 = self.upscale_blocks[1](u1, c3)
        u3 = self.upscale_blocks[2](u2, c2)
        u4 = self.upscale_blocks[3](u3, c1)

        return self.conv_layer(u4)

# Model training
optimizer = optimizers.Adam(learning_rate=LR, amsgrad=False, clipvalue=1.0)
model = DepthEstimationModel()
model.compile(optimizer)

train_loader = DataGenerator(data=tdf, batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH))
validation_loader = DataGenerator(data=vdf, batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH))
print(" Training data loader length:", len(train_loader),"\n","Validation data loader length:", len(validation_loader))

ReduceLROnPlateau = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1, min_lr=0.00001, mode="min")

model.fit(
    train_loader,
    steps_per_epoch=len(train_loader),
    callbacks=[ReduceLROnPlateau],
    epochs=EPOCHS,
    validation_data=validation_loader,
)

model.save("saved_model")
utils.plot_model(model, to_file='depthmodel_arch.png', show_shapes=True, show_layer_names=True)
model.summary()

