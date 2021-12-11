from style_transfer.train_utils import (
    TensorflowDatasetLoader,
    train_step,
    get_loss_model,
)
from utils.parse_arguments import get_train_arguments
from style_transfer.utility import load_image, load_url_image
from style_transfer.transform_net import ImageTransformNet
import tensorflow as tf
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Train stylized model with style image")

parser.add_argument("--config", "-con", help="Path to training config file")
parser.add_argument("--checkpoint", "-ckpt", help="Path to save Trained Checkpoints")
parser.add_argument("--style_image", "-style", help="url or file path to style image")
parser.add_argument(
    "--train_path",
    "-tp",
    default="data/train",
    help="Dataset path where train folder with images is located",
)
parser.add_argument(
    "--input_shape",
    "-shape",
    nargs="+",
    type=int,
    default=[256, 256],
    help="input image shape for training network",
)
parser.add_argument(
    "--batch_size", "-batch", default=4, type=int, help="training batch size"
)
parser.add_argument(
    "--learning_rate", "-lr", default=1e-3, type=float, help="number of training epochs"
)
parser.add_argument(
    "--epochs", "-e", default=2, type=int, help="number of training epochs"
)
parser.add_argument(
    "--content_layers_weights",
    "-clw",
    nargs="+",
    default=[1],
    type=float,
    help="Content weight with respect to each content layer respectively",
)
parser.add_argument(
    "--style_layers_weights",
    "-slw",
    nargs="+",
    default=[1, 1, 1, 1, 1],
    type=float,
    help="Style weight with respect to each style layer respectively",
)
parser.add_argument(
    "--content_weight", "-cw", default=1e1, type=float, help="Content weight"
)
parser.add_argument(
    "--style_weight", "-sw", default=1e4, type=float, help="Style weight"
)
parser.add_argument(
    "--total_variation_weight",
    "-tvw",
    default=0.004,
    type=float,
    help="Total variation weight",
)
parser.add_argument(
    "--content_layers",
    "-cl",
    nargs="+",
    default=["block4_conv2"],
    help="Content layers to calculate loss on",
)
parser.add_argument(
    "--style_layers",
    "-sl",
    nargs="+",
    default=[
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ],
    help="Style layers to calculate loss on",
)

args = get_train_arguments(parser.parse_args())

try:
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    tf.keras.mixed_precision.experimental.set_policy(policy)
except Exception as e:
    print("Mixed precision not enabled: ", e)

input_shape = args["input_shape"]
model_save_path = args["checkpoint"]
epochs = args["epochs"]
batch_size = args["batch_size"]

loader = TensorflowDatasetLoader(args["train_path"], batch_size, input_shape)
dataset = loader.dataset
print("Dataset loaded! Number of images: ", len(loader))
print("Dataset Specs: ", dataset.element_spec)
steps_per_epoch = len(loader) // batch_size

style_path = args["style_image"]
if style_path.startswith("http"):
    style_image = load_url_image(style_path, dim=input_shape, resize=True)
else:
    style_image = load_image(style_path, dim=input_shape, resize=True)

style_image = style_image / 255.0
style_image = style_image.astype(np.float32)
style_image_batch = np.repeat([style_image], batch_size, axis=0)
print("Style image loaded")
print("Style image specs", style_image_batch.shape)

loss_model = get_loss_model(args["content_layers"], args["style_layers"])
print("loss model loaded")

style_activations = loss_model.get_activations(style_image_batch)["style"]

os.makedirs(model_save_path, exist_ok=True)

style_model = ImageTransformNet()
if os.path.isfile(os.path.join(model_save_path, "model_checkpoint.ckpt.index")):
    style_model.load_weights(os.path.join(model_save_path, "model_checkpoint.ckpt"))
    print("resume training ...")
else:
    print("training from scratch ...")

optimizer = tf.keras.optimizers.Adam(learning_rate=args["learning_rate"])

print("training ...")
for epoch in range(1, epochs + 1):
    print(f"epoch: {epoch}")
    batch_loss = train_step(
        dataset,
        style_activations,
        steps_per_epoch,
        style_model,
        loss_model,
        optimizer,
        model_save_path,
        args["content_weight"],
        args["style_weight"],
        args["total_variation_weight"],
        args["content_layers_weights"],
        args["style_layers_weights"],
    )
    style_model.save_weights(os.path.join(model_save_path, "model_checkpoint.ckpt"))
    print("Model Checkpointed at: ", model_save_path)
    print(f"loss: {batch_loss.numpy()}")
