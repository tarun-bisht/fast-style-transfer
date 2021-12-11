from utils.utility import load_image, load_url_image, array_to_img
from utils.parse_arguments import get_optimization_style_image_arguments
from style_transfer.train_utils import get_loss_model, gatys_train_step
import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description="Create Styled Images using optimization")

parser.add_argument("--config", "-con", help="Path to style image config file")
parser.add_argument("--style", "-s", help="url or file path to style image")
parser.add_argument("--image", "-img", help="url or file path to image to style")
parser.add_argument(
    "--image_size",
    "-size",
    nargs="+",
    type=int,
    default=[256, 256],
    help="output image size",
)
parser.add_argument(
    "--output",
    "-out",
    default="output/optimization_styled.jpg",
    help="Path to save output image",
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

args = get_optimization_style_image_arguments(parser.parse_args())


input_shape = args["image_size"]
content_path = args["image"]
style_path = args["style"]
epochs = args["epochs"]

if content_path.startswith("http"):
    content_image = load_url_image(content_path, dim=input_shape, resize=True)
else:
    content_image = load_image(content_path, dim=input_shape, resize=True)
content_image = content_image / 255.0
content_image = content_image.astype(np.float32)
content_image = np.expand_dims(content_image, axis=0)
print("Content image loaded ...")

if style_path.startswith("http"):
    style_image = load_url_image(style_path, dim=input_shape, resize=True)
else:
    style_image = load_image(style_path, dim=input_shape, resize=True)
style_image = style_image / 255.0
style_image = style_image.astype(np.float32)
style_image = np.expand_dims(style_image, axis=0)
print("Style image loaded ...")

loss_model = get_loss_model(args["content_layers"], args["style_layers"])
print("loss model loaded")

content_activations = loss_model.get_activations(content_image)["content"]
style_activations = loss_model.get_activations(style_image)["style"]

image = tf.Variable(content_image)

optimizer = tf.keras.optimizers.Adam(learning_rate=args["learning_rate"])

steps_per_epoch = 100

print("training ...")
for epoch in range(1, epochs + 1):
    print(f"epoch: {epoch}")
    batch_loss = gatys_train_step(
        image,
        content_activations,
        style_activations,
        steps_per_epoch,
        loss_model,
        optimizer,
        args["content_weight"],
        args["style_weight"],
        args["total_variation_weight"],
        args["content_layers_weights"],
        args["style_layers_weights"],
    )
    current_image = array_to_img(image.numpy() * 255.0)
    current_image.save(f'{args["output"]}')
    print(f" loss: {batch_loss.numpy()}")
