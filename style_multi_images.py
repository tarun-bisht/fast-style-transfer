import os
from style_transfer.inference import inference
from style_transfer.transform_net import ImageTransformNet
from utils.utility import load_image, array_to_img
from utils.parse_arguments import get_style_multi_images_arguments
import argparse
import time

parser = argparse.ArgumentParser(description="Create Multiple Styled Images")

parser.add_argument("--config", "-con", help="Path to style multi images config file")
parser.add_argument("--checkpoint", "-ckpt", help="Path to trained style checkpoints")
parser.add_argument("--path", "-p", help="path to images folder to style")
parser.add_argument(
    "--image_size",
    "-size",
    nargs="+",
    type=int,
    default=[256, 256],
    help="output image size",
)
parser.add_argument(
    "--output", "-out", default="output", help="directory path to save output images"
)

args = get_style_multi_images_arguments(parser.parse_args())


path = args["path"]
images = os.listdir(path)
output_path = args["output"]

for img in images:
    style_model = ImageTransformNet()
    style_model.load_weights(args["checkpoint"])
    input_shape = args["image_size"]
    image_path = os.path.join(path, img)
    name = img.split(".")[0]
    image = load_image(image_path, dim=input_shape)
    start = time.time()
    styled_image = inference(style_model, image)
    end = time.time()
    print(f"Time Taken: {end-start:.2f}s")
    pil_image = array_to_img(styled_image)
    pil_image.save(f"{output_path}/{name}.jpg")
