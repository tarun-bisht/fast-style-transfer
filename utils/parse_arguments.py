import json


def get_train_arguments(args):
    if args.config:
        try:
            with open(args.config, "r") as f:
                train_args = json.load(f)
            return train_args
        except Exception as e:
            print("Error Occurred while loading config: ", e)

    train_args = {}
    if args.checkpoint:
        train_args["checkpoint"] = args.checkpoint
    else:
        raise Exception("No Path to save checkpoints found")
    if args.style_image:
        train_args["style_image"] = args.style_image
    else:
        raise Exception("No style image path found")
    train_args["train_path"] = args.train_path
    train_args["input_shape"] = args.input_shape
    train_args["batch_size"] = args.batch_size
    train_args["epochs"] = args.epochs
    train_args["content_layers_weights"] = args.content_layers_weights
    train_args["style_layers_weights"] = args.style_layers_weights
    train_args["content_weight"] = args.content_weight
    train_args["style_weight"] = args.style_weight
    train_args["total_variation_weight"] = args.total_variation_weight
    train_args["content_layers"] = args.content_layers
    train_args["style_layers"] = args.style_layers
    train_args["learning_rate"] = args.learning_rate
    return train_args


def get_style_image_arguments(args):
    if args.config:
        try:
            with open(args.config, "r") as f:
                style_args = json.load(f)
            return style_args
        except Exception as e:
            print("Error Occurred while loading config: ", e)

    style_args = {}
    if args.checkpoint:
        style_args["checkpoint"] = args.checkpoint
    else:
        raise Exception("No Path to saved checkpoints found")
    if args.image:
        style_args["image"] = args.image
    else:
        raise Exception("No image found to style")
    style_args["image_size"] = args.image_size
    style_args["output"] = args.output
    return style_args


def get_style_video_arguments(args):
    if args.config:
        try:
            with open(args.config, "r") as f:
                style_args = json.load(f)
            return style_args
        except Exception as e:
            print("Error Occurred while loading config: ", e)

    style_args = {}
    if args.checkpoint:
        style_args["checkpoint"] = args.checkpoint
    else:
        raise Exception("No Path to saved checkpoints found")
    if args.video:
        style_args["video"] = args.video
    else:
        raise Exception("No image found to style")
    style_args["format"] = args.format
    style_args["output"] = args.output
    if args.size:
        style_args["size"] = args.size
    return style_args


def get_style_webcam_arguments(args):
    if args.config:
        try:
            with open(args.config, "r") as f:
                style_args = json.load(f)
            return style_args
        except Exception as e:
            raise Exception("Error Occurred while loading config: ", e)

    style_args = {}
    if args.checkpoint:
        style_args["checkpoint"] = args.checkpoint
    else:
        raise Exception("No Path to saved checkpoints found")
    if args.video:
        style_args["camera"] = args.camera
    else:
        raise Exception("No image found to style")
    style_args["format"] = args.format
    style_args["output"] = args.output
    if args.size:
        style_args["size"] = args.size
    return style_args


def get_style_multi_images_arguments(args):
    if args.config:
        try:
            with open(args.config, "r") as f:
                style_args = json.load(f)
            return style_args
        except Exception as e:
            print("Error Occurred while loading config: ", e)

    style_args = {}
    if args.checkpoint:
        style_args["checkpoint"] = args.checkpoint
    else:
        raise Exception("No Path to saved checkpoints found")
    if args.path:
        style_args["path"] = args.path
    else:
        raise Exception("No image found to style")
    style_args["image_size"] = args.image_size
    style_args["output"] = args.output
    return style_args


def get_optimization_style_image_arguments(args):
    if args.config:
        try:
            with open(args.config, "r") as f:
                style_args = json.load(f)
            return style_args
        except Exception as e:
            print("Error Occurred while loading config: ", e)

    style_args = {}
    if args.image:
        style_args["image"] = args.image
    else:
        raise Exception("No image found to style")
    if args.style:
        style_args["style"] = args.style
    else:
        raise Exception("No style image found")
    style_args["image_size"] = args.image_size
    style_args["output"] = args.output
    style_args["epochs"] = args.epochs
    style_args["content_layers_weights"] = args.content_layers_weights
    style_args["style_layers_weights"] = args.style_layers_weights
    style_args["content_weight"] = args.content_weight
    style_args["style_weight"] = args.style_weight
    style_args["total_variation_weight"] = args.total_variation_weight
    style_args["content_layers"] = args.content_layers
    style_args["style_layers"] = args.style_layers
    style_args["learning_rate"] = args.learning_rate
    return style_args
