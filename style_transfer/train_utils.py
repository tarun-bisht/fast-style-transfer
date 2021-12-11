import tensorflow as tf
import numpy as np
import os
from .loss_model import LossModel
from pathlib import Path


class TensorflowDatasetLoader:
    def __init__(
        self, dataset_path, batch_size=4, image_size=(256, 256), num_images=None
    ):
        images_paths = [str(path) for path in Path(dataset_path).glob("*.jpg")]
        self.length = len(images_paths)
        if num_images is not None:
            images_paths = images_paths[0:num_images]
        dataset = tf.data.Dataset.from_tensor_slices(images_paths).map(
            lambda path: self.load_tf_image(path, dim=image_size),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.dataset = dataset

    def __len__(self):
        return self.length

    def load_tf_image(self, image_path, dim):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, dim)
        image = image / 255.0
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image

    def load_url_image(self, url, dim):
        img_request = requests.get(url)
        image = tf.image.decode_jpeg(BytesIO(img_request.content, channels=3))
        image = tf.image.resize(image, dim)
        image = image / 255.0
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image


def train_step(
    dataset,
    style_activations,
    steps_per_epoch,
    transform_net,
    loss_model,
    optimizer,
    checkpoint_path="./",
    content_weight=1e4,
    style_weight=1e-2,
    total_variation_weight=0.004,
    content_layers_weights=[1],
    style_layers_weights=[1] * 5,
):
    batch_losses = []
    steps = 1
    save_at = steps_per_epoch // 10
    save_path = os.path.join(checkpoint_path, "model_checkpoint.ckpt")
    for input_image_batch in dataset:
        if steps - 1 >= steps_per_epoch:
            break
        with tf.GradientTape() as tape:
            outputs = transform_net(input_image_batch)
            outputs = tf.clip_by_value(outputs, 0, 255)
            pred_activations = loss_model.get_activations(outputs / 255.0)
            content_activations = loss_model.get_activations(input_image_batch)[
                "content"
            ]
            curr_loss = perceptual_loss(
                pred_activations,
                content_activations,
                style_activations,
                content_weight,
                style_weight,
                content_layers_weights,
                style_layers_weights,
            )
            curr_loss += total_variation_weight * tf.image.total_variation(outputs)
        batch_losses.append(curr_loss)
        grad = tape.gradient(curr_loss, transform_net.trainable_variables)
        optimizer.apply_gradients(zip(grad, transform_net.trainable_variables))
        if steps % save_at == 0:
            print(f"Checkpoint created at: {save_path}", end=" ")
            transform_net.save_weights(save_path)
            print(f"Loss: {tf.reduce_mean(batch_losses).numpy()}")
        steps += 1
    return tf.reduce_mean(batch_losses)


def gatys_train_step(
    image_var,
    content_activations,
    style_activations,
    steps_per_epoch,
    loss_model,
    optimizer,
    content_weight=1e4,
    style_weight=1e-2,
    total_variation_weight=0.004,
    content_layers_weights=[1],
    style_layers_weights=[1] * 5,
):
    batch_losses = []
    for steps in range(steps_per_epoch):
        with tf.GradientTape() as tape:
            pred_activations = loss_model.get_activations(image_var)
            curr_loss = perceptual_loss(
                pred_activations,
                content_activations,
                style_activations,
                content_weight,
                style_weight,
                content_layers_weights,
                style_layers_weights,
            )
            curr_loss += total_variation_weight * tf.image.total_variation(image_var)
        batch_losses.append(curr_loss)
        grad = tape.gradient(curr_loss, image_var)
        optimizer.apply_gradients([(grad, image_var)])
        image_var.assign(tf.clip_by_value(image_var, 0, 1))
        print("=", end="")
    return tf.reduce_mean(batch_losses)


def get_loss_model(content_layers, style_layers):
    vgg = tf.keras.applications.vgg19.VGG19(weights="imagenet", include_top=False)
    loss_model = LossModel(vgg, content_layers, style_layers)
    return loss_model


def content_loss(placeholder, content, weight):
    assert (
        placeholder.shape == content.shape
    ), f"dimension not same {placeholder.shape} not = {content.shape}"
    return weight * tf.reduce_mean(tf.square(placeholder - content))


def gram_matrix(x):
    gram = tf.linalg.einsum("bijc,bijd->bcd", x, x)
    return gram / tf.cast(x.shape[1] * x.shape[2] * x.shape[3], tf.float32)


def style_loss(placeholder, style, weight):
    assert (
        placeholder.shape == style.shape
    ), f"dimension not same {placeholder.shape} not = {style.shape}"
    s = gram_matrix(style)
    p = gram_matrix(placeholder)
    return weight * tf.reduce_mean(tf.square(s - p))


def perceptual_loss(
    predicted_activations,
    content_activations,
    style_activations,
    content_weight,
    style_weight,
    content_layers_weights,
    style_layer_weights,
):
    pred_content = predicted_activations["content"]
    pred_style = predicted_activations["style"]
    c_loss = tf.add_n(
        [
            content_loss(
                pred_content[name], content_activations[name], content_layers_weights[i]
            )
            for i, name in enumerate(pred_content.keys())
        ]
    )
    c_loss = c_loss * content_weight
    s_loss = tf.add_n(
        [
            style_loss(
                pred_style[name], style_activations[name], style_layer_weights[i]
            )
            for i, name in enumerate(pred_style.keys())
        ]
    )
    s_loss = s_loss * style_weight
    return c_loss + s_loss
