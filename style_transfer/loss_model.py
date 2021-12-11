import tensorflow as tf


class LossModel:
    def __init__(self, pretrained_model, content_layers, style_layers):
        self.model = pretrained_model
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.loss_model = self.get_model()

    def get_model(self):
        self.model.trainable = False
        layer_names = self.style_layers + self.content_layers
        outputs = [self.model.get_layer(name).output for name in layer_names]
        new_model = tf.keras.models.Model(inputs=self.model.input, outputs=outputs)
        return new_model

    def get_activations(self, inputs):
        inputs = inputs * 255.0
        style_length = len(self.style_layers)
        outputs = self.loss_model(tf.keras.applications.vgg19.preprocess_input(inputs))
        style_output, content_output = outputs[:style_length], outputs[style_length:]
        content_dict = {
            name: value for name, value in zip(self.content_layers, content_output)
        }
        style_dict = {
            name: value for name, value in zip(self.style_layers, style_output)
        }
        return {"content": content_dict, "style": style_dict}
