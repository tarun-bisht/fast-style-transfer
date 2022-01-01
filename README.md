# Fast Style Transfer &middot; [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarun-bisht/fast-style-transfer/blob/master/notebooks/Fast_Style_Transfer_Colab.ipynb) [![download pretrained models](https://img.shields.io/badge/download-pretrained%20models-important)](https://www.dropbox.com/sh/dkmy123bxk7f1s0/AAA-opMlprMhssPJCR1I1k4Qa?dl=0) ![python version](https://img.shields.io/badge/Python-%3E3.6-orange) ![tensorflow version](https://img.shields.io/badge/TensorFlow-%3E2.2-blueviolet) ![Windows](https://svgshare.com/i/ZhY.svg) ![Linux](https://svgshare.com/i/Zhy.svg) ![macOS](https://svgshare.com/i/ZjP.svg)


> Convert any photos and videos into an artwork

<div>
  <img src='data/images/style.jpg' height="346px">
  <img src='data/images/content.jpg' height="346px">
  <img src='output/styled.jpg' height="512px">
</div>

Stylize any photo or video in style of famous paintings using Neural Style Transfer.
- This is hundreds of times faster than the optimization-based method presented by [Gatys et al](https://arxiv.org/abs/1508.06576) so called fast style transfer.
- We train a feedforward network that apply artistic styles to images using loss function defined in [Gatys et al](https://arxiv.org/abs/1508.06576) paper.
- Feed forward network is a residual autoencoder network that takes content image as input and spits out stylized image.
- Model also uses instance normalization instead of batch normalization based on the paper [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
- Training is done by using perceptual loss defined in paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).
- Vgg19 is used to calculate perceptual loss more working described on paper.

## Requirements
### System
- For inferencing or generating images any system will work. But size of output image is limited as per system. Large images needs more momory to process. GPU is not must for inferencing but having it will be advantageous.
- For training GPU is must with tensorflow-gpu and cuda installed.
- If there is no access to GPU at local but want to train new style, there is a notebook `Fast_Style_Transfer_Colab.ipynb` open it in colab and train. For saving model checkpoints google drive is used. You can trust this notebook but I do not take any responsibility for data loss from google drive. Before running check the model save checkpoints path as it can override existing data with same name.
- Training takes around 6 hours in colab for 2 epochs.

### Python 3
Python 3.6 or higher. Tested with Python 3.7, 3.8, 3.9 in Windows 10 and Linux.
### Packages
- `tensorflow-gpu>=2.0` or `tensorflow>=2.0`
- `numpy`
- `matplotlib`
- `pillow`
- `opencv-python`

This implementation is tested with tensorflow-gpu 2.0, 2.2, 2.7 in Windows 10 and Linux

## Installation
### Install Python
There are two ways to install python in windows using [Python 3 installer](https://www.python.org/downloads/) or [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Installing python with anaconda or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is recommended. In linux Python 3 is installed by default but we can also install miniconda or conda into linux.

### Creating Virtual Environment
Create a new python virtual environment using conda or venv and activate it. If Anaconda or Miniconda is installed use `conda` else use `venv` to create virtual environments.

- Using conda
```bash
conda create --name artist
conda activate artist
conda install pip
```

- Using venv in linux
```bash
python3 -m venv path/to/create/env/artist
source path/to/create/env/artist/bin/activate
```

- Using venv in windows
```bash
python -m venv path\to\create\env\artist
path\to\create\env\artist\Scripts\activate
```

### Installing dependencies
The command below will install all the required dependencies from `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### Download Pretrained Style Models
- Download some [Pretrained Models](https://www.dropbox.com/sh/dkmy123bxk7f1s0/AAA-opMlprMhssPJCR1I1k4Qa?dl=0) trained on different paintings styles to start playing without need to train network
- copy and unzip checkpoints inside `data/models`

### Additional guides:
If get stuck in installation part follow these additional resources
- [Python Installation](https://www.youtube.com/watch?v=YYXdXT2l-Gg&list)
- [pip and usage](https://www.youtube.com/watch?v=U2ZN104hIcc)
- [Anaconda installation and using conda](https://www.youtube.com/watch?v=YJC6ldI3hWk)

## Documentation
- Input Parameters for a script can be set by using config file or using command line arguments.
- All configs file are located inside `configs` folder.
- If config file path is passed as command line argument then all parameters will be read by script from it, otherwise input parameters are needed to be passed as command line arguments. Using config files will reduce lines to be typed in command line also easy to track all the parameters that can be tweaked.

### Image Stylization
#### Single Image Stylization

Stylize only one image at a time
```bash
python style_image.py --config=configs/image_config.json
```
or 
```bash
python style_image.py --checkpoint data/models/udnie/model_checkpoint.ckpt --image data/images/content.jpg --image_size 1366 768 --output output/styled.jpg
```
#### Multiple Images Stylization
Stylize all images inside a folder.
```bash
python style_multi_images.py --config=configs/multi_images_config.json
```
or 
```bash
python style_multi_images.py --checkpoint data/models/udnie/model_checkpoint.ckpt --path data/images/content.jpg --image_size 1366 768 --output output/styled.jpg
```

### Video Stylization
<div>
  <a href="http://www.youtube.com/watch?v=GrS4rWifdko"><img src='output/video.gif' alt="Pithoragarh style transfer"></a>
</div>

Use `style_video.py` to transfer style into a video.To view all the possible input parameters view its `configs/video_config.json` these parameters can be passed as command line arguments as well.

```bash
python style_video.py --config=configs/video_config.json
```

### Live Webcam Feed Stylization
<div>
  <img src='output/webcam.gif' alt="webcam output">
</div>

Use `style_webcam.py` to transfer style into live webcam recording.To view all the possible input parameters view its `configs/webcam_config.json` these parameters can be passed as command line arguments as well.

```bash
python style_webcam.py --config=configs/webcam_config.json
```

### Training a new style
- Download [coco 2014](http://images.cocodataset.org/zips/train2014.zip) dataset and extract it inside data/train directory. It can be extracted to anywhere in computer as long you defined its path inside `configs/train.json`. or in command line arguments. You can also use any other dataset with images, I have tried with some small datasets but results were not good. It needs more experimentation.
```bash
!wget http://images.cocodataset.org/zips/train2014.zip data/train
!unzip -qq data/train/train2014.zip
```
- Get your style image ready. It can be a file link over http or path to file in local
- Edit `configs/train.json` with parameters as your linking especially `checkpoint`, `style_image` and `train_path`. You can also pass these as command line arguments when running `train.py` script. It depends on your preferences.
- Run `train.py` script
```bash
python train.py --config=configs/train_config.json
```
- If want to train style images in google colab a notebook `Fast_Style_Transfer_Colab.ipynb` is provided. Open it in colab and train. For saving model checkpoints google drive is used. You can trust this notebook but I do not take any responsibility for data loss from google drive. Before running check the model save checkpoints path as it can override existing data with same name.

### Gatys Style Transfer (optimization based stylization)
We can also perform gatys style transfer by using `slow_style_transfer.py` script. You can use it for styling images but it takes more time to generate an image. To view all the possible input parameters view `configs/slow_style_config.json` these parameters can be passed as command line arguments as well.
The benefit of this method is that we can create different stylized images using different style images without training a network for that particular style, but it takes some time to create a single stylized image. This method is prefered to test different styles and check which one is working great before training a network for that style.

```bash
python slow_style_transfer.py --config=configs/slow_style_config.json
```

<div>
  <a href="http://www.youtube.com/watch?v=weVfBfWVuZw"><img src='http://img.youtube.com/vi/weVfBfWVuZw/0.jpg' alt="Gatys style transfer"></a>
</div>

## Stylized Results
<div>
  <img src='output/js_candy.jpg' height="346px">
</div>
<div>
  <img src='output/kido.jpg' height="346px">
</div>
<div>
  <img src='output/penguin.jpg' height="346px">
</div>
<div>
  <img src='output/styled_candy.jpg' height="346px">
  <img src='output/heather-gill-7Frxnyv7Ntg-unsplash-udine.jpg' height="346px">
  <img src='output/marc-olivier-jodoin-0TB3AiOu2g4-unsplash-starry.jpg' height="346px">
</div>

## Contribution
Contributions are highly welcome that will improve quality of project. Folks with high end machines are welcome to train new styles and contribute. For style models contribution [Contact me](https://tarunbisht.com/contact)

## License
Copyright (c) 2020 Tarun Bisht. Free for personal or research use. [Contact me](https://tarunbisht.com/contact) for commercial use.

## Attributions
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
- Image from [Unsplash](https://unsplash.com/s/photos/deep-meaning?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText) by [Heather Gill](https://unsplash.com/@heathergill?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText)

## Support
Support this project through [patreon](https://www.patreon.com/tarunbisht)
