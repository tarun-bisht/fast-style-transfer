import time
import cv2
import tensorflow as tf
import argparse
from style_transfer.inference import inference
from style_transfer.transform_net import ImageTransformNet
from utils.utility import load_image,load_url_image,array_to_img
from utils.parse_arguments import get_style_webcam_arguments

parser=argparse.ArgumentParser(description='Live Stylized Videos')
parser.add_argument("--config","-con",
                    help="Path to style image config file")
parser.add_argument("--checkpoint","-ckpt",
                    help="Trained Checkpoints Path")
parser.add_argument("--output","-out",
                    help="Output Path for saving recorded video")
parser.add_argument('--format',"-format",default='XVID', 
                    help='codec used in VideoWriter when saving video to file')
parser.add_argument('--size',"-s",nargs='+',type=int,help='output video size')
parser.add_argument('--camera',"-cam",default=0,type=int,
                    help='webcam number to record')
args=get_style_webcam_arguments(parser.parse_args())

try:
    tf.config.optimizer.set_jit(True)
except:
    pass

style_model=ImageTransformNet()
style_model.load_weights(args["checkpoint"])

vid = cv2.VideoCapture(int(args["camera"]))
out=None

if args.get("output"):
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.get("size"):
        width = int(args["size"][0])
        height = int(args["size"][1])
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*args["format"])
    out = cv2.VideoWriter(args["output"], codec, fps, (width, height))

while True:
    _, img = vid.read()

    if img is None:
        print("Empty Frame")
        time.sleep(0.1)
        break
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if args.get("size"):
        img_in=cv2.resize(img_in,tuple(args["size"]))
    styled_image=inference(style_model,img_in)
    if out:
        out.write(styled_image)
    cv2.imshow('output', styled_image)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()