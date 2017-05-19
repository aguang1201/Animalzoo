import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from config import config
import cv2
import os
from skimage import io

TRAIN_DIR = config['train_dir']
target_size = (config['im_width'], config['im_height']) #fixed size for InceptionV3 architecture
classes = os.listdir(TRAIN_DIR)

def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]


def plot_preds(img_array, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    img: PIL image
    preds: list of predicted labels and their probabilities
  """
  preds_index = np.argmax(preds)
  preds_name = classes[preds_index]
  prob = np.max(preds)


  cv2.putText(img_array, 'AI guess it`s a {}'.format(preds_name), (10, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
  cv2.putText(img_array, 'probality is {:.2f}%'.format(prob * 100), (10, 70),
              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
  cv2.imshow('Classifiction', img_array)
  cv2.waitKey(0)

  plt.figure()
  labels = ("cat", "dog")
  plt.barh([0, 1], preds, alpha=0.5)
  plt.yticks([0, 1], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()

if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--image", default="/data1/cat_dog_fire/test1/10.jpg", help="path to image")
  a.add_argument("--image_url", help="url to image")
  #a.add_argument("--image", help="path to image")
  #a.add_argument("--image_url", default="http://farm1.static.flickr.com/86/209203070_ac7c4ce5a2.jpg", help="url to image")
  a.add_argument("--model", default="inceptionv3_dog_cat20170518.model")
  args = a.parse_args()

  if args.image is None and args.image_url is None:
    a.print_help()
    sys.exit(1)

  model = load_model(args.model)
  if args.image is not None and args.image is not "":
    image_path = args.image
    img = Image.open(image_path)
    preds = predict(model, img, target_size)
    img_array = cv2.imread(image_path)
    plot_preds(img_array, preds)

  if args.image_url is not None and args.image_url is not "":
    response = requests.get(args.image_url)
    img = Image.open(BytesIO(response.content))
    preds = predict(model, img, target_size)
    img_array = io.imread(args.image_url)
    plot_preds(img_array, preds)