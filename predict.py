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
import glob
import os

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


def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  orig = cv2.imread(image)
  preds_index = np.argmax(preds)
  preds_name = classes[preds_index]
  prob = np.max(preds)

  cv2.putText(orig, 'AI guess it`s {}'.format(preds_name), (10, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255),2)
  cv2.putText(orig, 'probality is {:.2f}%'.format(prob * 100), (10, 70),
              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
  cv2.imshow('Classifiction', orig)
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
  a.add_argument("--image", default="/data1/cat_dog_fire/test1/5.jpg", help="path to image")
  a.add_argument("--image_url", default="", help="url to image")
  a.add_argument("--model", default="inceptionv3_dog_cat20170518.model")
  args = a.parse_args()

  if args.image is None and args.image_url is None:
    a.print_help()
    sys.exit(1)

  model = load_model(args.model)
  if args.image is not None:
    img = Image.open(args.image)
    preds = predict(model, img, target_size)
    plot_preds(args.image, preds)

  if args.image_url is not None:
    response = requests.get(args.image_url)
    img = Image.open(BytesIO(response.content))
    preds = predict(model, img, target_size)
    plot_preds(img, preds)