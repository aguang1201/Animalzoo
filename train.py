import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from config import config
import tensorflow as tf
from keras import backend as K
from keras import regularizers

NB_EPOCHS = config['nb_epochs']
TRAIN_DIR = config['train_dir']
VALIDATION_DIR = config['validation_dir']
IM_WIDTH = config['im_width']
IM_HEIGHT = config['im_height']
BAT_SIZE = config['batch_size']
FC_SIZE = config['fc_size']
NB_IV3_LAYERS_TO_FREEZE = config['nb_iv3_layers_to_freeze']


def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt


def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet

  Args:
    base_model: keras model excluding top
    nb_classes: # of classes

  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, kernel_regularizer=regularizers.l2())(x) #new FC layer, random init
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.5)(x)
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model


def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

  Args:
    model: keras model
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def train(args):
  """Use transfer learning and fine-tuning to train a network on a new dataset"""
  nb_train_samples = get_nb_files(args.train_dir)
  nb_classes = len(glob.glob(args.train_dir + "/*"))
  nb_val_samples = get_nb_files(args.val_dir)
  nb_epoch = int(args.nb_epoch)
  batch_size = int(args.batch_size)

  # data prep
  train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )
  validation_datagen = ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )

  train_generator = train_datagen.flow_from_directory(
    args.train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  validation_generator = validation_datagen.flow_from_directory(
    args.val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  # setup model
  base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
  model = add_new_last_layer(base_model, nb_classes)

  # transfer learning
  print('transfer learning start--------------------------')
  setup_to_transfer_learn(model, base_model)

  history_tl = model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    samples_per_epoch=nb_train_samples,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    class_weight='auto')

  # fine-tuning
  print('fine-tuning start--------------------------')
  setup_to_finetune(model)

  history_ft = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    class_weight='auto')

  model.save(args.output_model_file)

  if args.plot:
    plot_training(history_ft)


def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()


if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--train_dir", default=TRAIN_DIR)
  a.add_argument("--val_dir", default=VALIDATION_DIR)
  a.add_argument("--nb_epoch", default=NB_EPOCHS)
  a.add_argument("--batch_size", default=BAT_SIZE)
  a.add_argument("--output_model_file", default="inceptionv3.model")
  a.add_argument("--plot", action="store_true")

  args = a.parse_args()
  if args.train_dir is None or args.val_dir is None:
    a.print_help()
    sys.exit(1)

  if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
    print("directories do not exist")
    sys.exit(1)

  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
  config.gpu_options.allow_growth = True
  K.set_session(tf.Session(config=config))

  train(args)