from config import emotion_config as config
from modules.preprocessing import ImageToArrayPreprocessor
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from modules.io import HDF5DatasetGenerator
from modules.nn import EmotionVGGNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import argparse
import datetime
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoint", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())

trainAug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    rescale=1 / 255.0,
    horizontal_flip=True,
    fill_mode="nearest")
valAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE,
                                aug=trainAug, preprocessors=[iap],
                                classes=config.NUM_CALSSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, 
                              aug=valAug, preprocessors=[iap],
                              classes=config.NUM_CALSSES)

if args["model"] is None:
    print("[INFO] compiling model...")
    model = EmotionVGGNet.build(width=48, height=48, depth=1,
                                classes=config.NUM_CALSSES)
    steps_per_epoch = trainGen.numImages // BATCH_SIZE
    clr = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate=config.INIT_LR,
        maximal_learning_rate=config.MAX_LR,
        scale_fn=lambda x: 1 / (2.0**(x - 1)),
        step_size=2 * steps_per_epoch)
    opt = Adam(lr=clr)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["acc"])
else:
    model = EmotionVGGNet.build(width=48, height=48, depth=1,
                                classes=config.NUM_CALSSES)
    print(f"[INFO] loading {args['model']}...")
    model = model.load_weights(args["model"])
    
    print("[INFO] old learning rate: {}".format(
        K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-3)
    print("[INFO] new learning rate: {}".format(
        K.get_value(model.optimizer.lr)))

# figPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.png"])
# jsonPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.json"])
log_dir = os.path.sep.join([config.OUTPUT_PATH, "logs/fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")])
callbacks = [
    ModelCheckpoint(args["checkpoint"], verbose=1),
    TensorBoard(log_dir=log_dir, update_freq=1)]

model.fit(
    trainGen.generator(),
    steps_per_epoch=steps_per_epoch,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // config.BATCH_SIZE,
    epochs=15,
    max_queue_size=config.BATCH_SIZE * 2,
    callbacks=callbacks, verbose=1)

trainGen.close()
valGen.close()
