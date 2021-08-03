from config import emotion_config as config
from modules.preprocessing import ImageToArrayPreprocessor
from modules.io import HDF5DatasetGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to model")
args = vars(ap.parse_args())

testAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
                               aug=testAug, preprocessors=[iap],
                               classes=config.NUM_CALSSES)

print(f"[INFO] loading {args['model']}...")
model = load_model(args['model'])

(loss , acc) = model.evaluate(
    testGen.generator(),
    steps=testGen.numImages // config.BATCH_SIZE,
    max_queue_size=config.BATCH_SIZE * 2)
print(f"[INFO] accuracy: {acc * 100:.2f}")

testGen.close()
