import tensorflow as tf

print("Loading H5 model...")
model = tf.keras.models.load_model("mask_model.h5")  # load your model

print("Exporting as SavedModel (Keras 3 format)...")
model.export("mask_model_converted")                 # <-- THIS WORKS 100%

print("Model converted successfully!")
