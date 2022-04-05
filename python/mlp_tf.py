# %%
# https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb
from random import shuffle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.version.VERSION)
# from model_utils import save_model

# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_shape=(1,), activation='relu', kernel_initializer='random_normal'),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='random_normal'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.mse,
              metrics=['accuracy'])

# %%
batch_size, input_lenght, input_dim = 1, 8192, 1
x =  np.random.uniform(-1, 1, input_lenght).astype(np.float32)
# Shuffle the values to guarantee they're not in order
np.random.shuffle(x)

# Calculate the corresponding sine values
y = np.sin(8 * np.pi * x).astype(np.float32)
plt.scatter(x, y)

# %%

# We'll use 60% of our data for training and 20% for testing. The remaining 20%
# will be used for validation. Calculate the indices of each section.
TRAIN_SPLIT =  int(0.6 * input_lenght)
TEST_SPLIT = int(0.2 * input_lenght + TRAIN_SPLIT)

# Use np.split to chop our data into three parts.
# The second argument to np.split is an array of indices where the data will be
# split. We provide two indices, so the data will be divided into three chunks.
x_train, x_test, x_validate = np.split(x, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_test, y_validate = np.split(y, [TRAIN_SPLIT, TEST_SPLIT])

# Double check that our splits add up correctly
assert (x_train.size + x_validate.size + x_test.size) == input_lenght

# Plot the data in each partition in different colors:
plt.plot(x_train, y_train, 'b.', label="Train")
plt.plot(x_test, y_test, 'r.', label="Test")
plt.plot(x_validate, y_validate, 'y.', label="Validate")
plt.legend()

# %% Train
history = model.fit(x_train, y_train, batch_size=128, epochs=500, verbose=0,
                    validation_data=(x_validate, y_validate), shuffle=True)

test_loss, test_mae = model.evaluate(x_test, y_test)
model.summary()

# %%
# Make predictions based on our test dataset
y_test_pred = model.predict(x_test)

plt.scatter(x_test, y_test_pred, label="Test")
plt.scatter(x_test, y_test, label="GT")
plt.legend()
plt.savefig("mlp1024_tf.png")

# %%

train_loss = history.history['loss']
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'g.', label='Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# %%
# https://www.tensorflow.org/lite/convert

run_model = tf.function(lambda x: model(x))
# This is important, let's fix the input size.
BATCH_SIZE = 1
STEPS = 1024
INPUT_SIZE = 1
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([STEPS, INPUT_SIZE], tf.float32))

# model directory.
MODEL_DIR = "keras_mlp"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

# def representative_dataset():
#   for i in range(500):
#     yield([x_train[i].reshape(1, 1)])
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Enforce integer only quantization
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8
# converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

# Save the model.
with open('mlp1024.tflite', 'wb') as f:
  f.write(tflite_model)
# %%

# Save model for RTNeural
# save_model(model, 'mlp_rtneural.json')
# %%
