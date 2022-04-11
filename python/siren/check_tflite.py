import numpy as np
import tensorflow as tf
import argparse
from pathlib import Path
import torchaudio
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Convert ONNX file to TFLite.')
parser.add_argument('-i','--input_tflite', required=True, type=Path)
args = parser.parse_args()

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=str(args.input_tflite))
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)
input_data = np.linspace(0, 1, num=input_shape[0], dtype=np.float32).reshape([input_shape[0], 1])

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

output_data = output_data.T
torchaudio.save('siren_tf.wav', torch.from_numpy(output_data), 44100)
plt.legend()
plt.savefig('siren_tf.png')