# %%
# https://github.com/sithu31296/PyTorch-ONNX-TFLite
# mamba install -c conda-forge tensorflow-probability tensorflow
import onnx
from pathlib import Path
from onnx_tf.backend import prepare
import argparse

parser = argparse.ArgumentParser(description='Convert ONNX file to TFLite.')
parser.add_argument('-i','--input_onnx', required=True, type=Path)
parser.add_argument('-o','--output_tflite', required=True, type=Path)
args = parser.parse_args()

# %%
onnx_name = Path(args.input_onnx)

# Load the ONNX model
model = onnx.load(onnx_name)

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a Human readable representation of the graph
# onnx.helper.printable_graph(model.graph)

# %%
tf_model_path = f"/tmp/{onnx_name.with_suffix('')}_export"
tf_rep = prepare(model)
tf_rep.export_graph(tf_model_path)

# %%
import tensorflow as tf

model = tf.saved_model.load(tf_model_path)
model.trainable = False


# %%
tflite_model_path = f"{args.output_tflite}"
print(tflite_model_path)
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# %%



