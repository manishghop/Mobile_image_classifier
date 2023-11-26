
import torch
import tensorflow as tf
import onnx
import onnxruntime
import onnx
import numpy as np
import onnxruntime as ort
from model import Net

best_epoch_weight_path = "./image_classification_epoch_8.pth"

model = Net()

model.load_state_dict(torch.load(best_epoch_weight_path))
model.eval()
sample_input = torch.rand((16,3,256,256))
onnx_model_path = 'mobile_fruit_classification.onnx'


torch.onnx.export(
    model,                  # PyTorch Model
    sample_input,                    # Input tensor
    onnx_model_path,        # Output file (eg. 'output_model.onnx')
    input_names = ['modelInput'],   # the model's input names
    output_names = ['modelOutput']
)


channels = 3
height,width = 256,256

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

# Check that the IR is well formed
onnx.checker.check_model(onnx_model)

# Print a Human readable representation of the graph
onnx.helper.printable_graph(onnx_model.graph)

ort_session = ort.InferenceSession(onnx_model_path)

outputs = ort_session.run(
    None,
    {'modelInput': np.random.randn(16,channels , height, width).astype(np.float32)}
)
print(outputs)