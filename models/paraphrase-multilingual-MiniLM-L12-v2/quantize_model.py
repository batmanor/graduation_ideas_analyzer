from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

BASE_DIR = Path(__file__).parent

fp32_model = BASE_DIR / "model.onnx"
int8_model = BASE_DIR / "model.int8.onnx"

print("Input:", fp32_model)
print("Output:", int8_model)

quantize_dynamic(
    model_input=str(fp32_model),
    model_output=str(int8_model),
    weight_type=QuantType.QInt8,
)

print("Done.")