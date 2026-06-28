import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType
model_path = "./senior_shield_model"
print(f"Loading model from {model_path}...")
model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
print("Exporting to ONNX")
dummy_input = tokenizer("This is a sample scam message", return_tensors="pt")
torch.onnx.export(
    model, 
    (dummy_input['input_ids'], dummy_input['attention_mask']), 
    "senior_shield.onnx", 
    input_names=['input_ids', 'attention_mask'], 
    output_names=['logits'], 
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'}, 
        'attention_mask': {0: 'batch_size', 1: 'sequence'}
    }
)
# 3. Quantize
print("Quantizing model")
quantize_dynamic(
    "senior_shield.onnx",
    "senior_shield_int8.onnx",
    weight_type=QuantType.QInt8,
    reduce_range=True,
    op_types_to_quantize=['MatMul']
)
print("Created senior_shield_int8.onnx")