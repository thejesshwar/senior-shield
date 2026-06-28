import pandas as pd
import numpy as np
import onnxruntime as ort
from transformers import XLMRobertaTokenizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time
import sys
from config import SCAM_THRESHOLD, MAX_LENGTH, BATCH_SIZE, MODEL_PATH, ONNX_MODEL

def load_resources():
    print(f"Loading Quantized Model: {ONNX_MODEL}")
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(ONNX_MODEL, sess_options=sess_options, providers=providers)
    return tokenizer, session

def predict_batch(tokenizer, session, texts):
    # START TIMER HERE: Captures both tokenization and inference (End-to-End Latency)
    start = time.time()
    
    # 1. Tokenize
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="np")
    onnx_inputs = {
        session.get_inputs()[0].name: inputs['input_ids'].astype(np.int64),
        session.get_inputs()[1].name: inputs['attention_mask'].astype(np.int64)
    }
    
    # 2. Run Inference
    logits = session.run(None, onnx_inputs)[0]
    end = time.time()
    
    # 3. Softmax
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    # 4. Apply Centralized Threshold
    final_preds = []
    for i in range(len(texts)):
        scam_probability = probs[i][1]
        if scam_probability > SCAM_THRESHOLD:
            final_preds.append(1)
        else:
            final_preds.append(0)

    latency_ms = ((end - start) / len(texts)) * 1000
    return np.array(final_preds), latency_ms

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate.py <path_to_csv>")
        return
    csv_path = sys.argv[1]
    print(f"Loading Data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    cols = [c.lower() for c in df.columns]
    if 'text' not in cols:
        print("CSV must have a text column.")
        return
        
    text_col = df.columns[cols.index('text')]
    has_labels = 'label' in cols or 'is_scam' in cols
    label_col = None
    if has_labels:
        label_col = df.columns[cols.index('label')] if 'label' in cols else df.columns[cols.index('is_scam')]
        
    tokenizer, session = load_resources()
    texts = df[text_col].astype(str).tolist()
    print(f"Running inference on {len(texts)} rows") 
    
    all_preds = []
    total_latency = 0
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i+BATCH_SIZE]
        preds, lat = predict_batch(tokenizer, session, batch)
        all_preds.extend(preds)
        total_latency += lat
        print(f"Processed {min(i+BATCH_SIZE, len(texts))}/{len(texts)}", end='\r')
        
    avg_latency = total_latency / (len(texts) / BATCH_SIZE)
    print(f"\nAvg End-to-End Latency per Batch: {avg_latency:.2f} ms")
    
    df['predicted_label'] = all_preds
    df['prediction_text'] = ['SCAM' if p==1 else 'SAFE' for p in all_preds]
    df.to_csv("validation_results.csv", index=False)
    print("Predictions saved to 'validation_results.csv'")
    
    if has_labels:
        y_true = df[label_col].astype(int).tolist()
        acc = accuracy_score(y_true, all_preds)
        print(f"\nMODEL PERFORMANCE REPORT (Threshold: {SCAM_THRESHOLD})")
        print(f"Accuracy:  {acc*100:.2f}%")
        print(classification_report(y_true, all_preds, target_names=['Safe', 'Scam']))
        cm = confusion_matrix(y_true, all_preds)
        print(f"Confusion Matrix:\n{cm}")
        print(f"(Safe identified as Scam: {cm[0][1]}) False Positives")
        print(f"(Scams missed as Safe: {cm[1][0]}) False Negatives")

if __name__ == "__main__":
    main()