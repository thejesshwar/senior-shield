import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split 
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from config import MAX_LENGTH, BATCH_SIZE
import random
import nlpaug.augmenter.char as nac

def load_data():
    print("Loading Datasets")
    datasets = []
    try:
        df_real = pd.read_csv('data/public_unified_multimodal.csv')
        df_real = df_real[['text', 'is_scam']].rename(columns={'is_scam': 'label'})
        datasets.append(df_real)
        print(f"Loaded {len(df_real)} original rows.")
    except:
        print("Original dataset not found.")
    try:
        df_safe = pd.read_csv('data/synthetic_safe_data.csv')
        datasets.append(df_safe)
        print(f"Loaded {len(df_safe)} synthetic SAFE rows.")
    except:
        print("Synthetic SAFE dataset not found.")
    try:
        df_scam = pd.read_csv('data/synthetic_scam_data.csv')
        datasets.append(df_scam)
        print(f"Loaded {len(df_scam)} synthetic SCAM rows.")
    except:
        print("Synthetic SCAM dataset not found.")
    try:
        df_refund = pd.read_csv('data/synthetic_refund_scams.csv')
        datasets.append(df_refund)
        print(f"Loaded {len(df_refund)} specific REFUND SCAM rows.")
    except:
        print("Synthetic REFUND SCAM dataset not found.")
    try:
        df_missing = pd.read_csv('data/synthetic_missing_scams.csv')
        datasets.append(df_missing)
        print("   -> Loaded Electricity & Police Scams.")
    except:
        print("Synthetic MISSING SCAM dataset not found.")
    try:
        df_hindi = pd.read_csv('data/synthetic_hindi_final.csv')
        datasets.append(df_hindi)
        print(f"Loaded {len(df_hindi)} HINDI SCAM rows.")
    except:
        print("Synthetic HINDI SCAM dataset not found.")
    try:
        df_hinglish = pd.read_csv('data/synthetic_hinglish_scams.csv')
        datasets.append(df_hinglish)
        print(f"Loaded {len(df_hinglish)} HINGLISH SCAM rows.")
    except:
        print("Synthetic HINGLISH SCAM dataset not found.")
    try:
        df_hindi_missing=pd.read_csv('data/synthetic_hindi_finals.csv')
        datasets.append(df_hindi_missing)
        print("Loaded additional HINDI SCAMS.")
    except:
        print("No additional HINDI SCAMS found.")
    try:
        df_hinfi_safe=pd.read_csv('data/synthetic_hindi_safe.csv')
        datasets.append(df_hinfi_safe)
        print("Loaded HINDI SAFE rows.")
    except:
        print("No HINDI SAFE rows found.")
    try:
        df_hinglish_full=pd.read_csv('data/synthetic_hinglish_scam.csv')
        datasets.append(df_hinglish_full)
        print("Loaded additional HINGLISH SCAMS.")
    except:
        print("No additional HINGLISH SCAMS found.")
        
    if not datasets:
        print("No data found!")
        exit()
        
    dataset = pd.concat(datasets, ignore_index=True)
    dataset = dataset.drop_duplicates(subset=['text'])
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)
    scam_count = len(dataset[dataset['label'] == 1])
    safe_count = len(dataset[dataset['label'] == 0])
    print(f"Scams: {scam_count} | Safe: {safe_count}")
    print(f"(Balance: {safe_count/(scam_count+safe_count)*100:.2f}% Safe)")
    dataset['label'] = dataset['label'].astype(int)
    dataset = dataset.dropna(subset=['text'])
    return dataset

class ScamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def main():
    df = load_data()
    print("Splitting Data")
    cv_texts, test_texts, cv_labels, test_labels = train_test_split(
        df['text'].tolist(), 
        df['label'].tolist(), 
        test_size=0.10, 
        random_state=42,
        stratify=df['label'].tolist() 
    )
    
    test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})
    test_df.to_csv("blind_test_set.csv", index=False)
    print(f"Vault locked: Saved {len(test_df)} rows to 'blind_test_set.csv' for the Final Exam")
    
    print("Applying Typo/Obfuscation Augmentation")
    aug = nac.KeyboardAug(aug_char_p=0.1, aug_word_max=2)
    
    augmented_texts = []
    augmented_labels = []
    
    for text, label in zip(cv_texts, cv_labels):
        augmented_texts.append(text)
        augmented_labels.append(label)
        if label == 1 and random.random() < 0.30:
            try:
                aug_text = aug.augment(text)
                if isinstance(aug_text, list):
                    aug_text = aug_text[0]
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
            except:
                pass 
                
    cv_texts = augmented_texts
    cv_labels = augmented_labels
    print(f"Added synthetic obfuscations. New CV dataset size: {len(cv_texts)}")

    print("\nTokenizing data for Cross-Validation")
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(cv_texts, cv_labels)):
        print(f"\nStarting Fold {fold + 1}/5")
        
        train_texts = [cv_texts[i] for i in train_idx]
        train_labels = [cv_labels[i] for i in train_idx]
        val_texts = [cv_texts[i] for i in val_idx]
        val_labels = [cv_labels[i] for i in val_idx]
        
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
        
        train_dataset = ScamDataset(train_encodings, train_labels)
        val_dataset = ScamDataset(val_encodings, val_labels)
        
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=2)
        training_args = TrainingArguments(
            output_dir=f'./results_fold_{fold+1}', 
            num_train_epochs=3, 
            per_device_train_batch_size=8, 
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_steps=50, 
            weight_decay=0.01, 
            logging_dir=f'./logs_fold_{fold+1}', 
            logging_steps=50,
            eval_strategy="epoch",             
            save_strategy="epoch",            
            load_best_model_at_end=True,       
            metric_for_best_model="recall",   
            greater_is_better=True,            
            fp16=torch.cuda.is_available(),
        )
        
        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=train_dataset, 
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        
        print(f"\nEvaluating Fold {fold + 1}")
        metrics = trainer.evaluate()
        fold_metrics.append(metrics)
        print(f"Fold {fold + 1} Metrics: {metrics}")
        del model, trainer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    print("\nCross-Validation Complete")
    print("Training Final Production Model on 100% of CV Data")
    final_encodings = tokenizer(cv_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    final_dataset = ScamDataset(final_encodings, cv_labels)
    final_model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=2)
    final_training_args = TrainingArguments(
        output_dir='./results_final', 
        num_train_epochs=3, 
        per_device_train_batch_size=8, 
        warmup_steps=50, 
        weight_decay=0.01, 
        logging_dir='./logs_final', 
        logging_steps=50,
        save_strategy="no",            
        fp16=torch.cuda.is_available(),
    )
    
    final_trainer = Trainer(
        model=final_model, 
        args=final_training_args,
        train_dataset=final_dataset, 
    )
    final_trainer.train()
    
    print("Saving robust final model to './senior_shield_model'")
    final_model.save_pretrained("./senior_shield_model")
    tokenizer.save_pretrained("./senior_shield_model")
    
    avg_acc = np.mean([m['eval_accuracy'] for m in fold_metrics])
    avg_recall = np.mean([m['eval_recall'] for m in fold_metrics])
    print(f"Average Accuracy across 5 folds: {avg_acc*100:.2f}%")
    print(f"Average Recall across 5 folds:   {avg_recall*100:.2f}%")

if __name__ == "__main__":
    main()