import torch
from datasets import load_dataset, Audio, Dataset
import re
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from train_whisper import WhisperUzbekManager, DataCollatorSpeechSeq2SeqWithPadding

manager = WhisperUzbekManager()
train_ds, test_ds = manager.prepare_dataset(num_samples=16, seed=42)
print("Dataset columns:", train_ds.column_names)

collator = DataCollatorSpeechSeq2SeqWithPadding(processor=manager.processor)
batch = collator([train_ds[i] for i in range(2)])
print("Collator output keys:", batch.keys())
print("Labels shape:", batch["labels"].shape)

model = manager.load_model()
try:
    outputs = model(**batch)
    print("Forward pass successful!")
except Exception as e:
    import traceback
    traceback.print_exc()
