import torch
import evaluate
import os
import shutil
import gc
import re
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

class WhisperUzbekManager:
    def __init__(self, model_id="openai/whisper-small", dataset_name="islomov/it_youtube_uzbek_speech_dataset"):
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Yuklanmoqda: {model_id}")
        self.processor = WhisperProcessor.from_pretrained(model_id, language="Uzbek", task="transcribe")
        self.tokenizer = WhisperTokenizer.from_pretrained(model_id, language="Uzbek", task="transcribe")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
        
    def prepare_dataset(self, num_samples=2000, seed=42):
        """tasodifiy sample-larni olish va qayta ishlash."""
        print(f"Datasetdan {num_samples} ta tasodifiy sample olinmoqda (Seed: {seed})...")
        raw_dataset = load_dataset(self.dataset_name, split="train", streaming=True)
        
        # Tasodifiylikni ta'minlash
        shuffled_dataset = raw_dataset.shuffle(seed=seed, buffer_size=1000)
        dataset_head = shuffled_dataset.take(num_samples)
        dataset_head = dataset_head.cast_column("audio", Audio(sampling_rate=16000))
        
        def prepare_function(batch):
            audio = batch["audio"]
            batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            
            target_text = batch.get("text") or batch.get("sentence") or batch.get("transcript")
            if target_text:
                target_text = re.sub(r'[^a-z0-9ʻ’\'‘ ]', ' ', target_text.lower().strip())
                target_text = re.sub(r'\s+', ' ', target_text).strip()
            
            batch["labels"] = self.tokenizer(target_text or " ").input_ids
            return batch

        def gen():
            for item in dataset_head:
                yield prepare_function(item)

        from datasets import Dataset
        full_ds = Dataset.from_generator(gen)
        split_ds = full_ds.train_test_split(test_size=0.1)
        
        # RAM-ni tejash uchun keraksiz obyektni o'chiramiz
        del full_ds
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return split_ds["train"], split_ds["test"]

    def load_model(self):
        print(f"Model yuklanmoqda: {self.model_id}")
        model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="uzbek", task="transcribe")
        model.config.suppress_tokens = []
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        return model

    def run_training(self, train_ds, test_ds, model, output_dir="./results"):
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        metric = evaluate.load("wer")

        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = self.tokenizer.pad_token_id
            pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            wer = 100 * metric.compute(predictions=pred_str, references=label_str)
            return {"wer": wer}

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            weight_decay=0.01,
            label_smoothing_factor=0.1,
            warmup_steps=100,
            max_steps=1000,
            gradient_checkpointing=True,
            fp16=True,
            eval_strategy="steps",
            per_device_eval_batch_size=8,
            dataloader_num_workers=0, # RAM tejash uchun
            optim="adamw_bnb_8bit",
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=250,
            eval_steps=250,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=self.feature_extractor,
        )

        print("Training boshlanmoqda...")
        trainer.train()
        return trainer

    def save_final_model(self, trainer, final_dir):
        print(f"Final model saqlanmoqda: {final_dir}")
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)
        
        # Checkpointlarni tozalash
        output_dir = trainer.args.output_dir
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Vaqtinchalik checkpointlar o'chirildi: {output_dir}")

    def quick_test(self, model_path, audio_input):
        """
        Inference testi. audio_input: fayl yo'li, numpy array yoki URL
        """
        from transformers import pipeline
        pipe = pipeline("automatic-speech-recognition", model=model_path, device=0 if torch.cuda.is_available() else -1)
        result = pipe(audio_input, generate_kwargs={"language": "uzbek", "task": "transcribe"})
        return result["text"]

if __name__ == "__main__":
    # Namuna (Colab-da buni cell-ma-cell ishlating)
    # manager = WhisperUzbekManager()
    # train_ds, test_ds = manager.prepare_dataset(num_samples=2000, seed=1)
    # model = manager.load_model()
    # trainer = manager.run_training(train_ds, test_ds, model)
    # manager.save_final_model(trainer, "whisper-uz-v1")
    pass
