from transformers import BertTokenizer, WEIGHTS_NAME,TrainingArguments, BertForMaskedLM,BertConfig
from transformers import (AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)


pretrained_model_name = 'bert-base-uncased'
vocab_path = 'vocab.txt'
# with open(vocab_path, 'r', encoding="utf-8") as f:
#     for line in f.readlines():
#         cls, sentence = line.strip().split(",", 1)
tokenizer = BertTokenizer.from_pretrained(vocab_path)
config = BertConfig.from_pretrained(pretrained_model_name)
model = BertForMaskedLM.from_pretrained(pretrained_model_name)
model.resize_token_embeddings(len(tokenizer))

train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='corpus.txt', block_size=512)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

pretrain_batch_size = 4
num_train_epochs = 10

training_args = TrainingArguments(
    output_dir="fur_pre",  # select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy='steps',
    save_total_limit=2,
    eval_steps=200,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    prediction_loss_only=True,
    report_to="none")

trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)

trainer.train()
trainer.save_model('fur_pre/logs')
