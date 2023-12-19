import os
import re
import random
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as pl

import sentencepiece as spm
from transformers import AutoModelForCausalLM, AutoConfig, GPT2Config

import wandb
wandb.init(project="Addition_training_bynumber")

# 乱数シード設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(100)

N = 1000
MAX_LEN = 6
def tokenizer(p, q, type):
    text_input_ids = np.full_like(np.zeros([MAX_LEN], dtype=np.int64), N)
    target_input_ids = np.full_like(np.zeros([MAX_LEN], dtype=np.int64), N)
    text_attention_mask = np.zeros([MAX_LEN], dtype=np.int64)

    text_input_ids[0], text_input_ids[1], text_input_ids[2], text_input_ids[3], text_input_ids[4], text_input_ids[5] = p, N+2, q, N+3, p+q, N+1
    target_input_ids[0], target_input_ids[1], target_input_ids[2], target_input_ids[3], target_input_ids[4], target_input_ids[5] = -100, -100, -100, -100, p+q, N+1
    #token_id{ pad:N, eos:N+1, +:N+2, =:N+3 }
    text_attention_mask[0], text_attention_mask[1], text_attention_mask[2], text_attention_mask[3], text_attention_mask[4], text_attention_mask[5] = 1, 1, 1, 1, 1, 1
    if type == "valid" or type == "test":
        text_input_ids = np.delete(text_input_ids, [-1, -2])
        target_input_ids = text_input_ids
        text_attention_mask = np.delete(text_attention_mask, [-1, -2])

    text_input_ids = torch.from_numpy(text_input_ids).clone()
    target_input_ids = torch.from_numpy(target_input_ids).clone()
    text_attention_mask = torch.from_numpy(text_attention_mask).clone()

    tokenized_inputs = {"input_ids":text_input_ids,"attention_mask":text_attention_mask}
    tokenized_targets = {"input_ids":target_input_ids,"attention_mask":text_attention_mask}

    return tokenized_inputs, tokenized_targets 

class CreateTokenID(Dataset):
    def __init__(self, type_path,):
        self.type = type_path
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"input_ids": source_ids, "attention_mask": source_mask,
                "labels": target_ids, "decoder_attention_mask": target_mask}
    
    def _build(self):
        text_list = []
        for i in range(1,N):
            for j in range(1,N):
                if i+j >= N:
                    pass
                elif self.type == "valid" and (i > N/2 or j > N/2):
                    text_list.append(f"{str(i)}+{str(j)}={str(i+j)}")
                    tokenized_inputs, tokenized_targets = tokenizer(i, j, self.type)
                    self.inputs.append(tokenized_inputs)
                    self.targets.append(tokenized_targets)
                elif (self.type == "train" or self.type == "test") and (i <= N/2 and j <= N/2):
                    text_list.append(f"{str(i)}+{str(j)}={str(i+j)}")
                    tokenized_inputs, tokenized_targets = tokenizer(i, j, self.type)
                    self.inputs.append(tokenized_inputs)
                    self.targets.append(tokenized_targets)
                else:
                    pass

class GPT2DataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    
    def get_dataset(self, type_path):
        """データセットを作成する"""
        return CreateTokenID(
            type_path=type_path,
            )

    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            train_dataset = self.get_dataset(type_path="train")
            self.train_dataset = train_dataset
            val_dataset = self.get_dataset(type_path="valid")
            self.val_dataset = val_dataset
        if stage == 'test':
            test_dataset = self.get_dataset(type_path="test")
            self.test_dataset = test_dataset

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          drop_last=True, shuffle=True, 
                          num_workers=4)

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size, 
                          num_workers=4)
    
    def test_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=4)
    
class GPT2Trainer(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        config = GPT2Config.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_config(config)
        self.lr = lr
        self.beam = 1
        self.training_step_loss = []
        self.validation_step_loss = []
        self.training_step_acc = []
        self.validation_step_acc = []

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def _accuracy(self, input_ids, attention_mask, labels):
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LEN*2, 
            num_beams=self.beam,
            pad_token_id=N,  # PADトークンのIDを指定
            eos_token_id=N+1,   # EOSトークンのIDを指定
        )
        max_token_id = N+3
        output_ids = [[token_id.item() if token_id.item() <= max_token_id else N for token_id in i] for i in output_ids]
        decorded_text = [["" if l==N else str(l) for l in ["" if k==N+1 else k for k in ["=" if j == N+3 else j for j in ["+" if i == N+2 else i for i in sentence]]]] for sentence in output_ids]

        add = 0
        for i_idx , i in enumerate(decorded_text):
            text = ''.join(decorded_text[i_idx])
            x = re.sub(r"\=|\+", "", re.search(r"^(.*?)\+", text).group())
            y = re.sub(r"\=|\+", "", re.search(r"\+(.*?)\=", text).group())
            ans = re.sub(r"\=|\+", "", re.search(r"\=(.*?)$", text).group())
            if ans == "": ans = 0
            if int(x)+int(y) == int(ans):
                add += 1
        return add/len(output_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, _ = self(input_ids, attention_mask, labels)

        self.log("train_loss", loss, prog_bar=True)
        wandb.log({"train_step_loss": loss})
        self.training_step_loss.append(loss.item())

        return loss
    
    def on_train_epoch_end(self):
        epoch_train_loss = sum(self.training_step_loss) / len(self.training_step_loss)
        epoch_val_loss = sum(self.validation_step_loss) / len(self.validation_step_loss)
        wandb.log({"train_loss": epoch_train_loss})
        wandb.log({"val_loss": epoch_val_loss})
        epoch_val_acc = sum(self.validation_step_acc) / len(self.validation_step_acc)
        wandb.log({"val_acc": epoch_val_acc})
        print('-------- Current Epoch {} --------'.format(self.current_epoch + 1))
        print('train Loss: {:.4f} val Loss: {:.4f}'.format(epoch_train_loss, epoch_val_loss))
        print('val Accuracy: {:.4f}'.format(epoch_val_acc))
        self.training_step_loss.clear()
        self.validation_step_loss.clear()
        self.training_step_acc.clear()
        self.validation_step_acc.clear()

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, _ = self(input_ids, attention_mask, labels)
        acc = self._accuracy(input_ids, attention_mask, labels)

        self.log("val_loss", loss, prog_bar=True)
        wandb.log({"val_step_loss": loss})
        self.validation_step_loss.append(loss.item())
        self.log("val_acc", acc, prog_bar=True)
        wandb.log({"val_step_acc": acc})
        self.validation_step_acc.append(acc)
        return loss
        
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, _ = self(input_ids, attention_mask, labels)
        acc = self._accuracy(input_ids, attention_mask, labels)

        self.log("train_loss", loss, prog_bar=True)
        # self.log("test_loss", loss, prog_bar=True)
        wandb.log({"val_step_loss": loss})
        self.validation_step_loss.append(loss.item())
        self.log("train_acc", acc, prog_bar=True)
        # self.log("test_acc", acc, prog_bar=True)
        wandb.log({"val_step_acc": acc})
        self.validation_step_acc.append(acc)
        return loss
 
    def generate(self, x, y):
        tokenized_inputs, _ = tokenizer(x, y, "train")
        target_text = [["" if k==N+1 or k==N else str(k) for k in ["=" if j == N+3 else j for j in ["+" if i == N+2 else i for i in sentence]]] for sentence in tokenized_inputs["input_ids"].unsqueeze(dim=0).tolist()]
        tokenized_inputs, _ = tokenizer(x, y, "valid")
        input_ids, attention_mask = tokenized_inputs["input_ids"].unsqueeze(dim=0), tokenized_inputs["attention_mask"].unsqueeze(dim=0)
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LEN*2, 
            num_beams=self.beam,
            pad_token_id=N,  # PADトークンのIDを指定
            eos_token_id=N+1,   # EOSトークンのIDを指定
        )
        max_token_id = N+3
        output_ids = [[token_id.item() if token_id.item() <= max_token_id else N for token_id in i] for i in output_ids]
        decorded_text = [["" if k==N+1 or k==N else str(k) for k in ["=" if j == N+3 else j for j in ["+" if i == N+2 else i for i in sentence]]] for sentence in output_ids]
        return decorded_text[0], target_text[0]

batch_size = 512
learning_rate = 1e-4
num_epochs = 100

data_module = GPT2DataModule(batch_size)
GPT2_Module = GPT2Trainer(learning_rate)
trainer = pl.Trainer(devices=1, accelerator="gpu",max_epochs=num_epochs, precision=16)

trainer.fit(GPT2_Module, data_module)
GPT2_Module.model.save_pretrained("models/GPT2_addition_model_bynumber_N1000")

GPT2_Module.model = AutoModelForCausalLM.from_pretrained("models/GPT2_addition_model_bynumber_N10")
predicted, answer = GPT2_Module.generate(527, 336)
print(f"predicted:{''.join(predicted)}, answer:{''.join(answer)}")
trainer.test(GPT2_Module, data_module)