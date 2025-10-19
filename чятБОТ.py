# -*- coding: utf-8 -*-
import os
import json
import contextlib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW

# ---------- Config ----------
DATA_PATH = "dataset.json"
VOCAB_PATH = "vocab.pt"
MODEL_DIR = "my_gpt2_model"


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Warning: {name}={raw!r} is not an int. Using default {default}.")
        return default


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Warning: {name}={raw!r} is not a float. Using default {default}.")
        return default


MAX_LEN = _env_int("MAX_LEN", 128)
BATCH_SIZE = _env_int("BATCH_SIZE", 32)
EPOCHS = _env_int("EPOCHS", 8)
LR = _env_float("LR", 2e-4)
WEIGHT_DECAY = _env_float("WEIGHT_DECAY", 0.01)
LOG_EVERY = _env_int("LOG_EVERY", 20)

# ---------- Device ----------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    use_amp = True
else:
    device = torch.device("cpu")
    print("Using CPU. CUDA is not available.")
    use_amp = False

# ---------- Tokenizer ----------
class WordTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        # special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.sep_token = "<SEP>"
        self.eos_token = "<EOS>"
        for t in (self.pad_token, self.unk_token, self.bos_token, self.sep_token, self.eos_token):
            self.add_token(t)

    def add_token(self, token):
        if token not in self.vocab:
            self.vocab[token] = self.vocab_size
            self.inverse_vocab[self.vocab_size] = token
            self.vocab_size += 1

    def build_vocab(self, texts):
        for text in texts:
            for w in text.split():
                self.add_token(w)

    def ensure_specials(self):
        changed = False
        for t in (self.pad_token, self.unk_token, self.bos_token, self.sep_token, self.eos_token):
            if t not in self.vocab:
                self.add_token(t)
                changed = True
        return changed

    def encode_tokens(self, tokens, max_length):
        ids = [self.vocab.get(t, self.vocab[self.unk_token]) for t in tokens]
        ids = ids[:max_length]
        attn = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.vocab[self.pad_token])
            attn.append(0)
        return ids, attn

    def encode_prompt_and_target(self, prompt_text, target_text, max_length=128):
        prompt = prompt_text.split()
        target = target_text.split()
        tokens = [self.bos_token] + prompt + [self.sep_token] + target + [self.eos_token]
        ids = [self.vocab.get(t, self.vocab[self.unk_token]) for t in tokens]
        # mask everything up to and including SEP
        try:
            sep_index = tokens.index(self.sep_token)
        except ValueError:
            sep_index = len(tokens) - 1
        labels = [-100] * (sep_index + 1) + ids[(sep_index + 1):]
        if len(ids) > max_length:
            ids = ids[:max_length]
            labels = labels[:max_length]
        attn = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.vocab[self.pad_token])
            labels.append(-100)
            attn.append(0)
        return ids, attn, labels

    def detok(self, token_ids):
        toks = [self.inverse_vocab.get(i, self.unk_token) for i in token_ids]
        out = []
        for t in toks:
            if t in (self.pad_token, self.bos_token):
                continue
            if t == self.eos_token:
                break
            out.append(t)
        return " ".join(out).strip()

# ---------- Dataset ----------
class ChatDataset(Dataset):
    def __init__(self, pairs, tokenizer: WordTokenizer, max_length=128):
        self.data = pairs
        self.tk = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        ids, attn, labels = self.tk.encode_prompt_and_target(ex["input"], ex["output"], self.max_length)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# ---------- Load data ----------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} не знайдено.")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, list) or len(data) == 0:
    raise RuntimeError("dataset.json має бути списком прикладів {input, output}.")

# ---------- Tokenizer init / load vocab ----------
tokenizer = WordTokenizer()
if os.path.exists(VOCAB_PATH):
    print("Loading saved vocab...")
    loaded = torch.load(VOCAB_PATH, map_location="cpu")
    tokenizer.vocab = dict(loaded)
    tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
    tokenizer.vocab_size = len(tokenizer.vocab)
    if tokenizer.ensure_specials():
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.vocab_size = len(tokenizer.vocab)
        torch.save(tokenizer.vocab, VOCAB_PATH)
else:
    print("Building new vocab...")
    all_texts = [item.get("input", "") + " " + item.get("output", "") for item in data]
    tokenizer.build_vocab(all_texts)
    tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
    tokenizer.vocab_size = len(tokenizer.vocab)
    torch.save(tokenizer.vocab, VOCAB_PATH)

print(f"Vocab size: {tokenizer.vocab_size}")

# ---------- DataLoader ----------
train_ds = ChatDataset(data, tokenizer, max_length=MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# ---------- Model ----------
cfg = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=MAX_LEN,
    n_ctx=MAX_LEN,
    n_embd=256,
    n_layer=6,
    n_head=8,
    pad_token_id=tokenizer.vocab[tokenizer.pad_token],
    bos_token_id=tokenizer.vocab[tokenizer.bos_token],
    eos_token_id=tokenizer.vocab[tokenizer.eos_token],
)

if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    print("Loading existing model for finetune...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
else:
    print("Creating new model...")
    model = GPT2LMHeadModel(cfg)

# ensure embeddings cover tokenizer ids
max_token_id = max(tokenizer.vocab.values())
if max_token_id + 1 > model.config.vocab_size:
    model.resize_token_embeddings(max_token_id + 1)

# ensure positional embeddings cover MAX_LEN
if getattr(model.config, "n_positions", 0) < MAX_LEN:
    old_n = model.config.n_positions
    new_n = MAX_LEN
    old_wpe = model.transformer.wpe.weight.data  # (old_n, dim)
    dim = old_wpe.size(1)
    new_wpe = torch.nn.Embedding(new_n, dim)
    with torch.no_grad():
        new_wpe.weight[:old_n].copy_(old_wpe)
        new_wpe.weight[old_n:].copy_(old_wpe[-1].unsqueeze(0).repeat(new_n - old_n, 1))
    model.transformer.wpe = new_wpe
    model.config.n_positions = new_n
    model.config.n_ctx = new_n

model.to(device)

# ---------- Optimizer & Scheduler ----------
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = max(1, EPOCHS * len(train_loader))
warmup = max(10, int(0.06 * total_steps))
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

# ---------- AMP compatibility (robust) ----------
# We create a small wrapper for autocast that handles API differences across PyTorch versions.
if use_amp:
    try:
        # prefer torch.amp if available
        GradScaler = torch.amp.GradScaler
        _amp_autocast = torch.amp.autocast

        def autocast(*args, **kwargs):
            try:
                # try calling without device_type first
                return _amp_autocast(*args, **kwargs)
            except TypeError:
                # older/newer API requires explicit device_type
                return _amp_autocast(device_type="cuda")

        scaler = GradScaler()
    except Exception:
        # fallback to torch.cuda.amp
        GradScaler = torch.cuda.amp.GradScaler
        _amp_autocast = torch.cuda.amp.autocast

        def autocast(*args, **kwargs):
            return _amp_autocast(*args, **kwargs)

        scaler = GradScaler()
else:
    class DummyScaler:
        def scale(self, loss):
            return loss

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

    def autocast(*args, **kwargs):
        return contextlib.nullcontext()

    scaler = DummyScaler()

# ---------- Training ----------
model.train()
global_step = 0
for epoch in range(1, EPOCHS + 1):
    running = 0.0
    for step, batch in enumerate(train_loader, 1):
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # safety checks
        if torch.max(input_ids).item() >= model.config.vocab_size:
            raise RuntimeError("input_ids >= vocab_size")
        if (labels != -100).any() and torch.max(labels[labels != -100]).item() >= model.config.vocab_size:
            raise RuntimeError("labels >= vocab_size")

        with autocast():
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running += loss.item()
        global_step += 1
        if step % LOG_EVERY == 0 or step == len(train_loader):
            divisor = LOG_EVERY if step % LOG_EVERY == 0 else step % LOG_EVERY
            divisor = divisor if divisor != 0 else LOG_EVERY
            print(
                f"epoch {epoch} step {step}/{len(train_loader)} loss {running/divisor:.4f}"
            )
            running = 0.0

# ---------- Save ----------
os.makedirs(MODEL_DIR, exist_ok=True)
model.save_pretrained(MODEL_DIR)
torch.save(tokenizer.vocab, VOCAB_PATH)
print("Saved model and vocab.")

# ---------- Inference ----------
def generate_reply(prompt, max_new_tokens=40, temperature=0.7, top_p=0.9, top_k=50):
    prefix_tokens = [tokenizer.bos_token] + prompt.split() + [tokenizer.sep_token]
    prefix_ids, attn = tokenizer.encode_tokens(prefix_tokens, max_length=MAX_LEN)
    prompt_length = int(sum(attn))
    trimmed_ids = prefix_ids[:prompt_length]
    trimmed_mask = [1] * prompt_length
    available_new_tokens = max(1, MAX_LEN - prompt_length)
    effective_new_tokens = min(max_new_tokens, available_new_tokens)
    input_ids = torch.tensor([trimmed_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([trimmed_mask], dtype=torch.long, device=device)
    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=effective_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=tokenizer.vocab[tokenizer.eos_token],
        pad_token_id=tokenizer.vocab[tokenizer.pad_token],
    )[0].tolist()
    sep_id = tokenizer.vocab[tokenizer.sep_token]
    try:
        sep_pos = gen_ids.index(sep_id)
    except ValueError:
        sep_pos = len(trimmed_ids) - 1
    answer_ids = gen_ids[sep_pos + 1 :]
    eos_id = tokenizer.vocab[tokenizer.eos_token]
    if eos_id in answer_ids:
        answer_ids = answer_ids[: answer_ids.index(eos_id)]
    return tokenizer.detok(answer_ids)

model.eval()
for s in ["привет як справи", "як погода", "що нового"]:
    print("Q:", s)
    print("A:", generate_reply(s))
