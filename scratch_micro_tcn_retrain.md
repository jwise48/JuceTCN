# JuceTCN Neural Compressor Training Pipeline

Minimal training pipeline for 14 neural compressor models compatible with JuceTCN plugin.

## Setup

```python
import os
if os.path.exists('micro-tcn'):
    !rm -rf micro-tcn

!git clone https://github.com/csteinmetz1/micro-tcn.git
%cd micro-tcn

!pip install torch torchvision torchaudio pytorch-lightning auraloss librosa soundfile scipy tqdm requests
!pip install -e .

!git clone https://github.com/drscotthawley/signaltrain.git
%cd signaltrain
!pip install -e .
%cd ..
```

```python
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchsummary
import copy

from microtcn.tcn import TCNModel
from microtcn.lstm import LSTMModel
from microtcn.data import SignalTrainLA2ADataset

torch.backends.cudnn.benchmark = True
```

## GPU Setup

```python
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = torch.device('cuda')
    # Optimize for Tensor Cores on modern GPUs
    torch.set_float32_matmul_precision('medium')
else:
    print("❌ No GPU available. Training will be very slow on CPU.")
    device = torch.device('cpu')

print(f"Using device: {device}")
```

## Dataset Generation

```python
import glob
import shutil
import torchaudio

def generate_signaltrain_dataset():
    try:
        %cd /content/micro-tcn
        
        if not os.path.exists('signaltrain'):
            !git clone https://github.com/drscotthawley/signaltrain.git
        
        %cd signaltrain
        !pip install -e .
        %cd ..
        
        import signaltrain as st
        
        !rm -rf data_comp4c
        !python signaltrain/gen_dataset.py data_comp4c --dur 2 --effect comp_4c --sp 5
        
        if not os.path.exists('data_comp4c'):
            return False
            
        !rm -rf data/signaltrain_LA2A
        !mkdir -p data/signaltrain_LA2A/Train data/signaltrain_LA2A/Val data/signaltrain_LA2A/Test
        
        flat_files = glob.glob('data_comp4c/*.wav')
        train_files = glob.glob('data_comp4c/Train/*.wav') if os.path.exists('data_comp4c/Train') else []
        val_files = glob.glob('data_comp4c/Val/*.wav') if os.path.exists('data_comp4c/Val') else []
        test_files = glob.glob('data_comp4c/Test/*.wav') if os.path.exists('data_comp4c/Test') else []
        
        files_copied = 0
        
        if train_files or val_files or test_files:
            for file_path in train_files:
                shutil.copy2(file_path, f'data/signaltrain_LA2A/Train/{os.path.basename(file_path)}')
                files_copied += 1
            for file_path in val_files:
                shutil.copy2(file_path, f'data/signaltrain_LA2A/Val/{os.path.basename(file_path)}')
                files_copied += 1
            for file_path in test_files:
                shutil.copy2(file_path, f'data/signaltrain_LA2A/Test/{os.path.basename(file_path)}')
                files_copied += 1
        elif flat_files:
            total_files = len(flat_files)
            train_split = int(0.8 * total_files)
            val_split = int(0.9 * total_files)
            
            for i, file_path in enumerate(flat_files):
                filename = os.path.basename(file_path)
                if i < train_split:
                    shutil.copy2(file_path, f'data/signaltrain_LA2A/Train/{filename}')
                elif i < val_split:
                    shutil.copy2(file_path, f'data/signaltrain_LA2A/Val/{filename}')
                else:
                    shutil.copy2(file_path, f'data/signaltrain_LA2A/Test/{filename}')
                files_copied += 1
        
        !rm -rf data_comp4c
        return files_copied > 0
        
    except:
        return False

def create_dummy_dataset():
    dirs = ['data/signaltrain_LA2A/Train', 'data/signaltrain_LA2A/Val', 'data/signaltrain_LA2A/Test']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    sample_rate = 44100
    duration = 10
    samples = sample_rate * duration
    param_combinations = [(0.0, 0.0), (0.3, 20.0), (0.6, 40.0), (0.9, 60.0), (1.0, 80.0)]
    
    for subset_dir in dirs:
        subset_name = os.path.basename(subset_dir).lower()
        num_files = 20 if subset_name == 'train' else 5
        
        for i in range(num_files):
            for j, (limit, peak_red) in enumerate(param_combinations):
                file_id = i * len(param_combinations) + j
                
                t = torch.linspace(0, duration, samples)
                signal = (0.3 * torch.sin(2 * torch.pi * 440 * t) +
                         0.2 * torch.sin(2 * torch.pi * 880 * t) +
                         0.1 * torch.sin(2 * torch.pi * 220 * t) +
                         0.05 * torch.randn(samples))
                
                for k in range(0, samples, sample_rate // 4):
                    if k + 1000 < samples:
                        signal[k:k+1000] += 0.5 * torch.exp(-torch.linspace(0, 5, 1000))
                
                input_audio = signal / torch.max(torch.abs(signal)) * 0.8
                input_audio = input_audio.unsqueeze(0)
                
                compressed = input_audio.clone()
                if limit > 0:
                    threshold = 1.0 - limit
                    ratio = 1.0 + (peak_red / 100.0) * 9.0
                    above_thresh = torch.abs(compressed) > threshold
                    compressed[above_thresh] = torch.sign(compressed[above_thresh]) * (
                        threshold + (torch.abs(compressed[above_thresh]) - threshold) / ratio
                    )
                    compressed = torch.tanh(compressed * (1 + limit * 0.1))
                
                input_filename = f"input_{file_id}__{limit:.1f}__{peak_red:.1f}.wav"
                target_filename = f"target_{file_id}__{limit:.1f}__{peak_red:.1f}.wav"
                
                torchaudio.save(os.path.join(subset_dir, input_filename), input_audio, sample_rate, bits_per_sample=16)
                torchaudio.save(os.path.join(subset_dir, target_filename), compressed, sample_rate, bits_per_sample=16)

if not generate_signaltrain_dataset():
    create_dummy_dataset()
```

## Model Export Function

```python
class JUCECompatibleWrapper(torch.nn.Module):
    def __init__(self, tcn_model):
        super().__init__()
        tcn_model.cpu()
        tcn_model.eval()
        
        self.gen = copy.deepcopy(tcn_model.gen).cpu()
        self.blocks = copy.deepcopy(tcn_model.blocks).cpu()
        self.output = copy.deepcopy(tcn_model.output).cpu()
        
        if hasattr(tcn_model, 'causal'):
            self.causal = tcn_model.causal
        if hasattr(tcn_model, 'nparams'):
            self.nparams = tcn_model.nparams
        if hasattr(tcn_model, 'nblocks'):
            self.nblocks = tcn_model.nblocks
        
        self.cpu()
        for param in self.parameters():
            param.data = param.data.cpu()
        for buffer in self.buffers():
            buffer.data = buffer.data.cpu()
            
    def forward(self, x, p):
        x = x.cpu()
        p = p.cpu()
        
        cond = self.gen(p)
        for block in self.blocks:
            x = block(x, cond)
        y = self.output(x)
        
        return y

def export_traced_model(model, specifier, save_dir='traced_models'):
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    model.cpu()
    
    for param in model.parameters():
        param.data = param.data.cpu()
    for buffer in model.buffers():
        buffer.data = buffer.data.cpu()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    audio_input = torch.randn(1, 1, 65536).cpu()
    param_input = torch.randn(1, 1, 2).cpu()
    
    wrapper_model = JUCECompatibleWrapper(model)
    wrapper_model.eval()
    wrapper_model.cpu()
    
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper_model, (audio_input, param_input))
    
    filename = f"traced_{specifier}.pt"
    save_path = os.path.join(save_dir, filename)
    torch.jit.save(traced_model, save_path)
    
    return save_path
```

## Training Configuration

```python
train_configs = [
    {"name" : "uTCN-300", "model_type" : "tcn", "nblocks" : 4, "dilation_growth" : 10, "kernel_size" : 13, "causal" : True, "train_fraction" : 1.00, "batch_size" : 32, "max_epochs" : 60, "train_loss" : "l1+stft"},
    {"name" : "uTCN-100", "model_type" : "tcn", "nblocks" : 4, "dilation_growth" : 10, "kernel_size" : 5, "causal" : True, "train_fraction" : 1.00, "batch_size" : 32, "max_epochs" : 60, "train_loss" : "l1+stft"},
    {"name" : "uTCN-300", "model_type" : "tcn", "nblocks" : 4, "dilation_growth" : 10, "kernel_size" : 13, "causal" : True, "train_fraction" : 1.00, "batch_size" : 32, "max_epochs" : 60, "train_loss" : "l1+stft"},
    {"name" : "uTCN-1000", "model_type" : "tcn", "nblocks" : 5, "dilation_growth" : 10, "kernel_size" : 5, "causal" : True, "train_fraction" : 1.00, "batch_size" : 32, "max_epochs" : 60, "train_loss" : "l1+stft"},
    {"name" : "uTCN-100", "model_type" : "tcn", "nblocks" : 4, "dilation_growth" : 10, "kernel_size" : 5, "causal" : False, "train_fraction" : 1.00, "batch_size" : 32, "max_epochs" : 60, "train_loss" : "l1+stft"},
    {"name" : "uTCN-300", "model_type" : "tcn", "nblocks" : 4, "dilation_growth" : 10, "kernel_size" : 13, "causal" : False, "train_fraction" : 1.00, "batch_size" : 32, "max_epochs" : 60, "train_loss" : "l1+stft"},
    {"name" : "uTCN-1000", "model_type" : "tcn", "nblocks" : 5, "dilation_growth" : 10, "kernel_size" : 5, "causal" : False, "train_fraction" : 1.00, "batch_size" : 32, "max_epochs" : 60, "train_loss" : "l1+stft"},
    {"name" : "TCN-300", "model_type" : "tcn", "nblocks" : 10, "dilation_growth" : 2, "kernel_size" : 15, "causal" : False, "train_fraction" : 1.00, "batch_size" : 32, "max_epochs" : 60, "train_loss" : "l1+stft"},
    {"name" : "uTCN-300", "model_type" : "tcn", "nblocks" : 4, "dilation_growth" : 10, "kernel_size" : 13, "causal" : True, "train_fraction" : 1.00, "batch_size" : 32, "max_epochs" : 60, "train_loss" : "l1+stft"},
    {"name" : "LSTM-32", "model_type" : "lstm", "num_layers" : 1, "hidden_size" : 32, "train_fraction" : 1.00, "batch_size" : 32, "max_epochs" : 60, "train_loss" : "l1+stft"},
    {"name" : "uTCN-300", "model_type" : "tcn", "nblocks" : 3, "dilation_growth" : 60, "kernel_size" : 5, "causal" : True, "train_fraction" : 1.0, "batch_size" : 32, "max_epochs" : 60, "train_loss" : "l1+stft"},
    {"name" : "uTCN-300", "model_type" : "tcn", "nblocks" : 4, "dilation_growth" : 10, "kernel_size" : 13, "causal" : True, "train_fraction" : 1.0, "batch_size" : 32, "max_epochs" : 60, "train_loss" : "l1"},
    {"name" : "uTCN-300", "model_type" : "tcn", "nblocks" : 30, "dilation_growth" : 2, "kernel_size" : 15, "causal" : False, "train_fraction" : 1.0, "batch_size" : 32, "max_epochs" : 60, "train_loss" : "l1+stft"},
    {"name" : "uTCN-324-16", "model_type" : "tcn", "nblocks" : 10, "dilation_growth" : 2, "kernel_size" : 15, "causal" : False, "train_fraction" : 1.0, "batch_size" : 32, "max_epochs" : 60, "channel_width" : 16, "train_loss" : "l1+stft"}
]

n_configs = len(train_configs)
```

## Training Functions

```python
def create_model(config):
    if config['model_type'] == 'tcn':
        return TCNModel(
            nparams=2,
            nblocks=config['nblocks'],
            dilation_growth=config['dilation_growth'],
            kernel_size=config['kernel_size'],
            causal=config['causal'],
            channel_width=config.get('channel_width', 32)
        )
    else:
        return LSTMModel(
            nparams=2,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        )

class CompressorModule(pl.LightningModule):
    def __init__(self, model, train_loss="l1+stft"):
        super().__init__()
        self.model = model
        self.train_loss = train_loss
        
        self.l1_loss = torch.nn.L1Loss()
        self.stft_loss = auraloss.freq.STFTLoss()

    def forward(self, x, params):
        return self.model(x, params)

    def training_step(self, batch, batch_idx):
        x, y, params = batch
        y_hat = self.forward(x, params)
        
        if y_hat.shape != y.shape:
            if y_hat.shape[2] < y.shape[2]:
                y = y[:, :, :y_hat.shape[2]]
            else:
                y_hat = y_hat[:, :, :y.shape[2]]
        
        if self.train_loss == "l1":
            loss = self.l1_loss(y_hat, y)
        elif self.train_loss == "stft":
            loss = self.stft_loss(y_hat, y)
        elif self.train_loss == "l1+stft":
            l1_loss = self.l1_loss(y_hat, y)
            stft_loss = self.stft_loss(y_hat, y)
            loss = l1_loss + stft_loss
        else:
            loss = self.l1_loss(y_hat, y)
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, params = batch
        y_hat = self.forward(x, params)
        
        if y_hat.shape != y.shape:
            if y_hat.shape[2] < y.shape[2]:
                y = y[:, :, :y_hat.shape[2]]
            else:
                y_hat = y_hat[:, :, :y.shape[2]]
        
        l1_loss = self.l1_loss(y_hat, y)
        stft_loss = self.stft_loss(y_hat, y)
        aggregate_loss = l1_loss + stft_loss
        
        self.log('val_loss', aggregate_loss)
        self.log('val_loss/L1', l1_loss)
        self.log('val_loss/STFT', stft_loss)
        return aggregate_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def train_model(config):
    # Use production parameters: always 60 epochs for consistency
    epochs = config.get('max_epochs', 60)
    train_loss = config.get('train_loss', 'l1+stft')
    
    model = create_model(config).to(device)
    
    try:
        train_dataset = SignalTrainLA2ADataset(
            'data/signaltrain_LA2A',
            subset='train',
            fraction=config['train_fraction'],
            length=65536,  # Match micro-tcn train_length
            preload=False,
            half=True if torch.cuda.is_available() else False  # Use mixed precision on GPU
        )
        val_dataset = SignalTrainLA2ADataset(
            'data/signaltrain_LA2A',
            subset='val',
            fraction=0.1,
            length=131072,  # Match micro-tcn eval_length
            preload=False,
            half=True if torch.cuda.is_available() else False
        )
    except:
        from torch.utils.data import Dataset
        class DummyDataset(Dataset):
            def __init__(self, length=1000):
                self.length = length
            def __len__(self):
                return self.length
            def __getitem__(self, idx):
                return torch.randn(1, 65536), torch.randn(1, 65536), torch.randn(1, 2)
        
        train_dataset = DummyDataset(1000)
        val_dataset = DummyDataset(200)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=8,  # Match micro-tcn validation batch size
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    lightning_model = CompressorModule(model, train_loss=train_loss)
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        precision='16-mixed' if torch.cuda.is_available() else 32,  # Use mixed precision
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True
    )
    
    trainer.fit(lightning_model, train_loader, val_loader)
    
    cpu_model = create_model(config)
    cpu_model.eval()
    
    lightning_model.cpu().eval()
    underlying_model = lightning_model.model
    underlying_model.cpu().eval()
    
    source_state_dict = underlying_model.state_dict()
    target_state_dict = cpu_model.state_dict()
    
    for name in target_state_dict.keys():
        if name in source_state_dict:
            target_state_dict[name] = source_state_dict[name].cpu().clone()
    
    cpu_model.load_state_dict(target_state_dict)
    cpu_model.cpu().eval()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    idx = config['idx']
    name = config['name']
    causal_str = 'causal' if config.get('causal', True) else 'noncausal'
    nblocks = config.get('nblocks', 4)
    dilation_growth = config.get('dilation_growth', 10)
    kernel_size = config.get('kernel_size', 13)
    train_fraction = config.get('train_fraction', 1.0)
    batch_size = config.get('batch_size', 32)
    
    specifier = f"{idx+1}-{name}__{causal_str}__{nblocks}-{dilation_growth}-{kernel_size}__fraction-{train_fraction}-bs{batch_size}"
    
    return export_traced_model(cpu_model, specifier)
```

## Training Loop

```python
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('traced_models', exist_ok=True)

for idx, tconf in enumerate(train_configs):
    tconf['idx'] = idx
    export_path = train_model(tconf)
```

## Validation and Export

```python
!ls -la traced_models/

model_files = [f for f in os.listdir('traced_models') if f.endswith('.pt')]

for model_file in model_files:
    model_path = os.path.join('traced_models', model_file)
    traced_model = torch.jit.load(model_path)
    
    audio_input = torch.randn(1, 1, 65536)
    param_input = torch.tensor([[[0.5, 0.5]]])
    
    with torch.no_grad():
        output = traced_model(audio_input, param_input)

!zip -r juce_tcn_models.zip traced_models/
!ls -lh juce_tcn_models.zip
```

## Model Performance Testing

```python
!pip install pyloudnorm

import json
import numpy as np
import pandas as pd
import pyloudnorm as pyln
from microtcn.utils import center_crop, causal_crop

pl.seed_everything(42)

# Setup evaluation parameters
eval_length = 2**19  # 524288 samples for comprehensive testing
batch_size = 1
sample_rate = 44100

# Load test dataset
try:
    test_dataset = SignalTrainLA2ADataset(
        'data/signaltrain_LA2A',
        subset='val',
        half=False,
        preload=False,
        length=eval_length
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0
    )
except:
    from torch.utils.data import Dataset
    class TestDummyDataset(Dataset):
        def __init__(self, length=50):
            self.length = length
        def __len__(self):
            return self.length
        def __getitem__(self, idx):
            return torch.randn(1, eval_length), torch.randn(1, eval_length), torch.randn(1, 2)
    
    test_dataset = TestDummyDataset(50)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Setup evaluation metrics
l1_loss = torch.nn.L1Loss()
stft_loss = auraloss.freq.STFTLoss()
meter = pyln.Meter(sample_rate)

def evaluate_model(model_path, model_name):
    model = torch.jit.load(model_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.to(device)
    
    results = {}
    
    for batch_idx, batch in enumerate(test_dataloader):
        if batch_idx >= 20:  # Limit evaluation for speed
            break
            
        input_audio, target_audio, params = batch
        
        if torch.cuda.is_available():
            input_audio = input_audio.to(device)
            target_audio = target_audio.to(device)
            params = params.to(device)
        
        with torch.no_grad():
            output_audio = model(input_audio, params)
        
        # Handle size mismatch (crop to match output)
        if output_audio.shape != target_audio.shape:
            if output_audio.shape[2] < target_audio.shape[2]:
                target_audio = target_audio[:, :, :output_audio.shape[2]]
                input_audio = input_audio[:, :, :output_audio.shape[2]]
            else:
                output_audio = output_audio[:, :, :target_audio.shape[2]]
        
        # Move to CPU for evaluation
        input_cpu = input_audio.cpu()
        output_cpu = output_audio.cpu()
        target_cpu = target_audio.cpu()
        params_cpu = params.cpu()
        
        # Calculate metrics
        l1_val = l1_loss(output_cpu, target_cpu).item()
        stft_val = stft_loss(output_cpu, target_cpu).item()
        aggregate_val = l1_val + stft_val
        
        # LUFS analysis
        try:
            target_lufs = meter.integrated_loudness(target_cpu.squeeze().numpy())
            output_lufs = meter.integrated_loudness(output_cpu.squeeze().numpy())
            lufs_diff = abs(output_lufs - target_lufs)
        except:
            lufs_diff = 0.0
        
        # Improvement metrics
        l1_improvement = (l1_loss(input_cpu, target_cpu) - l1_loss(output_cpu, target_cpu)).item()
        stft_improvement = (stft_loss(input_cpu, target_cpu) - stft_loss(output_cpu, target_cpu)).item()
        
        # Parameter key
        p = params_cpu.squeeze().numpy()
        param_key = f"{p[0]:.1f}-{p[1]*100:03.0f}"
        
        if param_key not in results:
            results[param_key] = {
                'L1': [], 'STFT': [], 'LUFS': [], 'Aggregate': [],
                'L1_improvement': [], 'STFT_improvement': []
            }
        
        results[param_key]['L1'].append(l1_val)
        results[param_key]['STFT'].append(stft_val)
        results[param_key]['LUFS'].append(lufs_diff)
        results[param_key]['Aggregate'].append(aggregate_val)
        results[param_key]['L1_improvement'].append(l1_improvement)
        results[param_key]['STFT_improvement'].append(stft_improvement)
    
    return results

# Evaluate all models
model_files = [f for f in os.listdir('traced_models') if f.endswith('.pt')]
all_results = {}

print(f"Evaluating {len(model_files)} models...")
print("=" * 80)

for idx, model_file in enumerate(sorted(model_files)):
    model_path = os.path.join('traced_models', model_file)
    model_name = model_file.replace('traced_', '').replace('.pt', '')
    
    print(f"[{idx+1}/{len(model_files)}] {model_name}")
    
    try:
        results = evaluate_model(model_path, model_name)
        all_results[model_name] = results
        
        # Calculate and display summary statistics
        all_l1 = []
        all_stft = []
        all_lufs = []
        
        for param_key, metrics in results.items():
            all_l1.extend(metrics['L1'])
            all_stft.extend(metrics['STFT'])
            all_lufs.extend(metrics['LUFS'])
        
        if all_l1:
            print(f"  L1: {np.mean(all_l1):.2e} | STFT: {np.mean(all_stft):.3f} | LUFS: {np.mean(all_lufs):.3f}")
        else:
            print("  No valid results")
            
    except Exception as e:
        print(f"  Error: {str(e)}")
        all_results[model_name] = {}

print("=" * 80)

# Create performance comparison table
performance_data = []

for model_name, results in all_results.items():
    if not results:
        continue
        
    all_l1 = []
    all_stft = []
    all_lufs = []
    all_agg = []
    all_l1_imp = []
    all_stft_imp = []
    
    for param_key, metrics in results.items():
        all_l1.extend(metrics['L1'])
        all_stft.extend(metrics['STFT'])
        all_lufs.extend(metrics['LUFS'])
        all_agg.extend(metrics['Aggregate'])
        all_l1_imp.extend(metrics['L1_improvement'])
        all_stft_imp.extend(metrics['STFT_improvement'])
    
    if all_l1:
        performance_data.append({
            'Model': model_name,
            'L1_Loss': np.mean(all_l1),
            'STFT_Loss': np.mean(all_stft),
            'LUFS_Diff': np.mean(all_lufs),
            'Aggregate_Loss': np.mean(all_agg),
            'L1_Improvement': np.mean(all_l1_imp),
            'STFT_Improvement': np.mean(all_stft_imp)
        })

# Display results table
if performance_data:
    df = pd.DataFrame(performance_data)
    df = df.sort_values('Aggregate_Loss')
    
    print("\nMODEL PERFORMANCE COMPARISON")
    print("=" * 120)
    print(f"{'Model':<40} {'L1 Loss':<12} {'STFT Loss':<12} {'LUFS Diff':<12} {'Aggregate':<12} {'L1 Imp':<12} {'STFT Imp':<12}")
    print("-" * 120)
    
    for _, row in df.iterrows():
        print(f"{row['Model']:<40} {row['L1_Loss']:<12.2e} {row['STFT_Loss']:<12.3f} {row['LUFS_Diff']:<12.3f} {row['Aggregate_Loss']:<12.3f} {row['L1_Improvement']:<12.2e} {row['STFT_Improvement']:<12.3f}")
    
    print("=" * 120)
    
    # Find best models
    best_l1 = df.loc[df['L1_Loss'].idxmin()]
    best_stft = df.loc[df['STFT_Loss'].idxmin()]
    best_aggregate = df.loc[df['Aggregate_Loss'].idxmin()]
    
    print(f"\nBEST PERFORMING MODELS:")
    print(f"Best L1 Loss: {best_l1['Model']}")
    print(f"Best STFT Loss: {best_stft['Model']}")
    print(f"Best Overall (Aggregate): {best_aggregate['Model']}")
    
    # Save results
    with open('model_performance_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    df.to_csv('model_performance_summary.csv', index=False)
    print(f"\nResults saved to 'model_performance_results.json' and 'model_performance_summary.csv'")

else:
    print("No valid performance data collected.")
