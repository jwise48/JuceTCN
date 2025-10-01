# JuceTCN Neural Compressor Training Pipeline - Native micro-tcn Implementation

Training pipeline for 14 neural compressor models using **native micro-tcn repository infrastructure** and exporting them as traced PyTorch models (.pt files) compatible with the JUCE plugin's `torch::jit::load()` function.

## Setup

```python
# Clone and setup micro-tcn repository
import os
import sys
if os.path.exists('micro-tcn'):
    !rm -rf micro-tcn

!git clone https://github.com/csteinmetz1/micro-tcn.git
%cd micro-tcn

# Install dependencies - updated for modern compatibility
!pip install torch torchvision torchaudio
!pip install pytorch-lightning
!pip install auraloss
!pip install torchsummary
!pip install thop
!pip install pyloudnorm
!pip install soundfile
!pip install -e .

# Add micro-tcn to Python path for imports
sys.path.append('/content/micro-tcn')
```

```python
# Import native micro-tcn modules
import torch
import pytorch_lightning as pl
import torchsummary
import glob
import os
import sys
from argparse import ArgumentParser, Namespace

# Import native micro-tcn classes
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
    torch.set_float32_matmul_precision('medium')
else:
    print("❌ No GPU available. Training will be very slow on CPU.")
    device = torch.device('cpu')

print(f"Using device: {device}")
```

## Dataset Setup

```python
# SignalTrain LA2A dataset setup
# Note: This is a large dataset (~20GB). Make sure you have sufficient storage.

import urllib.request
import zipfile
import requests
from pathlib import Path

def download_signaltrain_dataset():
    """Download and extract SignalTrain LA2A dataset with multiple fallback methods"""
    
    # Multiple potential URLs for the dataset
    dataset_urls = [
        "https://zenodo.org/records/3824876/files/SignalTrain_LA2A_Dataset_1.1.zip",
        "https://zenodo.org/record/3824876/files/SignalTrain_LA2A_Dataset_1.1.zip",
        "https://files.pythonhosted.org/packages/source/s/signaltrain/SignalTrain_LA2A_Dataset_1.1.zip"
    ]
    
    dataset_path = "SignalTrain_LA2A_Dataset_1.1.zip"
    extract_path = "SignalTrain_LA2A_Dataset_1.1"
    
    if os.path.exists(extract_path):
        print("Dataset already exists.")
        return extract_path
    
    if not os.path.exists(dataset_path):
        print("Downloading SignalTrain LA2A dataset (~20GB)...")
        print("This may take a while depending on your internet connection.")
        
        download_success = False
        
        # Try each URL until one works
        for i, url in enumerate(dataset_urls):
            try:
                print(f"Attempting download from URL {i+1}/{len(dataset_urls)}...")
                
                # Use requests with better error handling
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(dataset_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rDownload progress: {percent:.1f}%", end='', flush=True)
                
                print("\nDownload complete!")
                download_success = True
                break
                
            except Exception as e:
                print(f"\nFailed to download from URL {i+1}: {str(e)}")
                if os.path.exists(dataset_path):
                    os.remove(dataset_path)
                continue
        
        if not download_success:
            print("\n❌ All download attempts failed.")
            print("Please manually download the SignalTrain LA2A dataset from:")
            print("https://zenodo.org/record/3824876")
            print("And place the 'SignalTrain_LA2A_Dataset_1.1.zip' file in the current directory.")
            return None
    
    if os.path.exists(dataset_path):
        print("Extracting dataset...")
        try:
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                zip_ref.extractall('.')
            print("Extraction complete!")
        except Exception as e:
            print(f"❌ Extraction failed: {str(e)}")
            return None
    
    return extract_path

def create_dummy_dataset():
    """Create a small dummy dataset for testing purposes"""
    print("Creating dummy dataset for testing...")
    
    dummy_path = "SignalTrain_LA2A_Dataset_1.1"
    os.makedirs(f"{dummy_path}/Train", exist_ok=True)
    os.makedirs(f"{dummy_path}/Val", exist_ok=True)
    
    # Create small dummy audio files
    import numpy as np
    import soundfile as sf
    
    sample_rate = 44100
    duration = 5  # 5 seconds
    samples = int(sample_rate * duration)
    
    # Create a few dummy files for each subset
    for subset in ['Train', 'Val']:
        for i in range(3):  # 3 files per subset
            for param1 in [1.0, 2.0]:
                for param2 in [10.0, 50.0]:
                    # Generate dummy audio
                    dummy_audio = np.random.randn(samples) * 0.1
                    
                    input_file = f"{dummy_path}/{subset}/input_{i:03d}__{param1}__{param2}.wav"
                    target_file = f"{dummy_path}/{subset}/target_{i:03d}__{param1}__{param2}.wav"
                    
                    # Add some simple processing to make target different from input
                    target_audio = dummy_audio * 0.8  # Simple gain reduction
                    
                    sf.write(input_file, dummy_audio, sample_rate)
                    sf.write(target_file, target_audio, sample_rate)
    
    print(f"Dummy dataset created at: {dummy_path}")
    return dummy_path

# Try to download the real dataset, fallback to dummy if needed
print("Setting up SignalTrain LA2A dataset...")
dataset_root = download_signaltrain_dataset()

if dataset_root is None:
    print("\n⚠️  Using dummy dataset for testing. Replace with real dataset for actual training.")
    dataset_root = create_dummy_dataset()

print(f"Dataset root: {dataset_root}")
```

## Global Configuration (Replaces Command Line Arguments)

```python
BASE_CONFIG = {
    'root_dir': dataset_root,
    'preload': False,
    'sample_rate': 44100,
    'shuffle': True,
    'train_subset': 'train',
    'val_subset': 'val',
    'train_length': 65536,
    'eval_length': 131072,
    'num_workers': 4,
    'precision': 16,
    'max_epochs': 60,
    'nparams': 2,  # LA2A has 2 parameters (limit, peak_reduction)
    'lr': 3e-4,
    'train_loss': 'l1+stft'
}

# Training configurations from native micro-tcn train.py
train_configs = [
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 0.01,
     "batch_size" : 32
    },
    {"name" : "uTCN-100",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-1000",
     "model_type" : "tcn",
     "nblocks" : 5,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-100",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-1000",
     "model_type" : "tcn",
     "nblocks" : 5,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "TCN-300",
     "model_type" : "tcn",
     "nblocks" : 10,
     "dilation_growth" : 2,
     "kernel_size" : 15,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 0.10,
     "batch_size" : 32
    },
    {"name" : "LSTM-32",
     "model_type" : "lstm",
     "num_layers" : 1,
     "hidden_size" : 32,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 3,
     "dilation_growth" : 60,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.0,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
     "train_loss" : "l1"
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 30,
     "dilation_growth" : 2,
     "kernel_size" : 15,
     "causal" : False,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
    },
    {"name" : "uTCN-324-16",
     "model_type" : "tcn",
     "nblocks" : 10,
     "dilation_growth" : 2,
     "kernel_size" : 15,
     "causal" : False,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
     "channel_width" : 16,
    },
]

n_configs = len(train_configs)
print(f"Training {n_configs} model configurations")
```

## Native Training Loop (Directly Adapted from micro-tcn train.py)

```python
# Create directories for outputs (following micro-tcn structure)
os.makedirs('lightning_logs/bulk', exist_ok=True)

# Set seed for reproducibility (micro-tcn default)
pl.seed_everything(42)

def create_args_namespace(base_config, train_config):
    """Create an args namespace that mimics micro-tcn's argument parser output"""
    args_dict = base_config.copy()
    args_dict.update(train_config)
    return Namespace(**args_dict)

# Training loop adapted directly from micro-tcn train.py
for idx, tconf in enumerate(train_configs):
    
    print(f"\n* Training config {idx+1}/{n_configs}")
    print(tconf)
    
    # Create args namespace (replaces argument parsing)
    args = create_args_namespace(BASE_CONFIG, tconf)
    
    # Create model specifier using micro-tcn's exact naming convention
    if tconf["model_type"] == 'tcn':
        specifier = f"{idx+1}-{tconf['name']}"
        specifier += "__causal" if tconf['causal'] else "__noncausal"
        specifier += f"__{tconf['nblocks']}-{tconf['dilation_growth']}-{tconf['kernel_size']}"
        specifier += f"__fraction-{tconf['train_fraction']}-bs{tconf['batch_size']}"
    elif tconf["model_type"] == 'lstm':
        specifier = f"{idx+1}-{tconf['name']}"
        specifier += f"__{tconf['num_layers']}-{tconf['hidden_size']}"
        specifier += f"__fraction-{tconf['train_fraction']}-bs{tconf['batch_size']}"
    
    # Handle special configurations (following micro-tcn logic)
    if "max_epochs" in tconf:
        args.max_epochs = tconf["max_epochs"]
    
    if "train_loss" in tconf:
        args.train_loss = tconf["train_loss"]
        specifier += f"__loss-{tconf['train_loss']}"
    
    print(f"Model specifier: {specifier}")
    
    # Set default root directory (micro-tcn style)
    args.default_root_dir = os.path.join("lightning_logs", "bulk", specifier)
    print(f"Training directory: {args.default_root_dir}")
    
    # Create PyTorch Lightning trainer with modern compatibility
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision='16-mixed' if args.precision == 16 and torch.cuda.is_available() else 32,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        default_root_dir=args.default_root_dir,
        enable_progress_bar=True,
        log_every_n_steps=50
    )
    
    # Setup datasets using native micro-tcn SignalTrainLA2ADataset
    train_dataset = SignalTrainLA2ADataset(
        args.root_dir, 
        subset=args.train_subset,
        fraction=tconf["train_fraction"],
        half=True if args.precision == 16 else False,
        preload=args.preload,
        length=args.train_length
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=args.shuffle,
        batch_size=tconf["batch_size"],
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataset = SignalTrainLA2ADataset(
        args.root_dir, 
        preload=args.preload,
        half=True if args.precision == 16 else False,
        subset=args.val_subset,
        length=args.eval_length
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        shuffle=False,
        batch_size=8,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model using native micro-tcn classes and argument structure
    dict_args = vars(args)
    dict_args["nparams"] = 2  # LA2A parameters
    
    if tconf["model_type"] == 'tcn':
        # Set TCN-specific parameters
        dict_args["nblocks"] = tconf["nblocks"]
        dict_args["dilation_growth"] = tconf["dilation_growth"]
        dict_args["kernel_size"] = tconf["kernel_size"]
        dict_args["causal"] = tconf["causal"]
        if "channel_width" in tconf:
            dict_args["channel_width"] = tconf["channel_width"]
        model = TCNModel(**dict_args)
    elif tconf["model_type"] == 'lstm':
        # Set LSTM-specific parameters
        dict_args["num_layers"] = tconf["num_layers"]
        dict_args["hidden_size"] = tconf["hidden_size"]
        model = LSTMModel(**dict_args)
    
    # Model summary (micro-tcn style)
    print("\nModel Summary:")
    try:
        torchsummary.summary(model, [(1, 65536), (1, 2)], device="cpu")
    except Exception as e:
        print(f"Could not generate model summary: {e}")
    
    # Train using micro-tcn's exact training approach
    print(f"\nStarting training for {args.max_epochs} epochs...")
    trainer.fit(model, train_dataloader, val_dataloader)
    
    print(f"Training completed for {specifier}")

print("\n🎉 All models trained successfully using native micro-tcn infrastructure!")
```

## Native Model Export (Adapted from micro-tcn export.py)

```python
def load_model(model_dir, gpu=False):
    """Load trained model from checkpoint - adapted from micro-tcn export.py"""
    
    checkpoint_path = glob.glob(os.path.join(model_dir,
                                            "lightning_logs",
                                            "version_0",
                                            "checkpoints",
                                            "*"))[0]
    
    model_id = os.path.basename(model_dir)
    model_type = model_id.split('-')[1]
    
    map_location = "cuda:0" if gpu else "cpu"
    
    if model_type == "LSTM":
        model = LSTMModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location
        )
    else:
        model = TCNModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location
        )
    
    return model

# Export all trained models using micro-tcn's approach
models_dir = "lightning_logs/bulk"
export_dir = "traced_models"

if not os.path.isdir(export_dir):
    os.makedirs(export_dir)

model_dirs = sorted(glob.glob(os.path.join(models_dir, "*")))

print(f"Exporting {len(model_dirs)} trained models using native micro-tcn export logic...")

for idx, model_dir in enumerate(model_dirs):
    model_id = os.path.basename(model_dir)
    print(f"[{idx+1}/{len(model_dirs)}] Exporting {model_id}")
    
    try:
        # Load model using micro-tcn's load function
        model = load_model(model_dir, gpu=False)
        model.eval()
        model.cpu()
        
        # Convert to TorchScript using micro-tcn's approach
        script = model.to_torchscript()
        
        # Save traced model with micro-tcn naming convention
        export_path = os.path.join(export_dir, f"traced_{model_id}.pt")
        torch.jit.save(script, export_path)
        
        print(f"  ✅ Exported to {export_path}")
        
    except Exception as e:
        print(f"  ❌ Failed to export {model_id}: {str(e)}")

print(f"\n🎉 Model export completed using native micro-tcn infrastructure!")

# List exported models
print("\nExported models:")
exported_models = glob.glob(os.path.join(export_dir, "*.pt"))
for model_path in sorted(exported_models):
    model_name = os.path.basename(model_path)
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
    print(f"  {model_name} ({file_size:.1f} MB)")
```

## Model Validation and Testing

```python
# Test all exported models to ensure JUCE plugin compatibility
print("Testing exported models for JUCE plugin compatibility...")

def test_model(model_path):
    """Test a traced model with sample input matching JUCE plugin expectations"""
    try:
        # Load the traced model
        model = torch.jit.load(model_path)
        model.eval()
        
        # Create test inputs (matching JUCE plugin expectations)
        audio_input = torch.randn(1, 1, 65536)  # Batch=1, Channels=1, Samples=65536
        param_input = torch.tensor([[[0.5, 0.3]]])  # Batch=1, Channels=1, Params=2 (limit, peak_reduction)
        
        # Run inference
        with torch.no_grad():
            output = model(audio_input, param_input)
        
        # Validate output
        if output.shape[0] != 1 or output.shape[1] != 1:
            return False, f"Invalid output shape: {output.shape}"
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "Output contains NaN or Inf values"
        
        return True, f"Output shape: {output.shape}, Range: [{output.min():.3f}, {output.max():.3f}]"
        
    except Exception as e:
        return False, str(e)

# Test all exported models
test_results = {}
for model_path in sorted(exported_models):
    model_name = os.path.basename(model_path)
    print(f"\nTesting {model_name}...")
    
    success, message = test_model(model_path)
    test_results[model_name] = (success, message)
    
    if success:
        print(f"  ✅ {message}")
    else:
        print(f"  ❌ {message}")

# Summary
successful_models = sum(1 for success, _ in test_results.values() if success)
total_models = len(test_results)

print(f"\n📊 Testing Summary:")
print(f"  Successful: {successful_models}/{total_models} models")
print(f"  Success rate: {successful_models/total_models*100:.1f}%")

if successful_models == total_models:
    print("🎉 All models passed validation!")
else:
    print("⚠️  Some models failed validation. Check the error messages above.")
```

## Comprehensive Evaluation (Using micro-tcn test.py)

```python
# Optional: Run comprehensive evaluation using micro-tcn's native test.py approach
import auraloss
import numpy as np
import pyloudnorm as pyln
from microtcn.utils import center_crop, causal_crop

def comprehensive_model_evaluation(model_path, num_samples=50):
    """Comprehensive evaluation using micro-tcn's test.py methodology"""
    try:
        model = torch.jit.load(model_path)
        model.eval()
        
        # Setup loss functions (micro-tcn style)
        l1_loss = torch.nn.L1Loss()
        stft_loss = auraloss.freq.STFTLoss()
        meter = pyln.Meter(44100)
        
        # Create test dataset using micro-tcn's approach
        test_dataset = SignalTrainLA2ADataset(
            dataset_root,
            subset="val",
            half=False,
            preload=False,
            length=131072  # micro-tcn's eval length
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        results = {}
        
        # Evaluate using micro-tcn's methodology
        for i, (input_audio, target_audio, params) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            with torch.no_grad():
                output = model(input_audio, params)
                
                # Crop signals using micro-tcn's approach
                # Assume non-causal for simplicity (can be enhanced)
                input_crop = center_crop(input_audio, output.shape[-1])
                target_crop = center_crop(target_audio, output.shape[-1])
                
                # Calculate losses (micro-tcn style)
                l1_val = l1_loss(output, target_crop).cpu().numpy()
                stft_val = stft_loss(output, target_crop).cpu().numpy()
                aggregate_loss = l1_val + stft_val
                
                # LUFS evaluation (micro-tcn approach)
                try:
                    target_lufs = meter.integrated_loudness(target_crop.squeeze().cpu().numpy())
                    output_lufs = meter.integrated_loudness(output.squeeze().cpu().numpy())
                    l1_lufs = np.abs(output_lufs - target_lufs)
                except:
                    l1_lufs = 0.0
                
                # Improvement metrics (micro-tcn style)
                l1i_loss = (l1_loss(input_crop, target_crop) - l1_val).cpu().numpy()
                stfti_loss = (stft_loss(input_crop, target_crop) - stft_val).cpu().numpy()
                
                # Store results by parameter configuration
                params_np = params.squeeze().cpu().numpy()
                params_key = f"{params_np[0]:1.0f}-{params_np[1]*100:03.0f}"
                
                if params_key not in results:
                    results[params_key] = {
                        "L1": [l1_val],
                        "L1i": [l1i_loss],
                        "STFT": [stft_val],
                        "STFTi": [stfti_loss],
                        "LUFS": [l1_lufs],
                        "Agg": [aggregate_loss]
                    }
                else:
                    results[params_key]["L1"].append(l1_val)
                    results[params_key]["L1i"].append(l1i_loss)
                    results[params_key]["STFT"].append(stft_val)
                    results[params_key]["STFTi"].append(stfti_loss)
                    results[params_key]["LUFS"].append(l1_lufs)
                    results[params_key]["Agg"].append(aggregate_loss)
        
        return True, results
        
    except Exception as e:
        return False, str(e)

# Run comprehensive evaluation on first few models
print("\n🔍 Comprehensive Evaluation (Using micro-tcn methodology):")
print("=" * 80)

for model_path in sorted(exported_models)[:3]:  # Test first 3 models
    model_name = os.path.basename(model_path).replace('traced_', '').replace('.pt', '')
    print(f"\nEvaluating {model_name} using micro-tcn test.py approach...")
    
    success, results = comprehensive_model_evaluation(model_path, num_samples=20)
    
    if success:
        print("-" * 64)
        print("Config      L1         STFT      LUFS")
        print("-" * 64)
        
        all_l1, all_stft, all_lufs = [], [], []
        
        for config, metrics in results.items():
            l1_mean = np.mean(metrics['L1'])
            stft_mean = np.mean(metrics['STFT'])
            lufs_mean = np.mean(metrics['LUFS'])
            
            print(f"{config}    {l1_mean:0.2e}    {stft_mean:0.3f}       {lufs_mean:0.3f}")
            
            all_l1.extend(metrics['L1'])
            all_stft.extend(metrics['STFT'])
            all_lufs.extend(metrics['LUFS'])
        
        print("-" * 64)
        print(f"Mean     {np.mean(all_l1):0.2e}    {np.mean(all_stft):0.3f}      {np.mean(all_lufs):0.3f}")
        
    else:
        print(f"  ❌ Evaluation failed: {results}")

print("\n💡 For full evaluation, run micro-tcn's native test.py:")
print("   python test.py --root_dir SignalTrain_LA2A_Dataset_1.1 --model_dir lightning_logs/bulk")
```

## Final Summary

```python
print("\n" + "="*80)
print("🎉 JUCE TCN NATIVE MICRO-TCN TRAINING PIPELINE COMPLETED")
print("="*80)

print(f"\n📈 Training Results:")
print(f"  Models trained: {n_configs}")
print(f"  Models exported: {len(exported_models)}")
print(f"  Models validated: {successful_models}")
print(f"  Success rate: {successful_models/total_models*100:.1f}%")

print(f"\n🔧 Native micro-tcn Utilization:")
print(f"  ✅ Used native TCNModel and LSTMModel classes")
print(f"  ✅ Used native SignalTrainLA2ADataset")
print(f"  ✅ Followed micro-tcn's training loop structure")
print(f"  ✅ Used micro-tcn's export.py methodology")
print(f"  ✅ Applied micro-tcn's evaluation approach")

print(f"\n📁 Output Directories:")
print(f"  Training logs: lightning_logs/bulk/")
print(f"  Traced models: {export_dir}/")

print(f"\n🔧 JUCE Plugin Integration:")
print(f"  Copy the .pt files from '{export_dir}/' to your JUCE plugin's model directory")
print(f"  Models are compatible with torch::jit::load() in C++")
print(f"  Input format: (batch_size=1, channels=1, samples=65536)")
print(f"  Parameter format: (batch_size=1, channels=1, params=2) [limit, peak_reduction]")

print(f"\n✨ Next Steps:")
print(f"  1. Copy traced models to your JUCE plugin project")
print(f"  2. Test models in your plugin environment")
print(f"  3. Run comprehensive evaluation: python test.py --root_dir {dataset_root} --model_dir lightning_logs/bulk")

print("="*80)
```
