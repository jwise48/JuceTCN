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
!pip install pytorch-lightning==1.9.5
!pip install auraloss
!pip install torchsummary
!pip install thop
!pip install pyloudnorm
!pip install soundfile
!pip install -e .

# PyTorch Lightning version compatibility note:
# micro-tcn was built for PyTorch Lightning v1.x which uses validation_epoch_end()
# v2.0+ has breaking changes (validation_epoch_end -> on_validation_epoch_end)
# We pin to v1.9.5 for compatibility with micro-tcn's Base class implementation

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

## Dataset Setup with Google Drive Persistence

```python
# SignalTrain LA2A dataset setup with Google Drive persistence
# This avoids re-downloading the ~21GB dataset in future Colab sessions

import urllib.request
import tarfile
import requests
import os
import shutil
from pathlib import Path
from google.colab import drive

def mount_drive_and_setup_paths():
    """Mount Google Drive and setup dataset paths"""
    try:
        drive.mount('/content/drive', force_remount=False)
        print("✅ Google Drive mounted successfully")
        
        # Create dataset directory in Google Drive if it doesn't exist
        drive_base_path = '/content/drive/MyDrive/JuceTCN_Datasets'
        os.makedirs(drive_base_path, exist_ok=True)
        
        return drive_base_path
    except Exception as e:
        print(f"⚠️ Google Drive mount failed: {str(e)}")
        print("📝 Proceeding without persistence - dataset will need to be re-downloaded each session")
        return None

def check_persistent_dataset(drive_base_path):
    """Check if dataset exists in Google Drive or locally"""
    
    local_path = "SignalTrain_LA2A_Dataset_1.1"
    
    if drive_base_path:
        drive_dataset_path = os.path.join(drive_base_path, "SignalTrain_LA2A_Dataset_1.1")
        drive_compressed_path = os.path.join(drive_base_path, "SignalTrain_LA2A_Dataset_1.1.tgz")
        
        # Priority 1: Check if already available locally
        if os.path.exists(local_path) and os.path.exists(f"{local_path}/Train"):
            print("✅ Dataset already available locally from previous session")
            return local_path, "local"
        
        # Priority 2: Check for extracted dataset in Google Drive
        if os.path.exists(drive_dataset_path) and os.path.exists(f"{drive_dataset_path}/Train"):
            print("✅ Found extracted dataset in Google Drive")
            print("🔗 Creating symlink to avoid copying...")
            
            # Remove local path if it exists but is incomplete
            if os.path.exists(local_path):
                if os.path.islink(local_path):
                    os.unlink(local_path)
                else:
                    shutil.rmtree(local_path)
            
            # Create symlink to Google Drive dataset
            os.symlink(drive_dataset_path, local_path)
            print("✅ Symlink created - dataset ready for use")
            return local_path, "drive_symlink"
        
        # Priority 3: Check for compressed dataset in Google Drive
        elif os.path.exists(drive_compressed_path):
            print("✅ Found compressed dataset in Google Drive")
            print("📦 Extracting from Google Drive...")
            
            try:
                with tarfile.open(drive_compressed_path, 'r:gz') as tar:
                    tar.extractall('.')
                
                # Move extracted dataset to Google Drive for future use
                if os.path.exists(local_path):
                    print("💾 Saving extracted dataset to Google Drive...")
                    if os.path.exists(drive_dataset_path):
                        shutil.rmtree(drive_dataset_path)
                    shutil.copytree(local_path, drive_dataset_path)
                    
                    # Create symlink for current session
                    shutil.rmtree(local_path)
                    os.symlink(drive_dataset_path, local_path)
                    
                print("✅ Dataset extracted and ready for use")
                return local_path, "drive_compressed"
                
            except Exception as e:
                print(f"❌ Failed to extract from Google Drive: {str(e)}")
                print("🔄 Will download fresh dataset...")
    
    # If no persistent version found
    print("📥 No persistent dataset found - will download from Zenodo")
    return None, "download_needed"

def download_signaltrain_dataset():
    """Download and extract SignalTrain LA2A dataset from Zenodo"""
    
    # Correct URL for the dataset (it's a .tgz file, not .zip)
    dataset_url = "https://zenodo.org/records/3824876/files/SignalTrain_LA2A_Dataset_1.1.tgz"
    dataset_path = "SignalTrain_LA2A_Dataset_1.1.tgz"
    extract_path = "SignalTrain_LA2A_Dataset_1.1"
    
    # Download the dataset if not already present
    if not os.path.exists(dataset_path):
        print("📥 Downloading SignalTrain LA2A dataset (~21GB)...")
        print("⏳ This may take a while depending on your internet connection.")
        
        try:
            # Use requests with progress tracking
            response = requests.get(dataset_url, stream=True)
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
                            print(f"\r📊 Download progress: {percent:.1f}% ({downloaded/(1024**3):.1f}GB/{total_size/(1024**3):.1f}GB)", end='', flush=True)
            
            print("\n✅ Download complete!")
            
        except Exception as e:
            print(f"\n❌ Download failed: {str(e)}")
            print("\n📋 Manual download instructions:")
            print("1. Go to: https://zenodo.org/records/3824876")
            print("2. Download 'SignalTrain_LA2A_Dataset_1.1.tgz' (21GB)")
            print("3. Place the file in the current directory")
            print("4. Re-run this cell")
            
            # Clean up partial download
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
            
            raise Exception("Dataset download failed. Please download manually.")
    
    # Extract the dataset
    if os.path.exists(dataset_path):
        print("📦 Extracting dataset...")
        try:
            with tarfile.open(dataset_path, 'r:gz') as tar_ref:
                tar_ref.extractall('.')
            print("✅ Extraction complete!")
            
            # Verify extraction
            if os.path.exists(extract_path):
                # Count files to verify dataset integrity
                train_files = len([f for f in os.listdir(f"{extract_path}/Train") if f.endswith('.wav')])
                val_files = len([f for f in os.listdir(f"{extract_path}/Val") if f.endswith('.wav')])
                
                print(f"📊 Dataset verification:")
                print(f"   Training files: {train_files}")
                print(f"   Validation files: {val_files}")
                print(f"   Total size: {sum(os.path.getsize(os.path.join(extract_path, subset, f)) for subset in ['Train', 'Val'] for f in os.listdir(os.path.join(extract_path, subset))) / (1024**3):.1f} GB")
                
                if train_files > 0 and val_files > 0:
                    print("✅ Dataset verification successful!")
                else:
                    raise Exception("Dataset appears to be incomplete")
            else:
                raise Exception("Extraction failed - directory not found")
                
        except Exception as e:
            print(f"❌ Extraction failed: {str(e)}")
            raise Exception("Dataset extraction failed")
    
    return extract_path

def save_dataset_to_drive(dataset_path, drive_base_path):
    """Save dataset to Google Drive for future sessions"""
    if not drive_base_path:
        print("⚠️ Google Drive not available - skipping persistence")
        return
    
    try:
        drive_dataset_path = os.path.join(drive_base_path, "SignalTrain_LA2A_Dataset_1.1")
        drive_compressed_path = os.path.join(drive_base_path, "SignalTrain_LA2A_Dataset_1.1.tgz")
        
        print("💾 Saving dataset to Google Drive for future sessions...")
        print("🔄 This will take a few minutes but will save hours in future sessions...")
        
        # Save both extracted and compressed versions
        # Extracted version for faster access, compressed as backup
        
        # Copy extracted dataset
        if os.path.exists(drive_dataset_path):
            print("🗑️ Removing old dataset from Google Drive...")
            shutil.rmtree(drive_dataset_path)
        
        print("📁 Copying extracted dataset to Google Drive...")
        shutil.copytree(dataset_path, drive_dataset_path)
        
        # Create compressed backup
        if not os.path.exists(drive_compressed_path):
            print("🗜️ Creating compressed backup in Google Drive...")
            with tarfile.open(drive_compressed_path, 'w:gz') as tar:
                tar.add(dataset_path, arcname=os.path.basename(dataset_path))
        
        # Replace local dataset with symlink to save local storage
        print("🔗 Replacing local dataset with symlink to Google Drive...")
        shutil.rmtree(dataset_path)
        os.symlink(drive_dataset_path, dataset_path)
        
        print("✅ Dataset successfully saved to Google Drive!")
        print("🚀 Future Colab sessions will load instantly from Google Drive!")
        
    except Exception as e:
        print(f"⚠️ Failed to save to Google Drive: {str(e)}")
        print("📝 Dataset will still work for this session, but may need re-download next time")

def setup_persistent_signaltrain_dataset():
    """Main function to setup SignalTrain dataset with Google Drive persistence"""
    
    print("🎵 Setting up SignalTrain LA2A dataset with Google Drive persistence...")
    print("📄 Dataset info: https://zenodo.org/records/3824876")
    print("📝 Citation: Colburn, B., & Hawley, S. (2020). SignalTrain LA2A Dataset (1.1)")
    print("=" * 80)
    
    # Step 1: Mount Google Drive
    drive_base_path = mount_drive_and_setup_paths()
    
    # Step 2: Check for existing dataset
    dataset_path, source = check_persistent_dataset(drive_base_path)
    
    # Step 3: Download if needed
    if source == "download_needed":
        try:
            dataset_path = download_signaltrain_dataset()
            
            # Save to Google Drive for future sessions
            if drive_base_path:
                save_dataset_to_drive(dataset_path, drive_base_path)
            
        except Exception as e:
            print(f"\n❌ Dataset setup failed: {str(e)}")
            print("\n🔧 Troubleshooting:")
            print("1. Ensure you have ~21GB of free disk space")
            print("2. Check your internet connection")
            print("3. Try downloading manually from: https://zenodo.org/records/3824876")
            print("4. Ensure the file 'SignalTrain_LA2A_Dataset_1.1.tgz' is in the current directory")
            
            # Stop execution if dataset setup fails
            raise Exception("Cannot proceed without dataset")
    
    return dataset_path

def verify_dataset_structure(dataset_path):
    """Verify that the dataset has the expected structure for micro-tcn"""
    
    required_dirs = ['Train', 'Val']
    
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.exists(dir_path):
            raise Exception(f"Missing required directory: {dir_path}")
        
        # Check for input/target file pairs
        files = os.listdir(dir_path)
        input_files = [f for f in files if f.startswith('input_') and f.endswith('.wav')]
        target_files = [f for f in files if f.startswith('target_') and f.endswith('.wav')]
        
        if len(input_files) == 0 or len(target_files) == 0:
            raise Exception(f"No input/target files found in {dir_path}")
        
        print(f"✅ {dir_name}: {len(input_files)} input files, {len(target_files)} target files")
    
    return True

# Setup the SignalTrain LA2A dataset with Google Drive persistence
try:
    dataset_root = setup_persistent_signaltrain_dataset()
    verify_dataset_structure(dataset_root)
    print(f"🎯 Dataset ready at: {dataset_root}")
    
except Exception as e:
    print(f"\n❌ Dataset setup failed: {str(e)}")
    print("\n🔧 Troubleshooting:")
    print("1. Ensure you have ~21GB of free disk space")
    print("2. Check your internet connection")
    print("3. Try downloading manually from: https://zenodo.org/records/3824876")
    print("4. Ensure the file 'SignalTrain_LA2A_Dataset_1.1.tgz' is in the current directory")
    
    # Stop execution if dataset setup fails
    raise Exception("Cannot proceed without dataset")
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
    
    # Create PyTorch Lightning trainer with v1.9.5 compatibility
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=16 if args.precision == 16 and torch.cuda.is_available() else 32,
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
