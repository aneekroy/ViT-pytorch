import torch
import sys

def test_torch_setup():
    print("\n=== PyTorch Configuration ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS (Metal) available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
    
    # Additional helpful information
    print("\n=== System Details ===")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    if torch.backends.mps.is_available():
        print("Metal Performance Shaders (MPS) is available")
        print("This machine supports GPU acceleration via Apple Metal")
    else:
        print("Metal Performance Shaders (MPS) is not available")
    
    print("\n=== Default Tensor Type ===")
    print(f"Default tensor type: {torch.get_default_dtype()}")

if __name__ == "__main__":
    test_torch_setup() 
