import os
import sys
import subprocess

def install_faiss():
    """Install FAISS properly"""
    print("Attempting to install FAISS...")
    
    # First, try with pip
    try:
        # Install the wheel directly
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "https://download.pytorch.org/whl/cpu/torch-2.2.0%2Bcpu-cp312-cp312-win_amd64.whl"
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "faiss-cpu==1.7.4", 
            "--no-deps", "--force-reinstall"
        ])
        print("FAISS installed successfully with pip!")
        return True
    except Exception as e:
        print(f"Pip installation failed: {e}")
    
    # If pip fails, suggest conda
    print("\nPip installation failed. Would you like to try installing with conda? (y/n)")
    choice = input().lower()
    if choice == 'y':
        try:
            subprocess.check_call(["conda", "install", "-c", "conda-forge", "faiss-cpu", "-y"])
            print("FAISS installed successfully with conda!")
            return True
        except Exception as e:
            print(f"Conda installation failed: {e}")
    
    print("\nFailed to install FAISS. You might need to install it manually.")
    return False

if __name__ == "__main__":
    install_faiss() 