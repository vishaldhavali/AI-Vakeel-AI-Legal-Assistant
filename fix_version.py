import subprocess
import sys

def fix_versions():
    """Fix the version conflict with langchain packages"""
    print("Uninstalling incompatible packages...")
    packages_to_uninstall = [
        "langchain-google-genai",
        "langchain-core",
        "langchain-community",
        "langchain"
    ]
    
    for package in packages_to_uninstall:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
            print(f"Successfully uninstalled {package}")
        except Exception as e:
            print(f"Error uninstalling {package}: {e}")
    
    print("\nInstalling compatible versions...")
    packages_to_install = [
        "langchain==0.0.335",
        "langchain-core==0.1.7",
        "langchain-google-genai==0.0.6",
        "langchain-community==0.0.9"
    ]
    
    for package in packages_to_install:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except Exception as e:
            print(f"Error installing {package}: {e}")
    
    print("\nDone fixing versions!")

if __name__ == "__main__":
    fix_versions() 