# YOLOv11 + SAM2.1b+ Pest Detector

This project combines YOLOv11 and SAM2.1b+ models for advanced pest detection in agricultural environments.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-repo-url.git
cd YOLOv11_SAM21bplus_pestDetector
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install PyTorch with CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. Install requirements:
```bash
pip install -r requirements.txt
```

5. Verify numpy version:
```bash
pip install numpy==2.0.2
```

## Requirements

- Python 3.8+
- CUDA 12.1 (required for PyTorch)
- Numpy 2.0.2 (exact version required)

## Troubleshooting

If you encounter issues:

1. Deactivate and reactivate your virtual environment:
```bash
deactivate
# Then reactivate using the appropriate command for your OS
```

2. Verify numpy version:
```bash
pip show numpy
```
Ensure the version is exactly 2.0.2. If not, run:
```bash
pip install numpy==2.0.2
```

3. Confirm CUDA version:
```bash
nvcc --version
```
Ensure CUDA 12.1 is installed.

## Important Notes

- This project requires CUDA 12.1 specifically - other versions will not work
- Numpy must be version 2.0.2 - other versions may cause compatibility issues
- Always work within the virtual environment to avoid dependency conflicts
