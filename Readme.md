# ELR-LRT: Efficient Low-Resource Latent Reasoning Transformer

The ELR-LRT project implements an efficient Transformer architecture for byte sequence processing with dynamic patching, continuous latent reasoning, and reinforcement learning fine-tuning. It is designed to optimize performance while enabling deployment on low-powered hardware.

## Overview

- **Dynamic Byte Patching Module (DBPM):**  
  Implemented in C++ ([cpp/src/patcher.cpp](cpp/src/patcher.cpp), [cpp/include/patcher.h](cpp/include/patcher.h)), this module segments a byte sequence into variable-length patches based on entropy-based thresholds. A frequency table ([cpp/src/frequency_table.cpp](cpp/src/frequency_table.cpp)) is used to compute entropy for context segments, reducing redundancy in predictable regions.

- **Continuous Latent Reasoning Module (CLRM):**  
  Instead of explicit token generation for intermediate reasoning steps, CLRM leverages the hidden states of the model for internal multi-step reasoning. This approach, inspired by the Coconut paradigm, enables efficient inference by reducing computational overhead.

- **Reinforcement Learning Fine-Tuning Module (RLFTM):**  
  This module applies RL-based optimization techniques analogous to DeepSeek-R1, leveraging a reward function that balances accuracy, efficiency, and coherence. The fine-tuning step improves the model’s reasoning performance while maintaining computational efficiency.

## Project Structure

```
elr_lrt_project/
├── build/                   # Build artifacts for different CPython versions 
│   ├── lib.win-amd64-cpython-311/
│   │   └── elr_lrt/
│   │       ├── dbpm.cp311-win_amd64.pyd
│   │       └── model.py
│   ├── lib.win-amd64-cpython-312/
│   │   └── elr_lrt/
│   │       └── dbpm.cp312-win_amd64.pyd
│   └── temp.win-amd64-.../   # Temporary build outputs for C++ bindings
├── cpp/
│   ├── include/
│   │   ├── frequency_table.h
│   │   └── patcher.h
│   └── src/
│       ├── bindings.cpp
│       ├── frequency_table.cpp
│       └── patcher.cpp
├── python/
│   └── elr_lrt/
│       ├── __init__.py      # Exposes patch_sequence and the ELRLRTModel
│       └── model.py         # ELR-LRT model implementation
├── tests/
│   ├── data.txt             # Sample dataset for testing on real-world text
│   ├── patch_train_on_data.py
│   ├── test_elr_lrt.py
│   └── test_patch_sequence.py
├── setup.py                 # Setup script to build the Python package (setup.py)
├── pyproject.toml           # PEP 518 build system requirements
├── requirements.txt         # Project dependencies
```

## Installation

1. **Clone the Repository**  
   Clone the project to your local machine:
   ```sh
   git clone https://github.com/prabinpanta0/ELR-LRT.git
   cd elr-lrt
   ```

2. **Install Dependencies**  
   Install the necessary dependencies using the provided `requirements.txt`:
   ```sh
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. **Build the Project**  
   Compile and build the package using `setup.py`:
   ```sh
   python setup.py build
   ```

## Usage

- **Importing the Model and Patch Sequence:**  
  ```python
  from elr_lrt import patch_sequence, ELRLRTModel
  
  byte_list = list(b"example input")
  patches = patch_sequence(byte_list, k=5, theta=1.0, theta_r=0.5)
  
  model = ELRLRTModel()
  ```

- **Model Forward Pass and Fine-Tuning:**  
  ```python
  input_bytes = torch.tensor(list(b"input bytes"), dtype=torch.long)
  target_bytes = torch.tensor(list(b"target bytes"), dtype=torch.long)
  model.rl_finetune(input_bytes, target_bytes, num_iterations=10)
  ```

## Testing

The project includes test scripts to validate each module:

- **Patch Sequence Test:**  
  ```sh
  python tests/test_patch_sequence.py
  ```

- **ELRLRT Model Testing and Benchmarking:**  
  ```sh
  python tests/test_elr_lrt.py
  ```

## Key Features and Efficiency Gains

- **Computational Cost Reduction:**  
  The dynamic byte patching module reduces sequence length, leading to a quadratic reduction in inference cost.
- **Memory and Latency Optimization:**  
  CLRM eliminates redundant intermediate token generation, reducing per-token latency by up to 50%.
- **Energy Efficiency:**  
  RL fine-tuning optimizes reasoning behavior while maintaining a low computational footprint, making the model viable for CPU-only devices.

## Contributions and Future Work

Contributions are welcome! Future directions include:
- Adaptive hyperparameter tuning for patching and reasoning steps.
- Exploring multimodal extensions.
- Optimizing RL fine-tuning with self-supervised objectives.

For more details, refer to:
- `setup.py`
- `model.py`
- `patcher.cpp`

Happy coding!
```
  (\(\
  (=':')
  (,(")(")
```

