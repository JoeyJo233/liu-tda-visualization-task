# Project Name

Run this project in a reproducible Conda environment using the provided `environment.yml`.

---

## 🚀 Quick Setup

1. **Install Miniconda** (if not already installed):  
   https://docs.conda.io/en/latest/miniconda.html

2. **Create the environment from `env.yml`**:
   ```bash
   conda env create -f env.yml --name your_custom_name
   ```

3. **Activate the environment**:
   ```bash
   conda activate your_custom_name
   ```

4. **Verify dependencies** (optional):
   ```bash
   python -c "import pandas as pd; import numpy as np; import pyvista as pv; print('✅ All packages loaded!')"
   ```

5. **Run your code**:
   ```bash
   python your_script.py
   ```

---

## 💡 Notes

- The environment name is defined in `environment.yml` (typically `myenv` or similar).  
- To update the environment after changes to `environment.yml`:
  ```bash
  conda env update -f environment.yml --prune
  ```
- To exit the environment:
  ```bash
  conda deactivate
  ```

> Compatible with Windows, macOS, and Linux.

conda env create -f environment.yml

```bash
python PyV.py 1
```