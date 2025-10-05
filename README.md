# 🧭 Scientific Visualization — Task Solutions

This repository provides solutions for several **Scientific Visualization** tasks.  
To ensure reproducibility, all scripts are designed to run within a **Conda environment** defined in `env.yml`.

---

## ⚙️ Environment Setup

### 1. Install Miniconda
If you don’t already have Conda installed, follow the official Miniconda installation guide:  
🔗 [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

### 2. Create the Conda Environment
Use the provided environment file to create a reproducible setup:
```bash
conda env create -f env.yml --name your_custom_name
```
> 💡 *Tip:* The `--name` argument is optional.  
> If omitted, the environment will use the default name **`yizho847_SciVis`** specified in `env.yml`.

### 3. Activate the Environment
```bash
conda activate your_custom_name
```

### 4. Verify Installed Packages (optional)
Run a quick check to confirm all dependencies load correctly:
```bash
python -c "import pandas as pd; import numpy as np; import pyvista as pv; import scipy as sp; import matplotlib as mp; print('✅ All packages loaded successfully!')"
```

### 5. Run the Code
You can now execute any of the provided scripts:
```bash
python your_script.py
```
Or, if using **VSCode**, simply:
- Select the Conda environment you just created under **“Select Interpreter”**.
- Use the **Run** button (▶️) to execute the code directly.

---

## 🧩 Task-Specific Instructions

### Task 3: Custom Isovalue Extraction
You can specify a custom **isovalue** for contour or surface extraction directly from the command line:
```bash
python isocontour_2D.py 1.0
```
or
```bash
python isocontour_3D.py 1.2
```
If no parameter is provided, the script will default to **isovalue = 0.0**.

---

## 💡 Tips for Command-Line Users (Windows)

When running scripts from the command line on Windows, it’s best to **navigate into the corresponding task directory** before execution.  
Otherwise, the script might not locate the dataset file `2d_scalar_field.csv`.

To avoid path issues, a backup copy of `2d_scalar_field.csv` is also placed in the project **root directory**.

---

## 📬 Contact

For any questions or issues, feel free to reach out:  
**Email:** [yizho847@student.liu.se](mailto:yizho847@student.liu.se)
