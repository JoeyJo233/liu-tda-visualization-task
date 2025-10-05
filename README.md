# Solution of Scientific Visualization Tasks 

Run this project in a reproducible Conda environment using the provided `env.yml`.

---

## 🚀 Quick Setup

1. **Install Miniconda** (if not already installed):  
   https://docs.conda.io/en/latest/miniconda.html

2. **Create the environment from `env.yml`**:
   ```bash
   conda env create -f env.yml --name your_custom_name
   ```
   Note: The `--name` parameter is optional. If omitted, the environment name will default to `yizho847_SciVis` as predefined in `env.yml`.

3. **Activate the environment**:
   ```bash
   conda activate your_custom_name
   ```

4. **Verify dependencies** (optional):
   ```bash
   python -c "import pandas as pd; import numpy as np; import pyvista as pv; import scipy as sp; import matplotlib as mp; print('✅ All packages loaded!')"
   ```

5. **Run your code**:
   ```bash
   python your_script.py
   ```
   Alternatively, if using VSCode: select the interpreter created above in the "Select Interpreter" settings, then use the GUI run button to execute the code.

   **Note for Windows users**: In testing, it was found that when using the command line on Windows, it's best to navigate to the corresponding task folder before running Python commands. Otherwise, you may encounter issues finding `2d_scalar_field.csv`. To avoid this problem, a copy of `2d_scalar_field.csv` has been placed in the root directory as well.

---

## Note
If you have any questions, please contact: yizho847@student.liu.se