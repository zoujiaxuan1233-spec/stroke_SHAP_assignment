It is recommended to use Python 3.12 and create an independent environment in this folder. The following commands should be executed in the `submission` folder:

```bash
python -m venv .venv
```

MacOS or Linux activation:

```bash
source .venv/bin/activate
```

Windows Activation:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python code/run_assignment.py
python code/verify_outputs.py
```

Run only Task 1:

```bash
python code/00_data_exploration.py
```

The Notebook has saved the running results. Just open it to view. If you want to re-run in the existing Jupyter environment, use the Python kernel that installs the dependencies of this project; additional kernel dependencies are listed in `requirements-notebook.txt`. The Notebook defaults to reading the saved results. Changing the first piece of code's `REGENERATE` to `True` allows for re-training. The running times and numerical details may vary slightly across different platforms.

The training script re-generates data, models, and charts; Word and PPT are the final products compiled based on the verified experimental results of this session, and will not be automatically overwritten by the training script.

## Result Directory

- `outputs/metrics.json`: validation and test metrics, as well as supplementary results with threshold 0.5.
- `outputs/run_details.json`: version, parameters, seed, case, SHAP validation results, and training prompts.
- `outputs/tables/`: classification attribution, individual test prediction, threshold selection, local contribution, and global ranking.
- `outputs/explanations/shap_values.npz`: complete SHAP array, background and explanation sample indices, feature order, and selected case positions.
- `outputs/explanations/category_codes.json`: reversible category mapping used by Kernel SHAP. This encoding is only for interpreter transmission; the model still uses one-hot encoding.
- `outputs/models/`: two fully fitted prediction processes; re-training can regenerate.
- `outputs/figures/`: force, summary, waterfall, confusion matrix, and performance curve.
- `outputs/execution.log`, `outputs/verification.log`: actual execution and independent verification records.
- `environment-lock.txt`: the complete dependency versions of the actual environment; `requirements.txt` only lists the main running dependencies.
- `original/`: sample code provided by the teacher and screenshots of requirements for reference and verification.
