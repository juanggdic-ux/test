# Upload this Lab 3 project to GitHub

## Files and folder placement

Extract `RAVDESS_Lab3_GitHub_Ready.zip` onto your computer. The extracted `RAVDESS_Lab3` folder contains the complete project. When uploading to a **new repository**, open `RAVDESS_Lab3` and upload **its contents**, keeping these entries in the root of the repository:

- `.gitignore`, `README.md`, `requirements.txt`, `run_pipeline.py`, `benchmark.py`
- `data/`, `src/`, `notebooks/`, `results/`, `tests/`, `docs/`
- `GITHUB_UPLOAD_GUIDE.md`

Do **not** upload this compressed ZIP as the only item: GitHub will store it as a file, not turn it into the described directories.

## Using the GitHub website

1. Log into GitHub, select **New repository** and choose a name (e.g. `RAVDESS-Lab3`). For this package, the easiest option is to **not** initialize the new repository with a README or `.gitignore` (both are included in the project).
2. Create the repository. Choose **uploading an existing file** in the empty repository, or later **Add file → Upload files**.
3. Extract the ZIP on your computer. Open the extracted `RAVDESS_Lab3` folder and drag its **contents (files and subfolders together)** onto the GitHub upload area. Verify that `src/analysis/features.py` is nested correctly.
4. Add a commit message, e.g. `Add modular Lab 3 project`, and commit the upload.
5. Open `README.md`, `notebooks/Lab3_report.ipynb`, and the three `src/*/complexity_analysis.ipynb` files on GitHub to verify the layout.

If the browser has trouble with directory uploads, use **GitHub Desktop** or the command-line method below, which reliably preserves nested folders.

## Using Git (recommended for future updates)

Open a terminal in the **extracted `RAVDESS_Lab3` folder**, then run:

```bash
git init
git add .
git commit -m "Lab 3: modular RAVDESS pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/RAVDESS-Lab3.git
git push -u origin main
```

Replace `YOUR_USERNAME` and the repository name with your own GitHub values. Sign in through the Git credential prompt as needed. To commit future changes, run `git add .`, `git commit -m "Description of changes"`, and `git push`.

## Reproducing locally

Run these commands **from the repository root** after installing Python 3.11+: 

```bash
pip install -r requirements.txt
python run_pipeline.py
python benchmark.py
python -m pytest -q
```

The default dataset is synthetic for software validation. To use the real RAVDESS recordings, see `README.md` and `data/raw/README.md`.

## What not to upload

Do not commit the original large RAVDESS archive, secrets, `.venv/`, `__pycache__/`, `.pytest_cache/`, or `.ipynb_checkpoints/`. The `.gitignore` is configured for these patterns. Review personal or sensitive material before pushing files to a public repository.
