from pathlib import Path
import kaggle

project_root = Path(__file__).parent.parent
data_raw_dir = project_root / "data" / "raw"

data_raw_dir.mkdir(parents=True, exist_ok=True)

print("Downloading phishing email dataset from Kaggle...")
kaggle.api.dataset_download_files(
    'naserabdullahalam/phishing-email-dataset',
    path=data_raw_dir,
    unzip=True
)

for file in data_raw_dir.glob("*.csv"):
    if file.name != "phishing_email.csv":
        file.unlink()
        print(f"Removed unnecessary file: {file.name}")

print(f"Dataset downloaded successfully to {data_raw_dir / 'phishing_email.csv'}")
