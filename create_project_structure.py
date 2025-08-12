import os


# Define the directory structure
dirs = [
    f"data/raw",
    f"data/processed",
    f"notebooks",
    f"src/data",
    f"src/models",
    f"src/pipeline",
    f"src/app",
    f"tests",
    f".github/workflows"
]

# Create directories
for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created: {dir_path}")

# Create empty files
files = {
    f"README.md": "# Uplift Predictor ML Project\n",
    f"requirements.txt": "",
    f"Dockerfile": "",
    f"setup.py": "",
    f"src/app/main.py": "",
    f"src/app/schema.py": "",
    f"src/data/load_data.py": "",
    f"src/data/preprocess.py": "",
    f"src/models/train_model.py": "",
    f"src/models/uplift_models.py": "",
    f"src/models/evaluate.py": "",
    f"src/pipeline/main.py": "",
    f"tests/test_data.py": "",
    f"tests/test_model.py": "",
    f"tests/test_api.py": "",
    f".github/workflows/ci.yml": ""
}

for file_path, content in files.items():
    with open(file_path, "w") as f:
        f.write(content)
    print(f"Created: {file_path}")

print("\n✅ Project structure created successfully.")
