from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="models",
    path_in_repo="models",
    repo_id="Amanatou444/sensante",
    repo_type="space",
    ignore_patterns=[]
)
print("Upload terminé !")