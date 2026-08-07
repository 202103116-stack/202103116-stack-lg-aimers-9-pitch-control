"""model/ 안의 학습된 아티팩트 + submission/script.py + requirements.txt를
Dacon 제출 규격(model/, script.py, requirements.txt가 zip 최상위에 바로
있어야 함)에 맞춰 submission/submit.zip으로 패키징한다.

사용법:
    python src/package_submission.py exp001_lgbm

- submission/submit.zip: 항상 최신 빌드 (Dacon에 이 파일을 업로드)
- submission/archive/<label>_submit.zip: 실험별 이력 보관용 사본
"""

import argparse
import os
import shutil
import zipfile

MODEL_DIR = "./model"
SUBMISSION_DIR = "./submission"


def build(label):
    sub_model_dir = os.path.join(SUBMISSION_DIR, "model")
    os.makedirs(sub_model_dir, exist_ok=True)
    for fname in os.listdir(MODEL_DIR):
        shutil.copy2(os.path.join(MODEL_DIR, fname), os.path.join(sub_model_dir, fname))

    zip_path = os.path.join(SUBMISSION_DIR, "submit.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    entries = {
        "script.py": os.path.join(SUBMISSION_DIR, "script.py"),
        "requirements.txt": os.path.join(SUBMISSION_DIR, "requirements.txt"),
    }
    for fname in os.listdir(sub_model_dir):
        if fname == ".gitkeep":
            continue
        entries[f"model/{fname}"] = os.path.join(sub_model_dir, fname)

    # zipfile은 arcname을 준 그대로 저장한다 (OS와 무관하게 '/' 를 직접
    # 넣어주면 Linux 평가 서버에서도 정상적으로 model/ 하위 폴더로 풀림).
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in entries.items():
            zf.write(path, arcname)

    archive_dir = os.path.join(SUBMISSION_DIR, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    archive_path = os.path.join(archive_dir, f"{label}_submit.zip")
    shutil.copy2(zip_path, archive_path)

    print(f"Built:    {zip_path}  (Dacon에 이 파일을 업로드)")
    print(f"Archived: {archive_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="예: exp001_lgbm")
    args = parser.parse_args()
    build(args.label)
