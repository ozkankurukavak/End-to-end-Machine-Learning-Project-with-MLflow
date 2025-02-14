import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')  # loglama seviyesini INFO olarak ayarlar ve her log mesajını tarih ve mesaj şeklinde formatlar.

project_name = 'mlProject'

list_of_files = [
    ".github/workflows/.gitkeep", # bu dosyanın amacı sadece boş klasörlerin Git tarafından izlenmesini sağlamaktır.
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entitiy/__init__.py",
    f"src/{project_name}/entitiy/config_entitiy.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "tamplates/index.html",
    "test.py"

]


for filepath in list_of_files:
    filepath = Path(filepath)  # Dosya yolunu Path objesine dönüştür
    filedir, filename = os.path.split(filepath)  # Klasör yolunu ve dosya adını ayır

    if filedir != "":  # Eğer dosya yolu boş değilse
        os.makedirs(filedir, exist_ok=True)  # Klasörü oluştur (zaten varsa hata vermez)
        logging.info(f"{filedir} dizini, {filename} dosyası için oluşturuluyor.")  # Log mesajı

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
    # Dosya mevcut değilse ya da boyutu sıfırsa, boş bir dosya oluşturuyoruz.
        with open(filepath, "w") as f:
            pass
            # Burada, dosya oluşturulurken bir log mesajı yazıyoruz.
            logging.info(f"boş dosya oluşturuluyor: {filepath}")
    else:
        logging.info(f"{filename} zaten mevcut")