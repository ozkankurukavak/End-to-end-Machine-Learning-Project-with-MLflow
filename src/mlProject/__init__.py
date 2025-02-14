import os
import sys
import logging

# Log formatını tanımlıyoruz.
loggin_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Log dosyasının bulunduğu dizini belirliyoruz.
log_dir = "logs"

# Dosya yolunu oluşturuyoruz. logs/running_logs.log
log_filepath = os.path.join(log_dir, "running_logs.log")

# logs klasörü yoksa oluşturulacak.
os.makedirs(log_dir, exist_ok=True)

# logging modülünü yapılandırıyoruz:
logging.basicConfig(
    level=logging.INFO,  # INFO seviyesinde log alacak şekilde yapılandırma
    format=loggin_str,  # Tanımladığımız log formatını kullanıyoruz.
    handlers=[
        logging.FileHandler(log_filepath),  # Logları dosyaya yazdıracak handler
        logging.StreamHandler(sys.stdout)   # Logları terminale yazdıracak handler
    ]
)

# logger nesnesi oluşturuyoruz.
logger = logging.getLogger("mlProjectLogger")  # Özel bir logger oluşturuyoruz.