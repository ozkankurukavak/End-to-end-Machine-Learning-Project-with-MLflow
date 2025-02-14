import os
from box.exceptions import BoxValueError
import yaml  # type: ignore
import json
import joblib # type: ignore
from ensure import ensure_annotations
from box import Box
from pathlib import Path
from typing import Any
import base64
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from mlProject import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> Box:
    """YAML dosyasını okur ve döner.

    Args:
        path_to_yaml (str): girdi olarak dosya yolu

    Raises:
        ValueError: Eğer YAML dosyası boşsa
        e: Boş dosya hatası

    Returns:
        ConfigBox: ConfigBox tipi
    """
    try:
        # YAML dosyasını açar ve içeriği okur
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)  # YAML içeriği yüklenir
            return Box(content)  # İçeriği Box formatında döner
    except BoxValueError:
        # YAML dosyası boş ise hata fırlatılır
        raise ValueError("yaml dosyası boş")
    except Exception as e:
        # Diğer hataları yakalar ve tekrar fırlatır
        raise e



@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Dizinlerin bir listesini oluşturur.

    Args:
        path_to_directories (list): Dizimlerin yol listesini
        ignore_log (bool, optional): Birden fazla dizin oluşturulacaksa görmezden gel. Varsayılan olarak False.
    """
    for path in path_to_directories:
        # Dizinlerin her biri için 'makedirs' komutu çalıştırılır
        os.makedirs(path, exist_ok=True)  # Dizin varsa hata vermez, varsa oluşturur
        if verbose:
            print(f"Dizin oluşturuldu: {path}")  # verbose=True ise oluşturulan dizin adı yazdırılır




@ensure_annotations
def save_json(path: Path, data: dict):
    # Dosya yolunu açmak için 'path' parametresi kullanılır ve 'data' adlı sözlük yazılır
    with open(path, "w") as f:
        # 'json.dump' fonksiyonu ile Python dictionary'si 'data', JSON formatında dosyaya yazılır
        json.dump(data, f, indent=4)
    
    # Dosya başarıyla kaydedildikten sonra log bilgisi yazılır
    logger.info(f"json dosyası kaydedildi: {path}")




@ensure_annotations
def load_json(path: Path) -> Box:
    """JSON dosyasını okur ve içeriğini bir ConfigBox nesnesine döner.

    Args:
        path (Path): JSON dosyasının yolu

    Returns:
        ConfigBox: JSON dosyasının içeriği, ConfigBox türünde döndürülür.
    """
    with open(path) as f:
        content = json.load(f)  # JSON dosyasını okuyup 'content' değişkenine yüklüyoruz
        
        logger.info(f"JSON dosyası başarıyla yüklendi: {path}")
        return Box(content)  # 'content' verisini Box (ConfigBox) nesnesine dönüştürerek döndürüyoruz.



@ensure_annotations
def get_size(path: Path) -> str:
    """KB cinsinden boyut alır.

    Args:
        path (Path): Dosyanın yolu

    Returns:
        str: Boyut KB cinsinden
    """
    # Dosyanın boyutunu alır ve KB'ye dönüştürür
    size_in_kb = round(os.path.getsize(path)/1024)  # Dosya boyutu byte cinsinden alınır, KB'ye dönüştürülür
    return f"~ {size_in_kb} KB"  # Sonuç string formatında döner


@ensure_annotations
def save_object(file_path: Path, obj):
    """Python nesnesini dosyaya kaydeder.

    Args:
        file_path (Path): Dosyanın yolunu belirtir
        obj (Any): Kaydedilecek nesne

    Raises:
        e: Herhangi bir hata durumunda hata fırlatılır
    """
    try:
        # Dosyanın bulunduğu dizini kontrol eder ve oluşturur
        dir_path = os.path.dirname(file_path)  # Dosya yolunun dizin kısmını alır
        os.makedirs(dir_path, exist_ok=True)  # Dizin yoksa oluşturur
        
        # Nesneyi binary (ikili) formatta dosyaya kaydeder
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)  # obj'yi dosyaya pickle ile kaydeder
        
    except Exception as e:
        # Hata oluşursa, bu hatayı tekrar fırlatır
        raise e
    


@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Veriyi binary formatında kaydeder.
    
    Args:
        data (Any): Kaydedilecek veri. Bu, her türlü Python veri yapısı olabilir (örneğin, model, numpy array, vb.).
        path (Path): Verinin kaydedileceği dosya yolunu belirtir. Bu, bir 'Path' nesnesi olmalıdır.

    Returns:
        None: Bu fonksiyon herhangi bir şey döndürmez, yalnızca veriyi kaydeder.
        
    İşleyiş:
        - Veriyi 'joblib' kullanarak belirtilen dosya yoluna binary formatta kaydeder.
        - Dosya kaydedildikten sonra, işlemin başarıyla tamamlandığına dair log kaydı oluşturulur.
    """
    
    try:
        # Veriyi belirtilen dosya yoluna kaydediyoruz.
        joblib.dump(value=data, filename=path)
        
        # Dosya başarıyla kaydedildiğinde bir log mesajı oluşturuluyor.
        logger.info(f"Binary dosya şu konumda kaydedildi: {path}")
    
    except Exception as e:
        # Hata oluşursa, hatayı logluyoruz.
        logger.error(f"Binary dosya kaydedilirken hata oluştu: {e}")
        raise e




@ensure_annotations
def load_bin(path: Path) -> any:
    """Binary formatındaki (joblib) bir dosyayı yükler.

    Args:
        path (Path): Binary dosyanın yolu.

    Returns:
        Any: Yüklenen dosyanın içeriği.
    """
    data = joblib.load(path)  # Binary dosyayı yükle
    logger.info(f"Binary dosya yüklendi: {path}")
    return data



def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report = {}
        for i in range(0,len(list(models))): # 0-7 döngü başlatır

            model = list(models.values())[i]  # i == 0 iken RandomForestRegressor() çalışır.
            para = param[list(models.keys())[i]]
            rc = RandomizedSearchCV(model,para,cv=3)

            rc.fit(X_train, y_train) # Bu search algoritmasını Train datalar üzerinden çalıştırdım
            model.set_params(**rc.best_params_) # Yukarıda çalışan search algoritmasının bulduğu en iyi parametreleri aldım
            model.fit(X_train, y_train) # En iyi parametrelerle eğitim yaptık

            y_test_pred = model.predict(X_test) # Tahmin değerlerini aldık

            test_model_score = r2_score(y_test,y_test_pred)


            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise e->ConfigBox