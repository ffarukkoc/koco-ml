# koco

koco, tablo verileri üzerinde regresyon (regression) ve ikili sınıflandırma (binary classification) için bir Python araç setidir. Modelin
tüm yaşam döngüsünü (life cycle) kapsar: **Analiz (analysis) → Ön işleme (preprocessing) → Değişken seçimi (feature selection) → Modelleme (modelling) → Skorlama (scoring)**.

Ön işleme (preprocessing) adımları yeniden kullanılabilir *planlar* olarak saklanır: bir plan eğitim verisi (training data) üzerinde
bir kez fit edilir ve test verisine (test data) ve yeni verilere değiştirilmeden uygulanır. Eğitilmiş bir model ve
planları tek bir dosyada birlikte kaydedilebilir ve ham veriyi skorlamak (scoring) için kullanılabilir.

📘 **İş akışı (workflow), uçtan uca örnekler (end-to-end example) ve fonksiyon referansı (function reference): [GuideTR.md](https://github.com/ffarukkoc/koco-ml/blob/main/GuideTR.md)**

## Gereksinimler

- Python 3.10 veya üzeri
- pandas, numpy, matplotlib, openpyxl, scikit-learn, scipy ve joblib (otomatik olarak kurulur)
- Opsiyonel (optional) özellikler ek paketler gerektirir (bkz. [Opsiyonel bağımlılıklar (optional dependency)](#opsiyonel-bağımlılıklar-optional-dependency))

## Kurulum

> **Not:** koco, PyPI'da **`koco-ml`** adıyla yayımlanır ve `koco` olarak import edilir. PyPI'daki `koco` adlı
> proje ilgisiz bir pakettir; onu kurmayın.

> **macOS için not:** aşağıdaki komutlar `python` kullanır. macOS'ta `python` genellikle sanal ortam (virtual environment)
> dışında kullanılamaz; bunun yerine `python3` kullanın.

Sanal ortam (virtual environment), koco'yu ve bağımlılıklarını (dependency) diğer projelerden ayrı tutar:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### PyPI'dan

```bash
python -m pip install koco-ml             # temel özellikler
python -m pip install "koco-ml[all]"      # temel özellikler + tüm opsiyonel bağımlılıklar (optional dependency)
```

### Kaynak klasörden (source folder) kurulum

Komutları `pyproject.toml` dosyasını içeren klasörden çalıştırın:

```bash
python -m pip install ".[all]"         # standart kurulum (standard install): kod ortama kopyalanır
python -m pip install -e ".[all]"      # düzenlenebilir kurulum (editable install): ortam bu klasördeki kodu kullanır
```

- **Standart kurulum (standard install)** klasörden bağımsız çalışır; klasör sonrasında taşınabilir veya silinebilir.
- **Düzenlenebilir kurulum (editable install)** klasördeki her değişikliği anında yansıtır. Geliştirme ve `git pull`
  güncellemeleri için uygundur. Klasör yerinde kalmalıdır (bkz. [Sorun giderme](#sorun-giderme)).
- pip bağımlılıkları (dependency) kurar; bu internet erişimi gerektirir. Tüm bağımlılıkların (dependency) zaten kurulu olduğu
  çevrimdışı bir makinede `--no-deps --no-build-isolation` ekleyin.
- `koco/` klasörünü elle `site-packages` içine kopyalamak desteklenmez: hiçbir bağımlılık (dependency) kurulmaz ve
  pip ile güncellenemez ya da kaldırılamaz.

### Opsiyonel bağımlılıklar (optional dependency)

Her ek paket yalnızca ona ihtiyaç duyan özellikler tarafından import edilir.

| Ek | Paketler | Kullanan |
|---|---|---|
| `sql` | SQLAlchemy, pyodbc | `Connection` |
| `parquet` | pyarrow | `ExportParquet`, `ImportParquet` |
| `stats` | statsmodels | `VIFReport`, `BackwardSelection` |
| `boosting` | xgboost, lightgbm, catboost | `XGB*`, `LGBM*`, `CatBoost*` algoritma anahtarları (algorithm key) |
| `dev` | pytest | test paketi |
| `all` | sql + parquet + stats + boosting | `dev` hariç her şey |

`Connection` ayrıca *ODBC Driver 17 for SQL Server* gerektirir. Bağlantı ayarları
[GuideTR.md](https://github.com/ffarukkoc/koco-ml/blob/main/GuideTR.md#sql-server-yapılandırması) içinde açıklanmıştır. macOS / Linux'ta ayrıca
[ODBC ve Kerberos kurulumu](https://github.com/ffarukkoc/koco-ml/blob/main/GuideTR.md#macos--linux-odbc-ve-kerberos-kurulumu) adımlarını izleyin (sürücü kaydı ve `krb5.conf`).

### Kurulumu kontrol etme

```bash
python -c "import koco; print(koco.__version__, koco.__file__)"
```

Yazdırılan yol hangi kopyanın kullanıldığını gösterir: standart kurulumda (standard install) `site-packages`,
düzenlenebilir kurulumda (editable install) kaynak klasör (source folder). Bir notebook'ta kernel aynı yorumlayıcıyı (interpreter) kullanmalıdır;
bunu `import sys; print(sys.executable)` ile kontrol edin ve kurulumdan sonra kernel'i yeniden başlatın.

### Güncelleme

- PyPI'dan: `python -m pip install --upgrade koco-ml`
- Klasörden standart kurulum (standard install): kurulum komutunu klasörün yeni sürümünden tekrar çalıştırın.
- Düzenlenebilir kurulum (editable install): klasörü güncelleyin (örneğin `git pull` ile). Sürüm numarası değiştiyse,
  `pip list` yeni sürümü göstersin diye kurulum komutunu tekrar çalıştırın.

### Kaldırma

```bash
python -m pip uninstall koco-ml
```

Kaynak klasör (source folder) bundan etkilenmez.

## Hızlı başlangıç

Eğitim verisi (training data) üzerinde bir boş değer (null) planı fit edin ve aynı değerleri yeni veriye uygulayın:

```python
import numpy as np
import pandas as pd
import koco

Train = pd.DataFrame({"Income": [3200, np.nan, 4100, 5000, 2800],
                      "Age":    [25, 40, np.nan, 35, 52]})
New = pd.DataFrame({"Income": [np.nan, 3900],
                    "Age":    [30, np.nan]})

Plan, Check = koco.FitNullPlan(Train, [("Income", "Median"), ("Age", "Mean")])
koco.ExportJson(Plan, "Plans/NullPlan.json")   # planı sonraki skorlama (scoring) için sakla

print(koco.ApplyNullPlan(New, Plan))
#    Income   Age
# 0  3650.0  30.0     <- Train'den Income medyanı (median)
# 1  3900.0  38.0     <- Train'den Age ortalaması (mean)
```

Veri yüklemeden model paketiyle (model bundle) skorlamaya (scoring) kadar eksiksiz regresyon (regression) ve ikili sınıflandırma (binary classification) örnekleri
[GuideTR.md](https://github.com/ffarukkoc/koco-ml/blob/main/GuideTR.md#uçtan-uca-örnek-end-to-end-example) içindedir.

## Kurallar (conventions)

- Fonksiyonlar (function), parametreler (parameter) ve değişkenler (variable) PascalCase kullanır. Girdi DataFrame parametresi (parameter) her zaman `Table`'dır.
- `ModelType`, `"Regression"` veya `"Binary"` değerini alır.
- Algoritma anahtarları (algorithm key) `R` (regresyon (regression)) veya `C` (ikili sınıflandırma (binary classification)) ile biter, örn. `RandFR` / `RandFC`.
  `koco.ModelRegistryGuide()` bunları listeler.
- Fit edilmiş planlar `{"Columns": {...}}` biçiminde düz dict'lerdir ve `ExportJson` ile kaydedilebilir.
- Her `Export*` fonksiyonu (function) `FilePath` alır: `None`, geçerli klasöre `<yymmdd>_<FunctionName>.<ext>` yazar;
  uzantılı bir yol dosya adı olarak kullanılır; bir klasör varsayılan (default) dosya adını alır. Ayrıntılar için
  [Çıktı dosyaları ve mesajlar](https://github.com/ffarukkoc/koco-ml/blob/main/GuideTR.md#çıktı-dosyaları-ve-mesajlar) bölümüne bakın.
- Yazdırılan mesajlar tek bir sembol seti kullanır: `❗` dikkat gerektirir, `ℹ` bilgi, `✓` tamamlandı,
  `•` ilerleme.

## Sorun giderme

### `ModuleNotFoundError: No module named 'koco'`

- **Notebook kurulumdan önce başlatıldı:** kurulu bir paket yalnızca yorumlayıcı (interpreter) başlarken algılanır.
  Kernel'i yeniden başlatın; çıktıları temizlemek yeterli değildir.
- **Farklı yorumlayıcı (interpreter):** notebook'taki `sys.executable` ile kurulum için kullanılan `python`'u karşılaştırın.
- **`koco.py` adlı yerel dosya:** notebook'un veya script'in yanındaki bu isimde bir dosya paketi gölgeler.
  Dosyayı yeniden adlandırın ve `__pycache__` klasörünü silin.
- **Klasör taşındıktan sonra düzenlenebilir kurulum (editable install):** ortam hâlâ eski konumu gösterir. Düzenlenebilir
  kurulumu (editable install) yeni konumdan tekrar çalıştırın.
- **Yeni Python sürümü:** paketler Python sürümü başına kurulur. koco'yu yeni yorumlayıcıyla (interpreter) tekrar kurun.

### `import koco` başarılı, ancak fonksiyonlar (function) eksik

Paket yerine dış proje klasörü (`pyproject.toml` içeren klasör) import ediliyor. Bu, söz konusu klasör
`site-packages` içine kopyalandığında veya `sys.path`'e eklendiğinde olur. Kopyayı kaldırın ve pip ile kurun.

### Notebook kod değişikliklerini görmüyor (düzenlenebilir kurulum (editable install))

Çalışan bir kernel eski kodu bellekte tutar. Kernel'i yeniden başlatın veya değişen modülü yeniden yükleyin:

```python
import importlib, koco
importlib.reload(koco.analysis)
importlib.reload(koco)
```

### `pip list` eski bir sürüm gösteriyor

Paket meta verisindeki (package metadata) sürüm kurulum anında yazılır. Kurulum komutunu tekrar çalıştırın.

### `Connection` macOS / Linux'ta hata veriyor

- **`Can't open lib 'ODBC Driver 17 for SQL Server' : file not found`:** unixODBC sürücü kaydını bulamıyor.
  `odbcinst -q -d` ile kontrol edin, ardından SQL yapılandırma dosyasına `"ODBCSYSINI"` (`odbcinst.ini` klasörü) ekleyin.
- **`Cannot generate SSPI context` veya diğer GSSAPI / Kerberos hataları:** bileti `klist` ile kontrol edin (`kinit` ile yenileyin)
  ve `krb5.conf` `/etc` altında değilse SQL yapılandırma dosyasına `"KRB5_CONFIG"` ekleyin.
- Bu ayarları değiştirdikten sonra kernel'i yeniden başlatın. Bkz.
  [ODBC ve Kerberos kurulumu](https://github.com/ffarukkoc/koco-ml/blob/main/GuideTR.md#macos--linux-odbc-ve-kerberos-kurulumu).

## Geliştirme

Geliştirme, test bağımlılıklarıyla (dependency) birlikte düzenlenebilir kurulum (editable install) kullanır:

```bash
python -m pip install -e ".[dev,all]"
python -m pytest
```

Test paketi ayrıca her null stratejisinin (strategy), ölçekleme (scaling) yönteminin (method), algoritma anahtarının (algorithm key) ve metriğin (metric)
rehberlerde bir karşılığı olduğunu da kontrol eder.

### Proje yapısı

```
koco/                        <- proje kökü
├── pyproject.toml           <- paket meta verisi (package metadata) ve bağımlılıklar (dependency)
├── README.md
├── GUIDE.md                 <- iş akışı (workflow) ve fonksiyon referansı (function reference)
├── tests/                   <- pytest paketi
└── koco/                    <- paket
    ├── __init__.py          <- sürüm; her modülün public isimlerini yeniden dışa aktarır
    ├── _helpers.py          <- modüller arasında paylaşılan private yardımcılar
    ├── general.py           <- veritabanı bağlantısı, import/export, model paketleri (model bundle)
    ├── analysis.py          <- veri özetleri, gruplama (binning), aykırı değer (outlier) ve temsil raporları (representation report), grafikler (chart)
    ├── preprocessing.py     <- null, aykırı değer (outlier), WOE ve ölçekleme (scaling) planları
    ├── feature_selection.py <- korelasyon (correlation), VIF, değişken seçimi (feature selection)
    ├── registry.py          <- algoritma kaydı (algorithm registry) (ModelSpec, MODEL_SPECS)
    ├── metrics.py           <- regresyon (regression) ve sınıflandırma (classification) metrikleri (metric)
    ├── modelling.py         <- girdi kontrolleri, train/test ayrımı (split), model eğitimi
    ├── scoring.py           <- tahminler (prediction) ve model paketi (model bundle) skorlama (scoring)
    └── guides.py            <- stratejiler (strategy), yöntemler (method), algoritmalar ve metrikler (metric) için açıklayıcı rehberler
```

Bağımlılık (dependency) kuralları (döngüsel import (circular import) yok):

- `_helpers` koco'dan hiçbir şey import etmez; `registry` ve `metrics` yalnızca `_helpers`'ı import eder.
- `guides` herhangi bir modülü import edebilir; hiçbir modül `guides`'ı import etmez.

### koco'yu genişletme

- **Fonksiyon (function):** ilgili modüle ve o modülün `__all__` listesine ekleyin, ardından `GUIDE.md` içinde
  belgeleyin. `__init__.py` her `__all__`'u yeniden dışa aktarır, böylece `koco.NewFunction()` başka bir
  değişiklik gerekmeden çalışır. İki modül aynı ismi dışa aktarırsa `import koco` bir `ImportError` fırlatır.
- **Algoritma:** `koco/registry.py` içindeki `MODEL_SPECS`'e bir `ModelSpec(...)` girdisi ekleyin.
  Kayıtlar, `ModelRegistryGuide` ve `CheckModellingInputs` içindeki kontroller buradan türetilir.
  Anahtarlar `R` veya `C` ile biter.
- **Metrik (metric):** sütununu `koco/metrics.py` içindeki `PERFORMANCE_COLUMNS`'a, İngilizce ve Türkçe metnini
  `koco/guides.py` içindeki `_METRIC_GUIDE`'a ekleyin.
- **Modül:** modülü (örneğin `koco/newgroup.py`) bir `__all__` listesiyle oluşturun ve şu satırları
  `koco/__init__.py`'ye ekleyin:

  ```python
  from . import newgroup
  from .newgroup import *
  __all__ += newgroup.__all__
  ```

### Kodlama kuralları (coding conventions)

- Fonksiyon (function) tanımlarında her satırda bir parametre (parameter).
- `pandas`, `numpy` ve `openpyxl` modül seviyesinde import edilir. Ağır veya opsiyonel (optional) paketler
  (sqlalchemy, matplotlib, scikit-learn, statsmodels, scipy, joblib) onları kullanan fonksiyonların (function)
  içinde import edilir.
- Birden fazla modül tarafından kullanılan isimler (`MODEL_SPECS`, `PERFORMANCE_COLUMNS`,
  `NULL_STRATEGIES`, ...) `_` ile başlamaz; tek bir modül içinde kullanılan isimler `_` ile başlar.
- Docstring'ler aşağıdaki bölümleri bu sırayla kullanır. Hiçbir hata fırlatılmıyorsa `Raises` `None.`
  olur; `Notes` ve `Notes (TR)` yalnızca gerektiğinde yer alır. Metin İngilizcedir; Türkçe metin
  yalnızca `Notes (TR)` içinde yer alır. Başlıklar 4, girdiler 7 boşlukla girintilenir.

  ```
  """
  One-line summary.

  Optional longer description.

  Arg:
     Name: Description.

  Returns:
     Description.

  Raises:
     ValueError: When ...

  Notes:
     Optional English notes.

  Notes (TR):
     Optional Turkish notes.
  """
  ```

### Sürümleme

Sürüm yalnızca `koco/__init__.py` içinde (`__version__`) tanımlanır; `pyproject.toml` onu oradan okur.
Değiştirdikten sonra paket meta verisinin (package metadata) güncellenmesi için kurulum komutunu tekrar çalıştırın.
