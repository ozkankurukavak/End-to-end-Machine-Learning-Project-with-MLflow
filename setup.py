import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "End-to-end-ML-Project-with-DVC-MLflow"
AUTHOR_USER_NAME = "ozkan"
SRC_REPO = "mlProject"
AUTHOR_EMAIL = "kurukavakozkan57@gmail.com"




setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author = AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="makıne ogrenımı python",
    long_description=long_description,
    long_description_content = "text/markdown",
    url= f"https://github.com/ozkankurukavak/End-to-end-Machine-Learning-Project-with-MLflow",
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src")
    
)