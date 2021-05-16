from setuptools import find_packages, setup

setup(
    name="ml_project",
    packages=find_packages(),
    version="1.1.0",
    description="MADE Production ML project",
    author="Nalitkin Aleksandr",
    author_email="sasha.studies@gmail.com",
    install_requires=[
        "dataclasses==0.6",
        "marshmallow-dataclass==8.4.1",
        "pandas==1.0.1",
        "numpy==1.18.1",
        "lightgbm==3.2.0",
        "scikit-learn==0.22.1",
        "matplotlib==3.1.3",
        "seaborn==0.10.0",
        "tqdm==4.42.1",
        "notebook==6.0.3",
        "ipython==7.12.0",
        "Faker==8.1.2",
        "hydra-core==1.0.6",
        "pylint==2.4.4",
        "pytest==5.3.5",
    ],
    license="MIT",
)