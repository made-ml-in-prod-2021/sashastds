from setuptools import find_packages, setup

setup(
    name="ml_project",
    packages=find_packages(),
    version="1.1.0",
    description="MADE Production ML project",
    author="Nalitkin Aleksandr",
    install_requires=[
        "dataclasses==0.8",
        "marshmallow-dataclass==8.4.1",
        "pandas==1.0.1",
        "numpy==1.18.1",
        "lightgbm==3.2.0",
        "scikit-learn==0.22.1",
        "tqdm==4.42.1",
        "Faker==8.1.2",
        "hydra-core==1.0.6",
        "pylint=2.8.2",
        "pytest==6.2.3",
    ],
    license="MIT",
)