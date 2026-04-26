from setuptools import find_packages, setup

setup(
    name="conservation-discovery",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
)

