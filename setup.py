# project/setup.py
from setuptools import setup

setup(
    name="iblnm",
    version="0.1",
    # packages=find_packages(),
    packages=["iblnm"],
    install_requires=[
        "xarray",
    ],
)
