import os
import pathlib
from setuptools import setup

_HERE = pathlib.Path(__file__).parent

setup(
    name="boundary_curvature",
    version="0.0.1",
    description="Method to calculate boundary curvature from binarized image",
    long_description=open(os.path.join(_HERE, "README.md"), "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mahdilamb/boundary_curvature",
    author="Mahdi Lamb",
    author_email="mahdilamb@gmail.com",
    license_files=("LICENSE",),
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",

    ],
    packages=["boundary_curvature"],
    include_package_data=True,
    install_requires=["numpy", "scikit-image"],

)
