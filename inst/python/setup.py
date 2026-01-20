from setuptools import setup, find_packages

setup(
    name="autoSDM",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "earthengine-api",
        "pandas",
        "numpy",
        "geopandas",
        "shapely",
        "scipy"
    ],
    entry_points={
        "console_scripts": [
            "autoSDM-py=autoSDM.cli:main",
        ],
    },
)
