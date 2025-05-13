from setuptools import setup, find_packages

setup(
    name="genomeBridge",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "peft",
        "scikit-learn",
        "gdown",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "genomebridge-cli = genomeBridge.cli:main",
        ],
    },
)
