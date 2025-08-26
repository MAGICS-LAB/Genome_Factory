from setuptools import setup, find_packages

setup(
    name="genomeFactory",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "peft",
        "scikit-learn",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "genomefactory-cli = genomeFactory.cli:main",
        ],
    },
)
