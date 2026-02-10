# Setup script for NIDS package

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nids",
    version="1.0.0",
    author="jivi001",
    author_email="jiviteshgd28@gmail.com",
    description="Production-grade Hybrid Network Intrusion Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jivi001/Network-IDS-ML",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nids-train=scripts.train:main",
            "nids-evaluate=scripts.evaluate:main",
            "nids-cross-eval=scripts.cross_dataset_eval:main",
        ],
    },
)
