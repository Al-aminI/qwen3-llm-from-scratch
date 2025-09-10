"""
Setup script for LoRA & QLoRA package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lora-qlora",
    version="1.0.0",
    author="LoRA & QLoRA Fine-tuning Team",
    author_email="team@lora-qlora.com",
    description="A comprehensive package for efficient fine-tuning using LoRA and QLoRA techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/lora-qlora",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "advanced": [
            "accelerate>=0.20.0",
            "bitsandbytes>=0.41.0",
            "peft>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lora-train=lora_qlora.examples.basic.lora_example:main",
            "qlora-train=lora_qlora.examples.basic.qlora_example:main",
        ],
    },
)
