"""
Setup script for QWEN-OMNI package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qwen-omni",
    version="0.1.0",
    author="QWEN-OMNI Team",
    author_email="contact@qwen-omni.com",
    description="Multimodal Language Model for Text-to-Speech with SNAC Audio Tokenization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qwen-omni/qwen-omni",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
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
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "advanced": [
            "flash-attn>=2.3.0",
            "triton>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qwen-omni-train=qwen_omni.pretraining.examples.basic.multimodal_training_example:main",
            "qwen-omni-infer=qwen_omni.pretraining.examples.basic.multimodal_inference_example:main",
        ],
    },
    include_package_data=True,
    package_data={
        "qwen_omni": ["*.yaml", "*.json", "*.txt"],
    },
)
