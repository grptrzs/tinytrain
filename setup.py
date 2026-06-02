"""tinytrain setup."""

from setuptools import setup, find_packages

setup(
    name="tinytrain",
    version="0.2.1",
    description="Dead simple GPU training for PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mikhail S",
    author_email="mikhail@tinytrain.dev",
    url="https://github.com/MrMiff/tinytrain",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "torchvision>=0.15"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="pytorch gpu training deep-learning machine-learning amp mixed-precision",
)
