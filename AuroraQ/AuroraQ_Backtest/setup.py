#!/usr/bin/env python3
"""
AuroraQ Backtest 설치 스크립트
"""

from setuptools import setup, find_packages
import os

# README 파일 읽기
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# requirements.txt 파일 읽기
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="auroraQ-backtest",
    version="1.0.0",
    author="AuroraQ Team",
    author_email="team@auroraQ.com",
    description="고성능 암호화폐 백테스팅 프레임워크",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/auroraQ/backtest",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "cupy>=11.0.0",
        ],
        "web": [
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
            "streamlit>=1.15.0",
        ],
        "ml": [
            "scikit-learn>=1.1.0",
            "statsmodels>=0.13.0",
            "tensorflow>=2.10.0",
            "torch>=1.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "auroraQ-backtest=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
    keywords="backtest trading cryptocurrency finance algorithm",
    project_urls={
        "Documentation": "https://docs.auroraQ.com",
        "Source": "https://github.com/auroraQ/backtest",
        "Tracker": "https://github.com/auroraQ/backtest/issues",
    },
)