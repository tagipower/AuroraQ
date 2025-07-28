#!/usr/bin/env python3
"""
AuroraQ Production 패키지 설치 스크립트
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="auroraQ-production",
    version="1.0.0",
    author="AuroraQ Team",
    author_email="contact@auroraQ.com",
    description="실시간 하이브리드 거래 시스템 - PPO와 Rule-based 전략 결합",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/auroraQ/auroraQ-production",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "gpu": [
            "torch[cuda]>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "auroraQ=main:main",
            "auroraQ-demo=main:main --mode demo",
            "auroraQ-test=main:main --mode test",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    keywords="trading, algorithmic-trading, machine-learning, reinforcement-learning, cryptocurrency, finance",
    project_urls={
        "Bug Reports": "https://github.com/auroraQ/auroraQ-production/issues",
        "Source": "https://github.com/auroraQ/auroraQ-production",
        "Documentation": "https://docs.auroraQ.com",
    },
)