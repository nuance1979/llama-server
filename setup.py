# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Yi Su. All rights reserved.
#
from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = (Path("llama_server") / "__init__.py").read_text(encoding="utf-8").split()
    return init[init.index("__version__") + 2][1:-1]


def get_install_requires() -> str:
    return [
        "pyllamacpp",
        "fastapi",
        "uvicorn[standard]",
        "sse-starlette",
        "click",
    ]


def get_extras_require() -> str:
    req = {
        "dev": [
            "httpx",
            "pre-commit",
            "pytest",
            "pytest-cov",
            "pytest-xdist",
            "numpy",
        ],
    }
    return req


setup(
    name="llama-server",
    version=get_version(),
    description="LLaMA Server combines the power of LLaMA C++ with the beauty of Chatbot UI.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/nuance1979/llama-server",
    author="Yi Su",
    author_email="nuance@gmail.com",
    license="MIT",
    python_requires=">=3.8",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="llama llama.cpp chatbot-ui chatbot",
    packages=find_namespace_packages(exclude=["test", "test.*", "docs", "docs.*"]),
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
    package_data={"llama_server": ["py.typed"], "llama_server.prompts": ["*.txt"]},
    entry_points={"console_scripts": ["llama-server = llama_server.server:main"]},
)
