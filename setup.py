# Copyright 2024 the dydxla.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from setuptools import find_packages, setup

# Package Metadata
PACKAGE_NAME = "llamaforge"
DESCRIPTION = "A project for fine-tuning Llama models using LoRA and DeepSpeed."
URL = "https://github.com/dydxla/LlamaForge"  # Replace with your GitHub URL
AUTHOR = "dydxla"  # Replace with your name
AUTHOR_EMAIL = "dydxla@gmail.com"  # Replace with your email
LICENSE = "Apache License 2.0"
PYTHON_REQUIRES = ">=3.10"

EXTRAS_REQUIRE = {
    "dev": ["pytest>=6.2.5", "black", "flake8>=3.7.9"],  # Development and testing tools
    "metrics": ["nltk", "rouge-score"],   # Optional evaluation metrics
}

# Dependencies from requirements.txt
def get_requires():
    import sys
    with open("requirements.txt", encoding="utf-8") as f:
        requirements = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Skip deepspeed on Windows
            if "deepspeed" in line and sys.platform == "win32":
                continue
            # Handle platform-specific requirements
            if "sys_platform" in line:
                if "win32" in line and sys.platform == "win32":
                    requirements.append(line.split(";")[0].strip())
                elif "darwin" in line and sys.platform == "darwin":
                    requirements.append(line.split(";")[0].strip())
                elif "linux" in line and sys.platform == "linux":
                    requirements.append(line.split(";")[0].strip())
            else:
                requirements.append(line)
        return requirements

# Long Description
def get_long_description():
    with open("README.md", encoding="utf-8") as f:
        return f.read()

# Setup Function
setup(
    name=PACKAGE_NAME,
    version="1.0.0",  # Replace with dynamic versioning if necessary
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    packages=find_packages(where="src"),  # Assuming your package code is in "src/"
    package_dir={"": "src"},              # Define package root directory
    python_requires=PYTHON_REQUIRES,
    install_requires=get_requires(),
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
