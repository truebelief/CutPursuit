from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cut-pursuit-l2",
    version="0.1.1",
    author="Zhouxin Xi",
    author_email="truebelief2010@gmail.com",
    description="A Python implementation of the Cut Pursuit algorithm for graph optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/truebelief/CutPursuit/",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy<2.0.0",
        "scipy",
        "PyMaxflow",
    ],
)