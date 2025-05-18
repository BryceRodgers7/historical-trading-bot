from setuptools import setup, find_packages

setup(
    name="historical-trading-bot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "matplotlib",
        "numpy"
    ]
) 