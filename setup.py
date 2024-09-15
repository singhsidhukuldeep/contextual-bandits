from setuptools import setup, find_packages

setup(
    name="contextual-bandits",
    version="0.1.0",
    description="A library for contextual multi-armed bandit algorithms.",
    author="Kuldeep Singh Sidhu",
    author_email="singhsidhukuldeep@gmail.com",
    url="https://github.com/singhsidhukuldeep/contextual-bandits",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "scikit-learn>=0.22.0",
        "torch>=1.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
