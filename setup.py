from setuptools import setup, find_packages

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Education",
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Programming Language :: Python",
]

setup(
    name="contextual-bandits-algos",
    version="0.1.0",
    description="A library for contextual multi-armed bandit algorithms.",
    # long_description=open("README.md").read(),
    author="Kuldeep Singh Sidhu",
    author_email="singhsidhukuldeep@gmail.com",
    url="https://github.com/singhsidhukuldeep/contextual-bandits",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "scikit-learn>=0.22.0",
        "torch>=1.7.0",
    ],
    classifiers=classifiers,
    keywords="Contextual Bandits, Multi-Armed Bandits, Reinforcement Learning, LinUCB, Epsilon-Greedy, UCB, Thompson Sampling, Kernel Methods, Neural Networks, Machine Learning, Python Library"
)

