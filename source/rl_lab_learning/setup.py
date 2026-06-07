"""Editable install metadata for the rl_lab_learning package."""

from setuptools import find_packages, setup


setup(
    name="rl-lab-learning",
    version="0.1.0",
    description="Learning-oriented reinforcement-learning algorithms on Isaac Lab.",
    author="Hall",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)

