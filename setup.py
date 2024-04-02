from setuptools import setup

setup(
    name="optical_rl_gym",
    version="0.0.2-alpha",
    install_requires=["gym==0.21.0", "numpy", "matplotlib", "networkx"],
    extras_requires={"dev": ["flake8", "mypy", "isort", "pylint", "black"]},
)
