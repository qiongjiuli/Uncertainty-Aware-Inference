from setuptools import setup, find_packages

setup(
    name         = "uncertainty-aware-inference",
    version      = "0.1.0",
    description  = "How Quantization Affects LLM Confidence Calibration",
    author       = "Columbia HPML — Team A/B/C × IBM Research",
    packages     = find_packages(where="src"),
    package_dir  = {"": "src"},
    python_requires = ">=3.10",
    install_requires = open("requirements.txt").read().splitlines(),
)
