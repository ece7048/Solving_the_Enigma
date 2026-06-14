from setuptools import find_packages, setup


setup(
    name="SR_XAI",
    version="0.1.0",
    description="Deep learning 3D XAI explanation optimizer in Python",
    author="Michail Mamalakis",
    author_email="mm2703@cam.ac.uk",
    license="GPL-3.0+",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "PyYAML",
        "wandb",
        "torch>=1.10",
        "nibabel",
        "unfoldNd",
        "torch_geometric",
        "scikit-image",
        "qiskit",
        "torchvision",
        "torchdata",
        "matplotlib==3.3.4",
        "quantus",
        "scikit-learn",
        "scipy",
        "captum",
        "monai",
        "pandas",
        "seaborn",
        "torchmetrics",
        "einops",
    ],
    entry_points={
        "console_scripts": [
            "sr-xai=SR_XAI.cli:main",
        ],
    },
    zip_safe=False,
)

