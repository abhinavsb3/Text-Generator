from setuptools import setup, find_packages

setup(
    name="text-gen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
    ],
    author="Abhinav SB",
    author_email="abhinavszb@gmail.com",
    description="A PyTorch-based text generation model trained on Shakespeare's works from Tiny Shakespeare dataset to generate Shakespeare-like text sequences",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/abhinavsb3/Text-Generator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "text-gen=text_gen.main:main",
        ],
    }
)