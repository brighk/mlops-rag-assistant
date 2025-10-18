from setuptools import setup, find_packages

setup(
    name="mlops-rag-assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "langchain",
        "chromadb",
        "fastapi",
        "mlflow",
    ],
    python_requires=">=3.10",
)
