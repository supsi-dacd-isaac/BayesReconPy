import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bayesreconpy",
    version="0.1",
    author="A.Biswas, L.Nespoli",
    author_email="anubhab.biswas@supsi.ch",
    description="Bayesian reconciliation for hierarchical forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/supsi-dacd-isaac/BayesReconPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy>=2.1.1",
                      "pandas>=2.2.2",
                      "PuLP>=2.9.0",
                      "scipy>=1.14.1,",
                      "KDEpy>=1.1.10",
                      ],
    python_requires='>=3.8',
)