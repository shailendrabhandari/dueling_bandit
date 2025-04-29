from setuptools import setup, find_packages

setup(
    name="dueling-bandit",
    version="0.1.1",
    packages=["dueling_bandit"],  
    package_dir={"dueling_bandit": "dueling_bandit"},
    include_package_data=True,def random_bt(k: int, d: int = 0, seed: Optional[int] = None) -> "DuelingBanditEnv":
    rng = np.random.default_rng(seed)
    utilities = rng.normal(size=k)
    P = 1.0 / (1.0 + np.exp(utilities[:, None] - utilities[None, :]))
    np.fill_diagonal(P, 0.5)
    features = rng.normal(size=(k, d)) if d > 0 else None
    return DuelingBanditEnv(P, features, seed)
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
    ],
    author="Shailendra",
    author_email="shailendra.bhandari@oslomet.no",  
    description="A toolkit for preference-based online learning with dueling bandits",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shailendrabhandari/dueling_bandit",  
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
)