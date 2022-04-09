from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


install_requires = ["numpy", "pandas"]
test_requires = ["pytest", "coverage"]
dev_requires = test_requires

# read version from _version.py
with open("multielo/_version.py") as version_file:
    version = version_file.read().strip().split("=")[1].replace('"', "").replace("'", "").strip()

setup(
    name="multielo",
    version=version,
    description="Elo ratings for multiplayer matchups.",
    long_description=readme(),
    keywords="elo ratings rankings multiplayer",
    url="https://github.com/djcunningham0/multielo",
    author="Danny Cunningham",
    author_email="djcunningham0@gmail.com",
    license="MIT",
    packages=["multielo"],
    python_requires=">=3",
    install_requires=install_requires,
    extras_require={"tests": test_requires, "dev": dev_requires},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
)
