from setuptools import setup

setup(
    name="alignment_calculation",
    version="0.22",
    packages=["alignment_calculation"],
    package_data={
        "alignment_calculation": ["data/scenario_data/*.csv", "parameters.yaml"]
    },
    include_package_data=True,
    description="With this code the alignment calculations can be performed using",
    author="Michiel Nijhuis",
    author_email="michiel.nijhuis@ecb.europa.eu",
    exclude_package_data={"": ["*.ipynb"]},
    url="https://gitlab.sofa.dev/Michiel.Nijhuis/pacta",
)
