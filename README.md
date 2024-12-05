# Alignment to decrabonisation pathway calculation

The Alignment to decrabonisation pathway calculation package is a Python package that helps you to calculate the alignment to a decarbonisation pathway based on a loan portfolio and asset production data. It provides functions to analyze and assess the degree of alignment of various counterparties with a specified decarbonisation pathway.

## Installation

You can install the package using pip by navigating to the folder of the package and using the following command:

```
pip install .
```

The code needs to be connected to a data source having a loan book and a data source containing the production data and production projections. Mock version of these data are now present in the data/loan_data and the data/company_data folders.   

## Usage

To use the Alignment Calculation can be imported as follows:

```python
from alignment_calculation import alignmentCalculator
```

Then, you can use the provided functions to analyze alignment to a decarbonisation pathway.

Here's a basic example:

```python
# Define the alignment calculator
ac = alignmentCalculator(loan_file=True)

# Calculate the alignment scores based on all the banks in the AnaCredit sample
results = ac.calculate_net_alignment()

# Get company secific results to further analyse the alignment scores
results = ac.calculate_net_alignment(facet_col=['company_id'])

# Recalculate the scores when aggregating the data
aggregated_results = ac.group_scores(results, ['jst_code'])
```

For a more extensive example see the notebook *perform_pacta_calculations.ipynb*


## Contributing

All contributions are welcome. If you want to contribute please ask what features still need to be developed. If you have developed a new feature feel free to make a pull request describing the added feature.


## License

The project can be reused under the MIT Liscense



