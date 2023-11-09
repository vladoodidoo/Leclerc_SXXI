# Leclerc_SXXI

Authors:
    - gilles.recouvreux 
    - vlad.argatu 
    - maxence.oden 
    - raphael.mourot-pelade
    - bastien.pouessel

## Architecture

The project contains multiple libraries in order to simplify the code and to make it more readable in the notebooks.
- `data_analysis` contains the code for the data analysis.
- `preprocessing` contains the code for the preprocessing.
- `model` contains the code for the model.

The `eda.ipynb` notebook contains the code for the data analysis.
The `benchmark_models.ipynb` notebook contains the code for the model and comparison with other models.
The `LSTM.ipynb` notebook contains the code for the LSTM model.

## Dependencies
- skrub - 0.0.1.dev0
- pyarrow
- scikit-learn
- imbalanced-learn
- pandas
- numpy
- matplotlib
- seaborn
- tensorflow

### How to install skrub

`pip install git+https://github.com/skrub-data/skrub.git`