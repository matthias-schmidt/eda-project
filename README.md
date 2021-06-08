# EDA Project

Matthias Schmidt

2021-06-07
 
Data science course at neuefische


## Data Set 

King_County_House_prices_dataset.csv from kaggle.com


## Pieces

- py script `king_county_house_prices.py` providing function `train()` which 
  - extracts data points according to preset cutoffs
  - sets the significant variables
  - replaces variable `zipcode` by `location_score`, defined as the mean price per sqft living area in zip code region 
  - determines linear regression parameters on these variables from training set
  - saves the linear regression parameters to `model_parameters.csv`
  - saves the location score table to `location_score.csv`
  - returns linear regression summary and RMSE computed on test set

- jupyter notebook `king_county_house_prices.ipynb` containing a detailed explanation of the data exploration behind

- pdf file `king_county_house_prices.pdf` with slides of project presentation
  
