def train():
    
    
    ## CUTOFF PARAMETERS (INCLUSIVE)
    cutoff_price = 2000000
    cutoff_sqft_living = 6500
    cutoff_sqft_lot = 600000
    cutoff_bedrooms = 9
    cutoff_bathrooms = 4
    cutoff_floors = 3
    cutoff_view_lower = 0
    cutoff_grade_lower = 4
    cutoff_grade_upper = 12
    cutoff_condition_lower = 2
    
    
    ## LOAD MODULES
    
    import pandas as pd
    #import numpy as np
    from scipy import stats
    import statsmodels.formula.api as smf
    from sklearn import model_selection


    ## IMPORT DATA
    
    df = pd.read_csv('kc_house_prices/King_County_House_prices_dataset.csv')

    
    ## EXPLORE DATA
    
    # price cutoff
    df = df.query('price <= '+str(cutoff_price))

    # size cutoff
    df = df.query('sqft_living <= '+str(cutoff_sqft_living))
    df = df.query('sqft_lot <= '+str(cutoff_sqft_lot))

    # turn n.s. for view and waterfront into -1
    df.view.fillna(value=-1, inplace=True)
    df.waterfront.fillna(value=-1, inplace=True)

    # remove further outliers
    df = df.drop('id', axis=1)
    df = df.query('view >= '+str(cutoff_view_lower)
                  +' and grade >= '+str(cutoff_grade_lower)
                  +' and grade <= '+str(cutoff_grade_upper)
                  +' and condition >= '+str(cutoff_condition_lower))
    df = df.query('bedrooms <= '+str(cutoff_bedrooms)
                  +'and bathrooms <= '+str(cutoff_bathrooms)
                  +' and floors <= '+str(cutoff_floors))

    
    ## BUILD MODEL
    
    # get missing basement area values from living size area minus area of upper floors
    df['sqft_basement'] = df.eval('sqft_living - sqft_above')

    # now, living size area can be dropped
    df = df.drop('sqft_living', axis=1)

    # introduce dummy variables for view, waterfront, condition, grade
    dum_view = pd.get_dummies(df.view, drop_first=True, prefix='view')
    dum_waterfront = pd.get_dummies(df.waterfront, drop_first=True, prefix='waterfront')
    dum_condition = pd.get_dummies(df.condition, drop_first=True, prefix='condition')
    dum_grade = pd.get_dummies(df.grade, drop_first=True, prefix='grade')
    df = pd.concat([df, dum_view, dum_waterfront, dum_condition, dum_grade], axis=1)
    df = df.drop(['view', 'waterfront', 'condition', 'grade'], axis=1)
    df.columns = [col.replace('.0', '') for col in df.columns]
    
    # drop variables that turn out to not be significant
    df = df.drop(['long', 'lat', 'waterfront_0', 'grade_5', 'grade_6', 'grade_7', 'grade_8', 'yr_renovated', 'date'], axis=1)

       
    ## TRAINING

    # split data set into training set and test set
    df_test, df_train = model_selection.train_test_split(df, random_state=42)
    
    # determine location score (median of price per sqft livin area for fixed zipcode) on training set
    D = pd.DataFrame(df_train[['price', 'zipcode', 'sqft_basement', 'sqft_above']])
    D['price_per_sqft'] = D.eval('price / (sqft_basement + sqft_above)')
    loc_score_map = dict([(z, round(D.query(f'zipcode == {z}').price_per_sqft.median(), 2)) 
                          for z in D.zipcode.unique()])

    # replace zipcode by location score on training and test sets
    df_test.zipcode.replace(loc_score_map, inplace=True)
    df_train.zipcode.replace(loc_score_map, inplace=True)
        
    # final variables
    vars = df_test.columns[1:]
    
    # linear regression
    lin_reg_res = smf.ols(formula = 'price ~ ' + ' + '.join(vars), data = df_test).fit()
    
    # compute RMSE
    def predict(ds):
        return lin_reg_res.params[0] + sum(lin_reg_res.params[var] * ds[var] for var in vars)
    
    residuals = [row.price - predict(row) for ctr, row in df_test.T.iteritems()]
    RMSE = int((sum(x**2 for x in residuals) / len(residuals))**.5)


    ## SAVE MODEL
    
    # save parameters
    export = pd.DataFrame(lin_reg_res.params)
    export.to_csv('model_parameters.csv')
    
    # save location score table
    export = pd.DataFrame([loc_score_map])
    export.to_csv('location_score.csv')        
    
    
    ## RETURN SUMMARY AND RSME
    
    return lin_reg_res.summary(), 'RMSE = '+str(RMSE)