import pandas as pd
import numpy as np
import missingno as msn
import matplotlib.pyplot as plt

def main():
    # Read the CSV file
    
    mpg = pd.read_csv("https://raw.githubusercontent.com/tidyverse/ggplot2/master/data-raw/mpg.csv")
    
    # Display information about the DataFrame
    print("DataFrame Information:")
    mpg.info()
    print("-------------------------------------------------------")
    
    # Print the columns with numeric data types
    print("Numeric Columns:")
    print(mpg.select_dtypes('number').columns)
    print("-------------------------------------------------------")
    
    # Equal interval discretization
    mpg['cty2interval'] = pd.cut(mpg['cty'], bins=3)
    print("Equal Interval Discretization:")
    print(mpg['cty2interval'].value_counts())
    print("-------------------------------------------------------")
    
    # Equal frequency discretization
    mpg['cty2freq'] = pd.qcut(mpg['cty'], q=2, labels=['low', 'high'], duplicates=)
    print("Equal Frequency Discretization:")
    print(mpg['cty2freq'].value_counts())
    print("-------------------------------------------------------")
    
    for (val, cat) in zip(range(10), pd.qcut(np.arange(10), 2)):
        print(f"{val}: {cat}")
              
    print("-------------------------------------------------------")       
    obj_col = mpg.select_dtypes('object').columns
    for col in obj_col:
        print(f"{col} has {mpg[col].nunique()} unique values")
    
    print(pd.concat([pd.get_dummies(mpg['drv']), mpg['drv']], axis=1))
    
    pivot = mpg.pivot()

if __name__ == "__main__":
    main()
