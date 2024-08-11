# Data Science Experiment 3 - Data Preprocessing: Applying Encoding Techniques

This repository contains the code and results for the third experiment in the Data Science Fundamentals with Python course. The focus of this experiment is on encoding categorical variables using different encoding techniques to prepare data for machine learning models.

## Experiment Overview

### **Introduction to Encoding Techniques**
Encoding techniques are essential for converting categorical variables into numerical format for use in machine learning algorithms. Two common encoding techniques used in this experiment are:

1. **One-Hot Encoding**: Converts categorical values into a set of binary columns where each unique category is represented by a binary column. This is ideal for nominal data without an inherent order.

2. **Ordinal Encoding**: Assigns unique integers to categories with an inherent order. This method is best suited for ordinal data where categories have a meaningful sequence.

## Steps to Reproduce

### 1. Set up Google Colab:
   - Open [Google Colab](https://colab.research.google.com/).
   - Create a new notebook.

### 2. Import Necessary Libraries:
   - Start by importing the required libraries.

    ```python
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
    ```

### 3. Load the Dataset:
   - Use the provided dictionary to create a DataFrame.

    ```python
    d = {'sales': [100000,222000,1000000,522000,111111,222222,1111111,20000,75000,90000,1000000,10000],
         'city': ['Tampa','Tampa','Orlando','Jacksonville','Miami','Jacksonville','Miami','Miami','Orlando','Orlando','Orlando','Orlando'],
         'size': ['Small', 'Medium','Large','Large','Small','Medium','Large','Small','Medium','Medium','Medium','Small']}
    df = pd.DataFrame(data=d)
    df.head()
    ```

### 4. One-Hot Encoding:
   - Apply One-Hot Encoding to the 'city' column.

    ```python
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
    ohetransform = ohe.fit_transform(df[['city']])
    df = pd.concat([df, ohetransform], axis=1).drop(columns='city')
    df.head()
    ```

   - Save the One-Hot Encoded DataFrame.

    ```python
    df.to_csv('OneHotEncoded.csv', index=True)
    ```

### 5. Ordinal Encoding:
   - Apply Ordinal Encoding to the 'size' column.

    ```python
    ordinal_df = pd.DataFrame(data=d)
    sizes = ['Small', 'Medium', 'Large']
    enc = OrdinalEncoder(categories=[sizes])
    ordinal_df['size'] = enc.fit_transform(ordinal_df[['size']])
    ordinal_df.head()
    ```

   - Save the Ordinal Encoded DataFrame.

    ```python
    ordinal_df.to_csv('OrdinalEncoded.csv', index=True)
    ```

## How to Use
1. Clone this repository to your local machine or run the provided code in Google Colab.
2. Follow the instructions in the notebook to apply encoding techniques to your dataset.

## Dataset
The dataset used in this experiment is synthetic and created within the notebook for demonstration purposes.

## Concepts Used
- **Pandas Library**: For data manipulation and analysis.
- **One-Hot Encoding**: Converting categorical data to binary columns.
- **Ordinal Encoding**: Encoding ordinal data into integer values based on a predefined order.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
