import numpy as np
import pandas as pd
import scipy
import sqlite3

def read_sqlite_db_to_pandas(query: str, db_name: str = "penguins.db") -> pd.DataFrame:
    """Read a table from a SQLite database into a pandas DataFrame.

    NOTE: You do not need to change this function.

    Args:
        query: SQL query to select the data from the table.
        db_name: Name of the SQLite database file.

    Returns:
        DataFrame with the data from the table.
    """
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_sql_query_basic() -> str:
    """Return a SQL query as a string.

    NOTE: 
        - Your query should get all columns and rows from the penguins table.
        - Do *not* aggregate, sort, or change the table in any way.

    Returns:
        SQL query as a string.
    """
    query = """
    INSERT_YOUR_QUERY_HERE.
    """
    return query

def get_sql_query_challenge() -> str:
    """Return a SQL query as a string.

    NOTE:
        - You should query the penguins table. 
        - Return the species, sex, and the sum of bill_length_mm (renamed to sum_bill_length_mm) PER SPECIES.
        - The order of the columns [species, sex, sum_bill_length_mm] is important.
        - Only include rows where the sex is 'male'.
        - Order by sum_bill_length_mm in ascending order.

    Returns:
        SQL query as a string.
    """
    query = """
    INSERT_YOUR_QUERY_HERE.
    """
    return query


def get_float64_column_names(df: pd.DataFrame) -> list[str]:
    """Get the float64 column names from a DataFrame.

    NOTE: This method should return a list of column names (str). NOT
    an Index. Pandas will often use non-standard python data
    structures for indices and arrays, and we want to be explicit
    about the return type. To cast a pandas Index to a list of strings,
    you can use the .tolist() method.

    Args:
        df: DataFrame to extract float64 columns from.

    Returns:
        List of strings with the float64 column names.
    """
    raise NotImplementedError("You need to implement this function.")
    
def get_missing_value_indices(df: pd.DataFrame, column: str) -> list[int]:
    """Get the row indices of missing values within a column.

    NOTE: This method should return a list of ints. NOT an Index.
    Pandas will often use non-standard python data structures for indices
    and arrays, and we want to be explicit about the return type.
    To cast a pandas Index to a list of integers, you can use the .tolist()
    method.

    Args:
        df: DataFrame with missing values.
        column: Column with missing values.

    Returns:
        Indices of missing values, as a list of ints.
    """
    raise NotImplementedError("You need to implement this function.")

def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values.

    NOTE: If **any** rows have missing values, drop those
    rows and return the resulting DataFrame.

    Args:
        df: DataFrame with missing values.

    Returns:
        DataFrame with missing values dropped.
    """
    raise NotImplementedError("You need to implement this function.")


def fill_float64_cols_with_random_sample(df: pd.DataFrame, random_state=2024) -> pd.DataFrame:
    """Fill float64 columns with a random sample from the column.

    NOTE: You don't need to change this function.
    
    Args:
        df: DataFrame with missing values and float64 type columns.
        random_state: Random seed for reproducibility, to be used with df.sample().

    Returns:
        DataFrame with missing values filled with random sample from the column.
    """
    float_columns = get_float64_column_names(df)
    for column in float_columns:
        # Get indices of missing values.
        missing_indices = get_missing_value_indices(df, column)

        # Get random sample from the column.
        random_sample = df[column].dropna().sample(len(missing_indices),
                                                   replace=True,
                                                   random_state=random_state)
        
        # Fill missing values with randomly sampled values.
        df.loc[missing_indices, column] = random_sample.values

    return df

def fill_float64_cols_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill float64 columns with the mean of the column.

    NOTE: You don't need to change this function.
    
    Args:
        df: DataFrame with missing values and float64 type columns.

    Returns:
        DataFrame with missing values filled with the mean of the column.
    """
    float_columns = get_float64_column_names(df)
    for column in float_columns:
        
        # Get indices of missing values.
        missing_indices = get_missing_value_indices(df, column)

        # Get the mean of the column.
        mean = df[column].mean(skipna=True)
        mean_values = [mean] * len(missing_indices)

        # Fill missing values with the mean.
        df.loc[missing_indices, column] = mean_values

    return df

def calculate_covariance_numpy(x: np.array, y: np.array) -> float:
    """Use only numpy to calculate the covariance.

    NOTE:
        - Do NOT use np.cov().
        - Use only basic numpy operations (like np.sum, np.mean, etc.)
    
    Args:   
        x: First array.
        y: Second array.
    
    Returns:
        Covariance between x and y.
    """
    raise NotImplementedError("You need to implement this function.")

def calculate_pearson_correlation_numpy(x: np.array, y: np.array) -> float:
    """Use only numpy to calculate pearson's correlation coefficient.

    NOTE:
        - Do NOT use np.corrcoef().
        - Use only basic numpy operations (like sum, mean, etc.)
    
    Args:
        x: First array.
        y: Second array.
    
    Returns:
        Pearson's correlation coefficient between x and y.
    """
    raise NotImplementedError("You need to implement this function.")
    
def calculate_pearson_correlation_scipy(x: np.array, y: np.array) -> float:
    """Use scipy to calculate pearson's correlation coefficient.

    NOTE:
        - You can use scipy.stats.pearsonr() to calculate the correlation
        coefficient.
        - Please return the "statistic" field from the function.
    
    Args:
        x: First array.
        y: Second array.
    
    Returns:
        Pearson's correlation coefficient between x and y.
    """
    raise NotImplementedError("You need to implement this function.")
    
def calculate_spearman_correlation_scipy(x: np.array, y: np.array) -> float:
    """Use scipy to calculate spearman's correlation coefficient.

    NOTE:
        You can use scipy.stats.spearmanr() to calculate the correlation
        coefficient. Please return the statistic from the function.
    
    Args:
        x: First array.
        y: Second array.
    
    Returns:
        Spearman's correlation coefficient between x and y.
    
    """
    raise NotImplementedError("You need to implement this function.")
   
def perform_hypothesis_test(x: np.array, y: np.array) -> tuple[float, float]:
    """Use scipy to perform the appropriate hypothesis test.

    NOTE:
        You must select the proper test. Here are your options (choose the best one):

            - scipy.stats.ttest_ind()
            - scipy.stats.ttest_rel()
            - scipy.stats.wilcoxon()
            - scipy.stats.mannwhitneyu()
            - scipy.stats.kruskal()
            - scipy.stats.f_oneway()
            - scipy.stats.chisquare()

    Please return the test statistic and p-value.
    
    Args:
        x: First array.
        y: Second array.
    
    Returns:
        Test statistic and p-value (float, float).
    """
    raise NotImplementedError("You need to implement this function.")
    
def check_normality(x: np.array) -> tuple[float, float]:
    """Check if a sample is normally distributed using Shapiro-Wilk test.

    NOTE:
        You can use scipy.stats.shapiro() to check for normality.
        Please return the test statistic and p-value.
    
    Args:
        x: Array to check for normality.
    
    Returns:
        Test statistic and p-value (float, float)from the Shapiro-Wilk test.
    """
    raise NotImplementedError("You need to implement this function.")
   
def check_variance_homogeneity(x: np.array, y: np.array) -> tuple[float, float]:
    """Check if two samples have equal variance using Levene's test.

    NOTE:
        You can use scipy.stats.levene() to check for equal variance.
        Please return the test statistic and p-value.
    
    Args:
        x: First array.
        y: Second array.
    
    Returns:
        Test statistic and p-value (float, float) from the Levene's test.
    """
    raise NotImplementedError("You need to implement this function.")