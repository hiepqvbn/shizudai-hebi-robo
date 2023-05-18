# Pandas

## Reading and Writin Data

### I/O API Tools
readers: read_
writers: to_
type: csv, excel, hdf, sql, json, html, stata, clipboard, pickle, msgpack, gbq, *table*

### RegExp
. : single character, except newline
\d : Digit
\D : Non-digit character
\s : white space character
\S : Non-whitespace character
\n : new line character
\t : tab character
\uxxxx : unicode character specified by the hexademical number xxxx

## Pandas in Depth

### Data Preparation

- Loading
- Assembling
    - Merging: pandas.merge() connects the rows in a dataframe based on one or more keys.
    - Concatenating: pandas.concat() concaenates the objects along an axis
    - Combining: pandas.DataFrame.combine_first() allows you to connect overlapped data in order to fill in missing values in a data structure by taking data from another structure
- Reshaping(pivoting)
- Removing

### Data Transformation

- Removing duplicates: duplicated()
- Mapping:
    - replace() replaces values
    - map() creates a new column
    - rename() replaces the index values
- Discretization and Binning
- Detecting and Filtering Outliers
- Permutation
- Random Sampling
- String Manipulation
    - Built-in Methods for String Manipulation
    - Regular Expressions
- Data Aggregation
    - Group by
    - Hierarchical Grouping
    - Group Iteration
    - Chain of Transformations
    - Functions on Groups
- Advanced Data Aggregation
