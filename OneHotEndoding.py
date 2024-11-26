import pandas as pd


class OneHotEncoder:
    """Custom preprocessing step for one-hot encoding specified columns."""

    def __init__(self, columns_to_encode):
        """
        Initialize the OneHotEncoder.

        Parameters:
        -----------
        columns_to_encode : list
            List of column indices to one-hot encode
        """
        self.columns_to_encode = columns_to_encode
        self.column_names = None
        self.unique_values = {}

    def fit(self, X, y=None):
        """
        Fit the encoder by learning the unique values in each specified column.

        Parameters:
        -----------
        X : pandas DataFrame
            Input features to fit the encoder
        y : array-like, optional
            Target variable (not used in this transformer)

        Returns:
        --------
        self
        """
        # Store original column names
        self.column_names = X.columns

        # For each column to encode, store its unique values
        for col_name in self.columns_to_encode:
            self.unique_values[col_name] = sorted(X[col_name].unique())

        return self

    def transform(self, X, y=None):
        """
        Transform the data by one-hot encoding the specified columns.

        Parameters:
        -----------
        X : pandas DataFrame
            Input features to transform
        y : array-like, optional
            Target variable (not used in this transformer)

        Returns:
        --------
        pandas DataFrame
            Transformed DataFrame with one-hot encoded features
        """
        # Make a copy of the input DataFrame
        X_transformed = X.copy()

        # Process each column that needs to be encoded
        for col_idx in self.columns_to_encode:
            try: 
                col_name = self.column_names[col_idx]

                # Create dummy variables for the current column
                dummies = pd.get_dummies(
                    X_transformed.iloc[:, col_idx], prefix=f"{col_name}", dtype=int
                )

                # Drop the original column and add the dummy variables
                X_transformed = X_transformed.drop(columns=[col_name])
                X_transformed = pd.concat([X_transformed, dummies], axis=1)
            except: 
                pass

        return X_transformed
