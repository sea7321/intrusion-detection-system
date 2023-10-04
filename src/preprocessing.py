import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def process(data, preprocessor, categorical_columns, numeric_columns, target_column):
    # Extract and keep the target variable
    y = data[target_column]

    # Apply preprocessing to the feature columns
    X_transformed = preprocessor.fit_transform(data.drop(target_column, axis=1))

    # Get the feature names after one-hot encoding
    categorical_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
        input_features=categorical_columns)

    # Combine the categorical feature names with numeric column names
    feature_names = list(categorical_feature_names) + numeric_columns

    # Convert the transformed array back to a DataFrame with appropriate column names
    transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    # Concatenate the feature DataFrame with the target variable
    return pd.concat([transformed_df, y], axis=1)


def preprocess():
    # Load the training and testing data
    print("Loading in the training data (dataset 1)...")
    training_data = pd.read_csv('../data/train.csv')

    print("Loading in the testing data (dataset 2)...")
    testing_data = pd.read_csv('../data/test.csv')

    # Separate categorical, numerical, and target columns
    categorical_columns = ['protocol_type', 'service', 'flag']
    numeric_columns = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                       'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                       'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                       'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                       'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                       'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                       'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
    target_column = ['class']

    # Create transformers for one-hot encoding and standard scaling
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse=False))
    ])

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Combine transformers using ColumnTransformer for feature columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns),
            ('num', numeric_transformer, numeric_columns)
        ], remainder='passthrough'  # This includes the target variable without transformation
    )

    # Process the training and testing data
    print("Processing the training data...")
    training_df = process(training_data, preprocessor, categorical_columns, numeric_columns, target_column)

    print("Processing the testing data...")
    testing_df = process(testing_data, preprocessor, categorical_columns, numeric_columns, target_column)

    # Fill columns with 0 if they don't exist in the other dataset
    for column in training_df.columns:
        if column not in testing_df:
            testing_df[column] = 0

    for column in testing_df.columns:
        if column not in training_df:
            training_df[column] = 0

    # Reorder columns and concatenate both datasets
    print("Reordering and combining datasets...")
    testing_df = testing_df[training_df.columns.tolist()]
    result = pd.concat([training_df, testing_df])

    # Send the resulting dataframe to a CSV file
    print("Writing the data to a CSV file...")
    result.to_csv(r'../data/training_data.csv', index=False)
    print("Wrote the data to ./data/training_data.csv")
