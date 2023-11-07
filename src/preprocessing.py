"""
File: preprocessing.py
Description: Preprocesses the NSL-KDD dataset
"""

# Third-Party Imports
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def process(data, preprocessor, categorical_columns, numeric_columns, target_column):
    """
    One-hot encodes and normalizes the data.
    :param data: (DataFrame) the dataset
    :param preprocessor: (ColumnTransformer) the column transformers
    :param categorical_columns: (Str[]) list of categorical columns in the dataset
    :param numeric_columns: (Str[]) list of numeric columns in the dataset
    :param target_column: (Str[]) the target column in the dataset
    :return: (DataFrame) the resulting DataFrame
    """
    # extract and keep the target variable
    y = data[target_column]

    # apply preprocessing to the feature columns
    X_transformed = preprocessor.fit_transform(data.drop(target_column, axis=1))

    # get the feature names after one-hot encoding
    categorical_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
        input_features=categorical_columns)

    # combine the categorical feature names with numeric column names
    feature_names = list(categorical_feature_names) + numeric_columns

    # convert the transformed array back to a DataFrame with appropriate column names
    transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    # concatenate the feature DataFrame with the target variable
    return pd.concat([transformed_df, y], axis=1)


def preprocess():
    """
    Preprocesses the NSL-KDD dataset into two datasets: misuse-based and anomaly-based.
    :return: None
    """
    # load the training and testing data
    print("\tLoading in the datasets...")
    training_data = pd.read_csv('../data/train.csv')
    testing_data = pd.read_csv('../data/test.csv')

    # separate categorical, numerical, and target columns
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

    # create transformers for one-hot encoding and standard scaling
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse_output=False))
    ])

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # combine transformers using ColumnTransformer for feature columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns),
            ('num', numeric_transformer, numeric_columns)
        ], remainder='passthrough'  # This includes the target variable without transformation
    )

    # process the training and testing data
    print("\tProcessing the datasets...")
    training_df = process(training_data, preprocessor, categorical_columns, numeric_columns, target_column)
    testing_df = process(testing_data, preprocessor, categorical_columns, numeric_columns, target_column)

    # fill columns with 0 if they don't exist in the other dataset
    for column in training_df.columns:
        if column not in testing_df:
            testing_df[column] = 0

    for column in testing_df.columns:
        if column not in training_df:
            training_df[column] = 0

    # reorder columns and concatenate both datasets
    print("\tReordering and combining datasets...")
    testing_df = testing_df[training_df.columns.tolist()]
    misuse_df = pd.concat([training_df, testing_df])

    # gather sqlattack and udpstorm attack rows
    sqlattack_df = misuse_df[misuse_df['class'] == "sqlattack"]
    udpstorm_df = misuse_df[misuse_df['class'] == "udpstorm"]

    # send first row to a separate testing file
    sqlattack_df.head(1).to_csv(r'../data/sqlattack_testing_data.csv', index=False)
    udpstorm_df.head(1).to_csv(r'../data/udpstorm_testing_data.csv', index=False)

    # drop sqlattack and udpstorm attacks from datasets
    misuse_df.drop(sqlattack_df.index, inplace=True)
    misuse_df.drop(udpstorm_df.index, inplace=True)

    # create a copy of the DataFrame for misuse data
    anomaly_df = misuse_df.copy()

    # replace values other than "normal"
    anomaly_df.loc[anomaly_df['class'] != 'normal', 'class'] = 'abnormal'

    # send the resulting dataframes to a CSV file
    print("\tWriting the data to a CSV file...")
    misuse_df.to_csv(r'../data/misuse_training_data.csv', index=False)
    print("\tWrote the data to ./data/misuse_training_data.csv")
    anomaly_df.to_csv(r'../data/anomaly_training_data.csv', index=False)
    print("\tWrote the data to ./data/anomaly_training_data.csv\n")
