import argparse
import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='raw_data')
    parser.add_argument('--output-dir', type=str, default='processed_data')
    parser.add_argument('--credit-appications', type=str, default='credit_applications.csv', help='name of file with transactions')
    parser.add_argument('--user-features', type=str, default='customers.csv', help='name of file with user features')
    parser.add_argument('--feature-cols', type=str, default='', help='comma separated id cols in transactions table')
    parser.add_argument('--label-col', type=str, default='credit_application', help='comma separated categorical cols in transactions')
    parser.add_argument('--train-data-ratio', type=float, default=0.8, help='fraction of data to use in training set')
    parser.add_argument('--test-data-ratio', type=float, default=0.1, help='fraction of data to use in validation set')
    return parser.parse_args()


def get_logger(name:str):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger


def load_data(data_dir, credit_data, features_data, label_col):

    credit_df = pd.read_csv(os.path.join(data_dir, credit_data),index_col=0)

    logging.info("Shape of credit data is {}".format(credit_df.shape))

    logging.info("Percent of  Credits in data : {}".format(credit_df[label_col].mean()*100))

    features_df = pd.read_csv(os.path.join(data_dir, features_data),index_col=0)

    logging.info("Shape of features data is {}".format(features_df.shape))

    assert len(features_df)==len(credit_df), "Credit data and features data must have same number of rows"

    merged_df = features_df.merge(credit_df, on=["client_nr","yearmonth"], how="left")

    logging.info("Shape of merged data is {}".format(merged_df.shape))

    return merged_df


def split_data(df, label_col, train_data_ratio, test_data_ratio):
    ## We will create seperate strarified sampling to replicate same ratio of credit/no credit as in original data
    val_data_ratio=1-train_data_ratio-test_data_ratio

    logging.info("Training, validation, and test data fraction are {}, {}, and {}, respectively".format(train_data_ratio, val_data_ratio, test_data_ratio))
    # Group by 'client_nr' and calculate mean 'credit_application'
    client_label_dist = df.groupby('client_nr')['credit_application'].mean().reset_index()

    # Create a binary stratum label based on label distribution
    label_dist = df['credit_application'].mean()
    client_label_dist['stratum'] = (client_label_dist['credit_application'] > label_dist).astype(int)

    # First Split: Separate Train+Validation and Test clients
    client_tv, client_test = train_test_split(client_label_dist, test_size=test_data_ratio, stratify=client_label_dist['stratum'], random_state=42)

    # Second Split: Separate Train and Validation clients from Train + Validation set
    client_train, client_val = train_test_split(client_tv, test_size=val_data_ratio/train_data_ratio, stratify=client_tv['stratum'], random_state=42) 

    # Aggregate data for Train, Validation, and Test sets based on the client splits
    train_df = df[df['client_nr'].isin(client_train['client_nr'])]
    val_df = df[df['client_nr'].isin(client_val['client_nr'])]
    test_df = df[df['client_nr'].isin(client_test['client_nr'])]

    logging.info("Percentage of training data : {}".format((len(train_df)/len(df))*100))
    logging.info("Percentage of validation data : {}".format((len(val_df)/len(df))*100))
    logging.info("Percentage of testing data : {}".format((len(test_df)/len(df))*100))

    logging.info("Percentage of credit applications for train data: {}".format(train_df[label_col].mean()*100))
    logging.info("Percentage of credit applications for validation data: {}".format(val_df[label_col].mean()*100))
    logging.info("Percentage of credit applications for test data: {}".format(test_df[label_col].mean()*100))
    logging.info("Percentage of credit applications for all data: {}".format(label_dist*100))

    return train_df,val_df,test_df




def get_features_and_labels(train_df, val_df, test_df, output_dir):

    train_df=train_df.copy()
    val_df=val_df.copy()
    test_df=test_df.copy()

    logging.info("CRG NA counts in train before imputation are {}".format(train_df['CRG'].isnull().sum()))

    crg_imputer=train_df['CRG'].median()
    train_df.loc[:, 'CRG'].fillna(crg_imputer, inplace=True)
    val_df.loc[:, 'CRG'].fillna(crg_imputer, inplace=True)
    test_df.loc[:, 'CRG'].fillna(crg_imputer, inplace=True)

    logging.info("CRG NA counts in train are {}".format(train_df['CRG'].isnull().sum()))

    X_train= train_df.drop(['client_nr','yearmonth','credit_application','nr_credit_applications'],axis=1)
    y_train= train_df['credit_application']

    X_val= val_df.drop(['client_nr','yearmonth','credit_application','nr_credit_applications'],axis=1)
    y_val = val_df['credit_application']

    X_test= test_df.drop(['client_nr','yearmonth','credit_application','nr_credit_applications'],axis=1)
    y_test= test_df['credit_application']
    #features['TransactionAmt'] = features['TransactionAmt'].apply(np.log10)
    #logging.info("Wrote features to file: {}".format(os.path.join(output_dir, 'features_xgboost.csv')))

    X_train.to_csv(os.path.join(output_dir, 'train_features.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'train_labels.csv'), index=False)
    logging.info("Wrote training files to: {} ".format(os.path.join(output_dir, 'train_features.csv')))

    X_val.to_csv(os.path.join(output_dir, 'val_features.csv'), index=False)
    y_val.to_csv(os.path.join(output_dir, 'val_labels.csv'), index=False)
    logging.info("Wrote validation files to: {} ".format(os.path.join(output_dir, 'val_features.csv')))


    X_test.to_csv(os.path.join(output_dir, 'test_features.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'test_labels.csv'), index=False)
    logging.info("Wrote Test files to: {} ".format(os.path.join(output_dir, 'test_features.csv')))

if __name__ == '__main__':

    logging = get_logger(__name__)

    args = parse_args()

    df = load_data(args.data_dir,
                            args.credit_appications,
                            args.user_features,
                            args.label_col)

    train_df,val_df,test_df = split_data(df, args.label_col, args.train_data_ratio, args.test_data_ratio)

    get_features_and_labels(train_df,val_df,test_df, args.output_dir)

