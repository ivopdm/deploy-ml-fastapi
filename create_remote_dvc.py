# Create a remote DVC remote pointing to your S3 bucket and commit the data
# to the remote
dvc remote add -d myremote s3://project3-nd0821
dvc push

# commit census.csv file to DVC remote pointing to your S3 bucket
dvc add data/census.csv
dvc push

# commit census_cat.csv file new version to DVC


