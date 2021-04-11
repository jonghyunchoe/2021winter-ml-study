from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "chicago_crime" dataset
dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

tables = list(client.list_tables(dataset))
for table in tables:
    print(table.table_id)

num_tables = 1

table_ref = dataset_ref.table("crime")
table = client.get_table(table_ref)
print(table.schema)

num_timestamp_fields = 2

fields_for_plotting = ['latitude', 'longitude']

