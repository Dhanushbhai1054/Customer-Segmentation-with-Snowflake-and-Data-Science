import snowflake.snowpark
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, datediff, count, sum, lit, max as sf_max
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Print library versions
print("Libraries loaded successfully!")

# Snowflake connection parameters
connection_parameters = {
    "account": "UE17324.eu-west-2.aws",
    "user": "Dhanush1054",
    "password": "Dhanush@1054",
    "role": "SYSADMIN",
    "warehouse": "COMPUTE_WH",
    "database": "ecommerce_db",
    "schema": "analytics"
}

# Create Snowflake session
session = Session.builder.configs(connection_parameters).create()
print("Connected to Snowflake!")

# Load data from customer_data table
df = session.table("customer_data")
df = df.with_column("TotalSpend", col("Quantity") * col("UnitPrice"))

# Check date range
min_date = df.selectExpr("MIN(InvoiceDate) AS min_date").collect()[0]["MIN_DATE"]
max_date = df.selectExpr("MAX(InvoiceDate) AS max_date").collect()[0]["MAX_DATE"]
print(f"Earliest date: {min_date}, Latest date: {max_date}")

# Use the latest date from the dataset
max_date = "2011-12-09"
df = df.filter(col("CustomerID").is_not_null())
df = df.with_column("TotalSpend", col("Quantity") * col("UnitPrice"))

# Calculate RFM metrics
rfm_df = df.group_by("CustomerID").agg(
    datediff("day", sf_max(col("InvoiceDate")), lit(max_date)).alias("Recency"),
    count("InvoiceNo").alias("Frequency"),
    sum("TotalSpend").alias("Monetary")
)
rfm_df.write.save_as_table("rfm_data", mode="overwrite")
print(rfm_df.limit(10).to_pandas())

# Reload and clean RFM data for clustering
rfm_df = session.table("rfm_data")
rfm_df = rfm_df.filter(col("CustomerID").is_not_null())
rfm_df = rfm_df.filter((col("Frequency") < 10000) & (col("Monetary") < 100000))
rfm_pandas = rfm_df.to_pandas()

# K-Means clustering
X = rfm_pandas[["RECENCY", "FREQUENCY", "MONETARY"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to determine optimal number of clusters
inertias = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(range(2, 9), inertias, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal Clusters")
plt.show()

# Apply K-Means with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
rfm_pandas["Cluster"] = kmeans.fit_predict(X_scaled)

# Check clustering quality with silhouette score
score = silhouette_score(X_scaled, rfm_pandas["Cluster"])
print(f"Silhouette Score: {score}")

# Save clustered data to Snowflake
session.write_pandas(rfm_pandas, "customer_segments", auto_create_table=True, overwrite=True)
print(rfm_pandas.head(10))
