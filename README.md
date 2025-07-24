# Customer Segmentation with Snowflake and Data Science

## Project Overview
This project, titled **Customer Segmentation with Snowflake and Data Science**, aims to segment customers of an online retail dataset using Recency, Frequency, and Monetary (RFM) analysis combined with K-Means clustering. The analysis leverages Snowflake for scalable data management, Python for computational tasks, and Power BI for interactive visualizations. The objective is to provide actionable insights for marketing strategies, simulating a toy store context. The project was developed as of July 24, 2025.

## Dataset
- **Source**: Online Retail Dataset from Kaggle[](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- **Format**: CSV file (`data.csv`)
- **Features**:
  - `InvoiceNo`: Unique invoice number (string)
  - `StockCode`: Product code (string)
  - `Description`: Product description (string)
  - `Quantity`: Number of items purchased (integer)
  - `InvoiceDate`: Date and time of purchase (timestamp)
  - `UnitPrice`: Price per unit (float)
  - `CustomerID`: Unique customer ID (string)
  - `Country`: Country of purchase (string)
- **Purpose**: Provides transactional data (2010-2011) for RFM-based segmentation.

## Project Structure
- `data.csv`: The raw dataset uploaded to Snowflake.
- `CustomerSegmentation.ipynb`: Jupyter Notebook containing Python code for RFM analysis and clustering.
- `snowflake_credentials.txt`: Secure file with Snowflake connection details (not to be shared publicly).
- `CustomerSegmentationDashboard.pbix`: Power BI file with visualizations.
- `CustomerSegmentationDashboard.pdf`: Exported PDF of the dashboard.
- `CustomerSegmentationReport.docx`: Final report document.
- `README.md`: This file, documenting the project.

- ## Methodology
### 1. Data Setup in Snowflake
- **Rationale**: Established a scalable environment for data storage and processing.
- **Steps**:
   Created database and schema:
   ```sql
    USE ROLE SYSADMIN;
    CREATE DATABASE ecommerce_db;
    USE DATABASE ecommerce_db;
    CREATE SCHEMA analytics;
    CREATE STAGE ecommerce_stage;
- Uploaded data.csv to ecommerce_stage and loaded into customer_data table:

``` sql
   CREATE TABLE customer_data (
    InvoiceNo STRING,
    StockCode STRING,
    Description STRING,
    Quantity INT,
    InvoiceDate TIMESTAMP,
    UnitPrice FLOAT,
    CustomerID STRING,
    Country STRING);
```
- This Dataset contains different timestamp format for column InvoiceDate so follow this code to change timestamp format
  ``` sql
  CREATE OR REPLACE FILE FORMAT my_csv_format
  TYPE = 'CSV'
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'
  SKIP_HEADER = 1
  TIMESTAMP_FORMAT = 'MM/DD/YYYY HH24:MI';
  ```
  
- Now copy the dataset into customer_data Table
  ``` sql
  COPY INTO customer_data
  FROM @ecommerce_stage/data.csv
  FILE_FORMAT = (TYPE = CSV SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"' ON_ERROR = 'CONTINUE')
  VALIDATION_MODE = RETURN_ERRORS;
  ```
## Now we are going to connect our database in jupyter notebook to Run RFM analysis & K- means etc.
 ** First We should install all libriries as following
  ``` python
  pip install snowflake-snowpark-python pandas scikit-learn scipy numpy matplotlib
  pip install snowflake-connector-python snowflake-sqlalchemy sqlalchemy
  pip install --upgrade snowflake-connector-python snowflake-snowpark-python

  ```
 ``` pthon
 import snowflake.snowpark
import pandas as pd
import sklearn
import scipy
import numpy as np
import matplotlib
print("Libraries good!")
 ```
 ** Now we have to Connect our Snowflake Db
 ``` python

from snowflake.snowpark import Session

connection_parameters = {
    "account": "UE17324.eu-west-2.aws",  # e.g., xy12345.us-east-1
    "user": "Dhanush1054",
    "password": "Dhanush@1054",
    "role": "SYSADMIN",
    "warehouse": "COMPUTE_WH",
    "database": "ecommerce_db",
    "schema": "analytics"
}
session = Session.builder.configs(connection_parameters).create()
print("Connected to Snowflake!")

```
** Now Load data from customer_data table
``` python
from snowflake.snowpark.functions import col, datediff, count, sum, max, current_date

# Load data from customer_data table
df = session.table("customer_data")
df = df.with_column("TotalSpend", col("Quantity") * col("UnitPrice"))
```

### We have a challenge after this point , for the best Recency we should not use current time and we should use the max date in the data set . The data set is old and for this follow this code below

- First Check Date Range
``` python
# Check date range
df = session.table("customer_data")
min_date = df.selectExpr("MIN(InvoiceDate) AS min_date").collect()[0]["MIN_DATE"]
max_date = df.selectExpr("MAX(InvoiceDate) AS max_date").collect()[0]["MAX_DATE"]
print(f"Earliest date: {min_date}, Latest date: {max_date}")
```
- Below we are using Latest date as max_date and we should clean the data is there is no null in customer id 
``` python
from snowflake.snowpark.functions import col, datediff, count, sum, lit, max as sf_max

max_date = "2011-12-09"

df = session.table("customer_data")
df = df.filter(col("CustomerID").is_not_null())  # Remove null CustomerIDs
df = df.with_column("TotalSpend", col("Quantity") * col("UnitPrice"))
```
- Calculate RFM
 ``` python
rfm_df = df.group_by("CustomerID").agg(
    datediff("day", sf_max(col("InvoiceDate")), lit(max_date)).alias("Recency"),
    count("InvoiceNo").alias("Frequency"),
    sum("TotalSpend").alias("Monetary")
)

rfm_df.write.save_as_table("rfm_data", mode="overwrite")

print(rfm_df.limit(10).to_pandas())
 ```

-K- Means 

``` python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Reload and clean RFM data
rfm_df = session.table("rfm_data")
rfm_df = rfm_df.filter(col("CustomerID").is_not_null())
rfm_df = rfm_df.filter((col("Frequency") < 10000) & (col("Monetary") < 100000))
rfm_pandas = rfm_df.to_pandas()

# K-Means with corrected data
X = rfm_pandas[["RECENCY", "FREQUENCY", "MONETARY"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm_pandas["cluster"] = kmeans.fit_predict(X_scaled)
```

``` pyhton
# Draw a picture to pick the best number of groups
plt.plot(range(2, 9), inertias, marker="o")
plt.xlabel("Number of Groups")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
```
``` python
# Choose 4 groups (or adjust based on the picture)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm_pandas["Cluster"] = kmeans.fit_predict(X_scaled)
```
 ** Check Quality by using Silhouette Score

 ``` python
# Check quality
score = silhouette_score(X_scaled, rfm_pandas["cluster"])
print(f"Silhouette Score: {score}")
```
- Make sure the score should always >0.5 to be good Quality

- Now save to Snowflake
``` python

# Save to Snowflake (overwrite old customer_segments)
session.write_pandas(rfm_pandas, "customer_segments", auto_create_table=True, overwrite=True)
print(rfm_pandas.head(10))
```
 ### Now come to Snowflake and type following SQL query  to check clusters and Avg Recency , Avg Frequency , Avg Monetary
 ``` sql
SELECT "cluster", 
       ROUND(AVG("RECENCY"), 0) AS AvgRecency, 
       ROUND(AVG("FREQUENCY"), 0) AS AvgFrequency, 
       ROUND(AVG("MONETARY"), 2) AS AvgMonetary
FROM "customer_segments"
GROUP BY "cluster"
ORDER BY "cluster";
```


-Run the Following Query to see the tables which we are going to use in our Power BI for Analysis and Charts 

``` sql
USE DATABASE ecommerce_db;
USE SCHEMA analytics;
SHOW TABLES;
 ```
 - We can see now CUSTOMER_DATA, RFM_DATA, customer_segments ready for analytics.

## COnnect To Power BI





  
