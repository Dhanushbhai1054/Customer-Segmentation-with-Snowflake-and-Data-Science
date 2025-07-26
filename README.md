# E-Commerce Customer Segmentation with Snowflake and Data Science

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

## Connect to Power BI  

- Use snowflake DB to connect Power BI and load the tables of Customer_data, customer_segments, RFM data tables and chack the data types and load .
  ##  E-Commerce Customer Segmentation Dashboard

This **Power BI dashboard** provides an insightful view of **customer segmentation** for an e-commerce business.  
It uses **RFM (Recency, Frequency, Monetary)** analysis to group customers into meaningful clusters, helping the business identify high-value segments and target them effectively.

![E-Commerce Customer Segmentation Dashboard](https://github.com/Dhanushbhai1054/Customer-Segmentation-with-Snowflake-and-Data-Science/blob/main/Screenshot%202025-07-25%20170927.png)

#  Customer Segmentation Dashboard using Power BI

## Power BI Dashboard Reporting

###  KPIs Included
- **Total Customers**
- **Total Revenue**
- **Average Frequency**
- **Average Recency**
- **Average Monetary**

---

### Slicer with Clusters

#### Reporting
The slicer offers a dropdown to filter data by clusters (0, 1, 2, 3), providing a user-friendly way to focus on specific segments. It enhances analysis by isolating each group’s performance.

####  View
A clean dropdown interface allowing quick switches between clusters for detailed exploration.

---

### Distribution of Customers by Cluster (Pie Chart)

![Pie Chart](https://github.com/Dhanushbhai1054/Customer-Segmentation-with-Snowflake-and-Data-Science/blob/main/Screenshot%202025-07-26%20113308.png)

####  Reporting
This pie chart shows the proportion of customers across the four clusters. Based on SQL averages:
- **Cluster 0** likely dominates due to moderate activity.
- **Cluster 3**, despite being small, contributes significantly due to high frequency.

####  View
A colorful pie chart with segments of varying sizes, highlighting the largest group (likely Cluster 0) and the smallest (likely Cluster 3).

---

###  Average RFM by Cluster (Clustered Bar Chart)

![Clustered Bar Chart](https://github.com/Dhanushbhai1054/Customer-Segmentation-with-Snowflake-and-Data-Science/blob/main/Screenshot%202025-07-26%20113502.png)

#### Reporting
The bar chart compares average Recency, Frequency, and Monetary values:
- **Cluster 3** stands out with:
  - Recency: 2 days  
  - Frequency: 5,918 purchases  
  - Monetary: $42,177.93  
- **Cluster 1** lags with:
  - Recency: 246 days  
  - Monetary: $451.36

####  View
Clustered bars clearly show Cluster 3 towering in Frequency and Monetary, while Cluster 1 ranks lowest across all metrics.

---

###  Scatter Plot of Frequency vs. Monetary by Cluster

#### Reporting
This scatter plot maps Frequency against Monetary, color-coded by cluster.
- **Cluster 3** appears in the upper-right corner with ~5,918 purchases and ~$42,177.93 spend.
- **Cluster 1** appears in the lower-left, indicating lower value customers.

#### View
A scatter plot with distinct colored clusters. Cluster 3 is a standout point, with other clusters spread across lower values.

---

###  Maps of Country by Cluster

![Scatter Plot](https://github.com/Dhanushbhai1054/Customer-Segmentation-with-Snowflake-and-Data-Science/blob/main/Screenshot%202025-07-26%20113820.png)
#### Reporting
The map displays customer distribution by country for each cluster. Given the UK focus of the dataset, most clusters likely concentrate in the UK. Cluster 3 may show a more specific regional pattern due to its smaller size.

####  View
A map with shaded regions, predominantly focused on the UK, varying in intensity by cluster.
![Map](https://github.com/Dhanushbhai1054/Customer-Segmentation-with-Snowflake-and-Data-Science/blob/main/Screenshot%202025-07-26%20113950.png)

---

###  Total Products by Cluster

####  Reporting
This chart or table lists unique products per cluster.
- **Cluster 3**, due to high activity, likely leads in product diversity.
- **Cluster 1** likely shows fewer unique products, indicating lower engagement.

####  View
A table or bar chart, with Cluster 3 potentially at the top for product count.
![Product Table](https://github.com/Dhanushbhai1054/Customer-Segmentation-with-Snowflake-and-Data-Science/blob/main/Screenshot%202025-07-26%20114110.png)
---

##  Findings and Business Impact

###  Cluster Insights

- **Cluster 0**
  - Recency: 40 days  
  - Frequency: 102 purchases  
  - Monetary: $1,670.07  
  - Represents a large, steady, engaged customer base.

- **Cluster 1**
  - Recency: 246 days  
  - Frequency: 27 purchases  
  - Monetary: $451.36  
  - Low engagement; needs reactivation.

- **Cluster 2**
  - Recency: 6 days  
  - Frequency: 642 purchases  
  - Monetary: $37,427.89  
  - Highly loyal and valuable segment.

- **Cluster 3**
  - Recency: 2 days  
  - Frequency: 5,918 purchases  
  - Monetary: $42,177.93  
  - Small but high-value group (VIPs or bulk buyers).

---

## Business Help

These insights allow the e-commerce platform to:

- Prioritize **Cluster 3** for exclusive offers to maintain their $42,177.93 average.
-  Enhance **Cluster 2**’s loyalty with tailored incentives.
-  Re-engage **Cluster 1** with targeted promotions.
-  Leverage **Cluster 0**’s size for scalable marketing campaigns.

---

## Suggestions for the Business

### Marketing Strategies
- Offer a premium membership for **Cluster 3** with personalized deals.
- Introduce loyalty rewards for **Cluster 2** to sustain their high activity.
- Send reactivation emails with discounts to **Cluster 1**.
- Launch seasonal promotions for **Cluster 0** to boost their $1,670.07 average.

### Inventory Management
- Stock a wide range of products for **Cluster 3** based on their diverse purchases.
- Focus on UK-popular items as indicated by the map.

### Geographic Focus
- Concentrate marketing efforts in the **UK**, with plans to explore other regions.

### Technology Use
- Use Power BI dashboards for ongoing segment monitoring and performance tracking.


## Conclusion

This project segmented customers into **four distinct groups**, with **Cluster 3** emerging as the highest-value segment, averaging **$42,177.93 in spend**.  
By leveraging **Snowflake, Python, and Power BI**, we built a powerful dashboard that equips the e-commerce business with data-driven insights to improve engagement, optimize campaigns, and drive revenue.







  
