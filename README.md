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
  - Created database and schema:
    ```sql
    USE ROLE SYSADMIN;
    CREATE DATABASE ecommerce_db;
    USE DATABASE ecommerce_db;
    CREATE SCHEMA analytics;
    CREATE STAGE ecommerce_stage;
-Uploaded data.csv to ecommerce_stage and loaded into customer_data table:

- ``` sql
   CREATE TABLE customer_data (
    InvoiceNo STRING,
    StockCode STRING,
    Description STRING,
    Quantity INT,
    InvoiceDate TIMESTAMP,
    UnitPrice FLOAT,
    CustomerID STRING,
    Country STRING
);
``` sql
COPY INTO customer_data
FROM @ecommerce_stage/data.csv
FILE_FORMAT = (TYPE = CSV SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"' ON_ERROR = 'CONTINUE')
VALIDATION_MODE = RETURN_ERRORS;
