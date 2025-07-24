USE ROLE SYSADMIN;
CREATE DATABASE ecommerce_db;
USE DATABASE ecommerce_db;
CREATE SCHEMA analytics;
CREATE STAGE ecommerce_stage;

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

CREATE OR REPLACE FILE FORMAT my_csv_format
    TYPE = 'CSV'
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    SKIP_HEADER = 1
    TIMESTAMP_FORMAT = 'MM/DD/YYYY HH24:MI';

COPY INTO customer_data
FROM @ecommerce_stage/data.csv
FILE_FORMAT = (TYPE = CSV SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"' ON_ERROR = 'CONTINUE')
VALIDATION_MODE = RETURN_ERRORS;

SELECT "cluster", 
       ROUND(AVG("RECENCY"), 0) AS AvgRecency, 
       ROUND(AVG("FREQUENCY"), 0) AS AvgFrequency, 
       ROUND(AVG("MONETARY"), 2) AS AvgMonetary
FROM "customer_segments"
GROUP BY "cluster"
ORDER BY "cluster";

USE DATABASE ecommerce_db;
USE SCHEMA analytics;
SHOW TABLES;
