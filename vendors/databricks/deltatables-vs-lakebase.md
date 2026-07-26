# Delta Tables vs Lakebase in Databricks

## Overview

Delta Tables and Lakebase are complementary components of the Databricks platform. Delta Tables handle large-scale analytics (OLAP), while Lakebase provides a transactional database for applications and AI agents (OLTP). Both are governed by Unity Catalog.

## Delta Tables

Delta Tables use the Delta Lake format (Parquet files + a transaction log) stored on cloud object storage (S3, ADLS, GCS). They are designed for analytical workloads: querying and transforming large datasets, running ETL pipelines, powering dashboards, and supporting machine learning.

- **Latency:** seconds to minutes per query
- **Scale:** petabyte-scale
- **Query pattern:** aggregations, joins, and scans over large datasets
- **Typical user:** data engineers, analysts, data scientists
- **Storage:** cloud object storage

## Lakebase

Lakebase is a fully managed, serverless PostgreSQL database built into the Databricks platform. It is designed for operational workloads: serving AI agents, powering web applications, and handling high-frequency reads and writes on individual records.

- **Latency:** milliseconds per operation
- **Scale:** up to 2 TB per instance; 1,000 concurrent connections
- **Query pattern:** single-row inserts, updates, and lookups
- **Typical user:** application developers, AI agents
- **Storage:** managed Postgres storage engine

## Key Differences

| Dimension         | Delta Tables                    | Lakebase                          |
| ----------------- | ------------------------------- | --------------------------------- |
| Workload type     | OLAP (analytics)                | OLTP (transactional)              |
| Latency           | Seconds to minutes              | Milliseconds                      |
| Storage backend   | Cloud object storage            | Managed Postgres                  |
| Max data size     | Petabyte-scale                  | 2 TB per instance                 |
| Query style       | Batch aggregations and scans    | Single-row reads and writes       |
| Primary audience  | Data engineers and analysts     | App developers and AI agents      |
| Wire protocol     | Spark / SQL Warehouse           | Standard Postgres (psycopg2, etc.)|

## How They Work Together

Lakebase's **Synced Tables** feature automatically mirrors Delta Tables into Lakebase, so analytical data built in pipelines can be served to applications at low latency without a separate ETL process. Both systems share Unity Catalog for unified governance, access control, and data lineage.

In short: Delta Tables are the analytical backbone; Lakebase is the operational database layer: and they are designed to be used side by side.

---

## Appendix: Key Terms Explained

**Database**: A system for storing, organising, and retrieving data. Think of it like a very powerful, structured spreadsheet that can hold millions or billions of rows and let many people (or applications) read and update data at the same time.

**OLAP (Online Analytical Processing)**: A style of database work focused on answering big-picture questions: "What were total sales last quarter?", "Which region grew fastest?" These queries scan huge amounts of data to produce summaries, charts, and reports. Speed per individual record matters less than the ability to crunch large volumes.

**OLTP (Online Transactional Processing)**: A style of database work focused on fast, small operations: "Add this item to the customer's cart", "Update the user's address", "Record this payment." Each operation touches one or a few rows, but it needs to happen in milliseconds because a real user or application is waiting for the result.

**Transactional**: Describes a system that guarantees each operation either fully succeeds or fully fails, with nothing left in a half-finished state. If you transfer money between two bank accounts, a transactional system ensures the debit and credit both happen: or neither does. This property is often summarised by the acronym ACID (Atomicity, Consistency, Isolation, Durability).

**Latency**: The time it takes for a system to respond to a request. Low latency (milliseconds) means fast responses, which is essential for apps and agents. Higher latency (seconds to minutes) is acceptable when running large analytical queries.

**ETL (Extract, Transform, Load)**: The process of pulling data out of one system, cleaning or reshaping it, and loading it into another. For example, extracting sales records from a web app's database, calculating daily totals, and loading the results into an analytics platform.

### Storage Formats

**Delta Lake format (Parquet + transaction log)**: The file format used by Delta Tables. Data is stored in Parquet files, which are columnar (optimised for reading specific columns across many rows very quickly). Alongside the data sits a transaction log that tracks every change, enabling features like time-travel (querying data as it looked yesterday) and reliable concurrent writes. These files live on cloud object storage: essentially folders of files in services like Amazon S3, Azure Data Lake Storage, or Google Cloud Storage.

**PostgreSQL (Postgres)**: A widely-used, open-source relational database that has been around since the 1980s. It stores data in traditional rows and tables and is optimised for fast, transactional operations: reading and writing individual records quickly. Lakebase is built on PostgreSQL, which means any tool, library, or programming language that already knows how to talk to Postgres can connect to Lakebase without changes. Think of Parquet/Delta Lake as the format built for scanning warehouses of data, and PostgreSQL as the engine built for serving one customer at a time, very quickly.

### Other Terms

**Unity Catalog**: Databricks' governance layer. It acts as a central directory that knows about every table, file, model, and user permission across the platform, so security rules and data lineage are managed in one place rather than separately for each tool.

**Synced Tables**: A Databricks feature that automatically copies data from a Delta Table into a Lakebase Postgres table and keeps it up to date. This lets analytical data become instantly available to apps and agents without building a separate pipeline.

**Cloud object storage**: File storage services provided by cloud platforms (Amazon S3, Azure Data Lake Storage, Google Cloud Storage). They can hold virtually unlimited amounts of data at low cost, but reading individual small records from them is slower than reading from a traditional database.