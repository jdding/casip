# Data Placement

This package does not include raw data.

Expected local layout for full reproduction:

```text
data/
  Rees46/
    2019-Dec.csv.gz
    2020-Jan.csv.gz
    2020-Feb.csv.gz
    2020-Mar.csv.gz
    2020-Apr.csv.gz
  tmall/
    data_format1/
      user_log_format1.csv
      user_info_format1.csv
  synerise_dataset/
    add_to_cart.parquet
    page_visit.parquet
    product_buy.parquet
    product_properties.parquet
    remove_from_cart.parquet
    search_query.parquet
```

The smoke test does not require these files.
