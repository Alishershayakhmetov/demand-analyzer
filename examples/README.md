# Example files for the web app (open data, not M5)

The CSV files here are generated, not stored in git (`python -I -X utf8 scripts/18_external_examples.py` downloads the sources into
`data/external/` and writes them; `pip install openpyxl` is needed once for the Online Retail file). In the app choose
"Upload CSV" and press one of the example buttons, or pick a file with the file button and map the columns yourself.

| File | Source (no account needed) | Columns to map | What it shows |
|---|---|---|---|
| `bakery_daily.csv` | Kaggle "Transactions from a bakery" (BreadBasket), GitHub mirror yortos/bakery | date, item, sales | a small shop, very low volumes (median 0.04 units per item and day), 159 days |
| `hierarchical_sales.csv` | autogluon/fev_datasets `hierarchical_sales` (Hugging Face) | date, item, sales | 118 daily series over 5 years, volumes close to M5 |
| `rohlik_prague1_top300.csv` | Rohlik Group grocery e-commerce, autogluon/fev_datasets `rohlik_sales`, warehouse Prague_1, 300 best sellers | date, item, sales, price | high volumes (median 129 per day), a price column, missing days |
| `online_retail_ii_top300_invoice_lines.csv` | UCI Online Retail II (CC BY 4.0), UK gift wholesaler, 300 best-selling products | InvoiceDate, StockCode, Quantity, Price | raw invoice lines: many lines per day, timestamps, returns as negative quantities |
| `rossmann_300_stores_euro.csv` | Rossmann Store Sales, autogluon/fev_datasets `rossmann`, first 300 stores | date, store, sales | store-level sales in euro: a scale stress test, not unit sales |

Check each source's terms before using the data outside coursework. Results of forecasting these files are in the main README
(section "Check on other data").
