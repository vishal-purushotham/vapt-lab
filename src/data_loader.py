import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from opensearchpy import OpenSearch
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Placeholder for column type conversion logic (Simplified from DataTypeTransformer)
def convert_data_types(df: pd.DataFrame, columns_config: dict) -> pd.DataFrame:
    """Converts DataFrame column types based on config."""
    logger.info("Converting data types...")
    for col, dtype in columns_config.items():
        if col not in df.columns:
            logger.warning(f"Column '{col}' specified in config not found in data.")
            continue
        try:
            if dtype == 'int':
                # Use Int64Dtype for nullable integers
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(pd.Int64Dtype())
            elif dtype == 'float':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            elif dtype == 'datetime':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif dtype == 'bool':
                # Handle various 'truthy'/'falsy' strings robustly
                bool_map = {'true': True, 'false': False, 'yes': True, 'no': False, '1': True, '0': False, 1: True, 0: False}
                # Convert to lower string if not already bool/numeric, then map
                df[col] = df[col].apply(lambda x: str(x).lower() if pd.notna(x) and not isinstance(x, (bool, int, float)) else x).map(bool_map).astype(pd.BooleanDtype())
            elif dtype == 'string':
                df[col] = df[col].astype(pd.StringDtype())
            # Add other type conversions as needed
        except Exception as e:
            logger.error(f"Error converting column '{col}' to {dtype}: {e}")
            # Optionally decide how to handle errors, e.g., leave column as is or fill with NaN
            # df[col] = pd.NA # Or some other strategy
    return df

# Placeholder for aggregation logic (Simplified from DataAggregator)
def aggregate_data(df: pd.DataFrame, time_column: str, agg_config: dict, time_window: str) -> pd.DataFrame:
    """Aggregates DataFrame over a time window."""
    logger.info(f"Aggregating data with window '{time_window}'...")
    if time_column not in df.columns or df[time_column].isnull().all():
        logger.error(f"Time column '{time_column}' not found or empty. Cannot aggregate.")
        return pd.DataFrame() # Return empty DataFrame

    # Ensure time column is datetime type
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    df = df.dropna(subset=[time_column]) # Drop rows where time conversion failed
    if df.empty:
        logger.error("DataFrame is empty after dropping rows with invalid timestamps.")
        return df

    df = df.sort_values(by=time_column).set_index(time_column)

    # Use pandas resampling
    try:
        aggregated_df = df.resample(time_window).agg(agg_config)
    except Exception as e:
        logger.error(f"Error during resampling/aggregation: {e}")
        return pd.DataFrame()

    # Optional: Fill NaNs resulting from aggregation if needed
    # aggregated_df = aggregated_df.fillna(0) # Example: fill with 0

    logger.info(f"Aggregation complete. Result shape: {aggregated_df.shape}")

    # Fill potential missing values resulting from aggregation (e.g., empty intervals)
    aggregated_df = aggregated_df.fillna(0)

    # Return the dataframe with the timestamp as the index
    return aggregated_df

# --- Wazuh/OpenSearch Interaction (Simplified from WazuhDataIngestor) ---
def get_index_name_for_date(date_str: str, prefix: str = "wazuh-alerts-4.x-") -> str:
    """Generates the Wazuh index name for a given date."""
    try:
        # Expecting date_str in YYYY-MM-DD format
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return f"{prefix}{dt.strftime('%Y.%m.%d')}"
    except ValueError:
        logger.error(f"Invalid date format: {date_str}. Expected YYYY-MM-DD.")
        raise

def fetch_data_from_os(client: OpenSearch, index_name: str, start_time: datetime, end_time: datetime, columns: list) -> list:
    """Fetches documents from OpenSearch within a time range using pagination."""
    all_docs = []
    try:
        # Format timestamps for OpenSearch query (ISO format with 'Z' for UTC)
        start_iso = start_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        end_iso = end_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        logger.info(f"Fetching data from index '{index_name}' between {start_iso} and {end_iso}")

        # Use the Point in Time (PIT) API for stable pagination
        pit_response = client.create_point_in_time(index=index_name, keep_alive="1m")
        pit_id = pit_response['pit_id']

        query = {
            "range": {
                "timestamp": { 
                    "gte": start_iso,
                    "lt": end_iso,
                    "format": "strict_date_optional_time||epoch_millis"
                }
            }
        }

        search_after = None
        while True:
            body = {
                "size": 1000, # Adjust batch size as needed
                "query": query,
                "pit": {
                    "id": pit_id,
                    "keep_alive": "1m"
                },
                "_source": columns, # Request only specific columns
                "sort": [
                    {"timestamp": "asc"}, # Sort by timestamp
                    {"_doc": "asc"}      # Tie-breaker for stability
                ]
            }
            if search_after:
                body["search_after"] = search_after

            response = client.search(body=body)
            hits = response['hits']['hits']
            if not hits:
                break

            all_docs.extend([hit['_source'] for hit in hits])
            search_after = hits[-1]['sort']

        # Close the Point in Time context
        client.delete_point_in_time(body={'pit_id': pit_id})
        logger.info(f"Fetched {len(all_docs)} documents.")

    except Exception as e:
        logger.error(f"Error fetching data from OpenSearch: {e}")
        # Attempt to close PIT if it exists, even on error
        try:
            if 'pit_id' in locals():
                client.delete_point_in_time(body={'pit_id': pit_id})
        except Exception as pit_e:
            logger.error(f"Error closing PIT context: {pit_e}")
        raise # Re-raise the original error

    return all_docs

# --- Main Function --- #
def load_and_prepare_data(
    os_host: str,
    os_port: int,
    os_auth: tuple,
    date: str, # Date for which to fetch data (YYYY-MM-DD)
    start_time_str: str, # Start time (HH:MM:SS)
    end_time_str: str,   # End time (HH:MM:SS)
    columns_config: dict, # { 'col_name': 'dtype', ... }
    aggregation_config: dict, # { 'col_name': 'agg_method', ... }
    aggregation_window: str, # e.g., '1min', '5T', '1H'
    time_column: str = 'timestamp',
    index_prefix: str = "wazuh-alerts-4.x-",
    use_ssl: bool = True,
    verify_certs: bool = False,
    ssl_show_warn: bool = False
) -> pd.DataFrame:
    """Loads data from Wazuh/OpenSearch, processes types, and aggregates features."""

    try:
        logger.info(f"Connecting to OpenSearch at {os_host}:{os_port}")
        client = OpenSearch(
            hosts=[{'host': os_host, 'port': os_port}],
            http_compress=True,
            http_auth=os_auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_assert_hostname=not verify_certs, # Assert hostname if not verifying certs
            ssl_show_warn=ssl_show_warn,
            timeout=60 # Add a timeout
        )
        if not client.ping():
            raise ConnectionError("Failed to connect to OpenSearch.")
        logger.info("Connection successful.")

        index_name = get_index_name_for_date(date, index_prefix)
        logger.info(f"Target index: {index_name}")

        # Combine date with time strings
        start_dt = datetime.strptime(f"{date} {start_time_str}", '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(f"{date} {end_time_str}", '%Y-%m-%d %H:%M:%S')

        # Fetch raw data
        raw_docs = fetch_data_from_os(client, index_name, start_dt, end_dt, list(columns_config.keys()))

        if not raw_docs:
            logger.warning("No documents found for the specified time range.")
            return pd.DataFrame() # Return empty DataFrame

        # Convert to DataFrame
        df = pd.json_normalize(raw_docs)

        # Convert data types
        df = convert_data_types(df, columns_config)

        # Aggregate data
        aggregated_df = aggregate_data(df, time_column, aggregation_config, aggregation_window)

        return aggregated_df

    except ConnectionError as ce:
        logger.error(f"OpenSearch connection error: {ce}")
    except Exception as e:
        logger.error(f"An error occurred during data loading and preparation: {e}", exc_info=True)

    return pd.DataFrame() # Return empty DataFrame on error

# Example Usage (comment out when using as a module):
# if __name__ == '__main__':
#     # Load config from a file or define here
#     wazuh_config = {
#         'host': 'localhost', # Replace with your Wazuh/OS host
#         'port': 9200,      # Replace with your Wazuh/OS port
#         'user': 'admin',      # Replace with your user
#         'password': 'SecretPassword' # Replace with your password
#     }
#
#     # Define which columns to fetch and their types
#     cols_config = {
#         'timestamp': 'datetime',
#         'rule.level': 'int',
#         'agent.ip': 'string',
#         # Add other relevant columns for supply chain attacks
#         'data.command': 'string',
#         'data.srcip': 'string',
#         'data.dstip': 'string',
#         'data.protocol': 'string',
#         'data.srcport': 'int',
#         'data.dstport': 'int',
#     }
#
#     # Define how to aggregate numeric columns
#     agg_config = {
#         'rule.level': ['mean', 'max', 'count'], # Example: mean/max level, count alerts
#         'data.srcport': 'nunique', # Example: count unique source ports
#         'data.dstport': 'nunique', # Example: count unique dest ports
#         # Add aggregation for other numeric columns
#     }
#
#     target_date = '2024-01-15' # Replace with a date with data
#     start_t = '00:00:00'
#     end_t = '23:59:59'
#     agg_window = '5min' # Aggregate in 5-minute windows
#
#     prepared_data = load_and_prepare_data(
#         os_host=wazuh_config['host'],
#         os_port=wazuh_config['port'],
#         os_auth=(wazuh_config['user'], wazuh_config['password']),
#         date=target_date,
#         start_time_str=start_t,
#         end_time_str=end_t,
#         columns_config=cols_config,
#         aggregation_config=agg_config,
#         aggregation_window=agg_window,
#         verify_certs=False, # Set to True in production if using valid certs
#         ssl_show_warn=False
#     )
#
#     if not prepared_data.empty:
#         print("Data loaded and prepared successfully:")
#         print(prepared_data.head())
#         print(prepared_data.info())
#     else:
#         print("Failed to load or prepare data.")
