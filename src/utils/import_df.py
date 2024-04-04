import pandas as pd



def import_df(csv_url:str,
              sample:bool = False,
              sample_size:float = 0.1,
              random_state:int = 42) -> pd.DataFrame:
    """
    Import a DataFrame from a CSV file.
    
    Parameters:
        csv_url (str): The URL or file path of the CSV file to import.
        sample (bool, optional): Whether to sample the DataFrame or not. Defaults to False.
        sample_size (float, optional): The fraction of the DataFrame to sample if `sample` is True. Defaults to 0.1.
        random_state (int, optional): The random seed for reproducible sampling. Defaults to 42.
    
    Returns:
        pd.DataFrame: The imported DataFrame.
    """
    # import DF
    df = pd.read_csv(csv_url, low_memory=False)

    # Sampling if true
    if sample == True:
        df = df.sample(frac=sample_size, random_state=random_state)
        df.reset_index(inplace=True, drop=True)
        percent_sample_size = sample_size * 100
        print(f"DF sampled with {percent_sample_size}% from original dataset, shape:", df.shape)
        print(f"Unique species in sampled DF: {df['species'].nunique()}")


    else:
        print("DF loaded with shape:", df.shape)
        print(f"Unique species in DF: {df['species'].nunique()}")


    return df