import pandas as pd
import os



def shrink_csv(input_file, output_file, target_size_mb=25, chunk_size=10000):
    size_limit = target_size_mb * 1024 * 1024  # Convert MB to bytes
    
    # Read the file in chunks
    reader = pd.read_csv(input_file, chunksize=chunk_size)
    
    # Initialize an empty DataFrame
    reduced_df = pd.DataFrame()
    
    for chunk in reader:
        reduced_df = pd.concat([reduced_df, chunk])
        
        # Write the DataFrame to CSV
        reduced_df.to_csv(output_file, index=False)
        
        # Check the size of the file
        if os.path.getsize(output_file) > size_limit:
            # Remove the last chunk if the size exceeds the limit
            reduced_df = reduced_df.iloc[:-chunk_size]
            reduced_df.to_csv(output_file, index=False)
            break

    print(f"Reduced file size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")



# Parameters
input_file = 'Real_Estate_Sales_2001-2021_GL.csv'  # Your large CSV file
output_file = 'Real_Estate.csv'  # The smaller output CSV file

shrink_csv(input_file, output_file)
