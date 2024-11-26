import os
import gdown
import zipfile

def benchmark_DNABERT2(benchmark):
    if benchmark == "GUE":
        # Define download link and output filename
        url = 'https://drive.google.com/uc?id=1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2'
        output = 'GUE_data.zip'

        # Download the file
        print("Downloading data...")
        gdown.download(url, output, quiet=False)

        # Extract the file to the current directory
        print("Extracting data...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('.')

        # Get the data path (current directory)
        data_path = os.path.abspath('.')  # Current directory

        # Run the shell script
        command = f'sh run_dnabert2.sh {data_path}'
        os.system(command)
benchmark_DNABERT2("GUE")

    