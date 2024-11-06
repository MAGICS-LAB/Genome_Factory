# NcbiFtp.py

from ftplib import FTP
from pathlib import Path
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class NcbiFtpDownloader:
    def __init__(self, ftp_url="ftp.ncbi.nlm.nih.gov"):
        # Set the FTP URL for the NCBI server
        self.ftp_url = ftp_url

    def connect(self):
        """Connect to the NCBI FTP server."""
        ftp = FTP(self.ftp_url)  # Initialize FTP connection to specified URL
        ftp.login()  # Log in to the FTP server (anonymous login)
        return ftp  # Return the connected FTP instance

    def download_files(self, download_folder: str, ftp_directory: str, file_ext=".gz"):
        """Download files with the specified extension from a given FTP directory."""
        ftp = self.connect()  # Establish a connection to the FTP server

        # Navigate to the specified directory on the FTP server
        ftp.cwd(ftp_directory)

        # Get a list of all files in the directory that match the specified file extension
        files = [f for f in ftp.nlst() if f.endswith(file_ext)]

        # Create the local download folder if it doesn't exist
        download_folder = Path(download_folder)
        os.makedirs(download_folder, exist_ok=True)

        # Loop through each file in the list and download it
        for filename in files:
            download_file = download_folder.joinpath(filename)  # Local file path for download
            # Check if the file already exists to avoid re-downloading
            if not os.path.exists(download_file):
                with open(download_file, "wb") as fp:  # Open the file in binary write mode
                    # Download the file from the FTP server and write its content
                    ftp.retrbinary(f"RETR {filename}", fp.write)
                print(f"Downloaded: {filename}")  # Notify about successful download
                time.sleep(1)  # Wait 1 second between downloads to reduce server load
            else:
                print(f"File already exists: {filename}")  # Notify if the file already exists

        ftp.quit()  # Close the FTP connection

    def download_concurrent(self, species_list, download_folder_base, ftp_directory_base, file_ext=".gz"):
        """Download files for multiple species concurrently."""
        # Set the number of threads based on available CPU cores
        max_workers = os.cpu_count()
        print(f"Using {max_workers} threads based on CPU core count")

        # Use a ThreadPoolExecutor to download files concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit a download task for each species in the list
            futures = {
                executor.submit(
                    self.download_files, 
                    f"{download_folder_base}/{specie}", 
                    f"{ftp_directory_base}/{specie}/", 
                    file_ext
                ): specie for specie in species_list
            }
            # Process each completed future as they finish
            for future in as_completed(futures):
                specie = futures[future]
                try:
                    future.result()  # Retrieve result or exception
                except Exception as exc:
                    print(f"{specie} generated an exception: {exc}")  # Handle any errors during download