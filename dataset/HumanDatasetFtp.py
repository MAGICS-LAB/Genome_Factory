from NcbiFtp import NcbiFtpDownloader
import gzip
from Bio import SeqIO

class HumanDataset:
    def __init__(self, download_folder=None, download=True):
        # Set the download folder to a custom path if provided; otherwise, default to "./human"
        self.download_folder = download_folder if download_folder else "./human"
        
        # Define the file name of the human genome dataset
        self.file_name = "GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
        
        # Construct the full file path based on the download folder and file name
        self.file_path = f"{self.download_folder}/{self.file_name}"
        
        # Download the file if the download flag is True
        if download:
            downloader = NcbiFtpDownloader()  # Instantiate the downloader
            downloader.download_files(
                download_folder=self.download_folder, 
                ftp_directory="/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/", 
                file_ext="p14_genomic.fna.gz"  # File extension to download
            )
        
    def load_sequences(self):
        # Initialize a list to store sequence information
        sequences = []  
        
        # Open the compressed .fna.gz file and parse it as a FASTA file
        with gzip.open(self.file_path, "rt") as fna_gz_file:
            for record in SeqIO.parse(fna_gz_file, "fasta"):
                # Collect relevant sequence information, truncating the sequence for preview
                sequence_info = {
                    "id": record.id,  # Sequence identifier
                    "description": record.description,  # Description of the sequence
                    "length": len(record.seq),  # Length of the sequence
                    "sequence": str(record.seq[:100]) + "..."  # First 100 bases with truncation
                }
                sequences.append(sequence_info)  # Append to the list of sequences
        return sequences  # Return the list of sequences with metadata
if __name__ == "__main__":
    # Usage example with a custom folder
    dataset = HumanDataset("custom_folder", download=True)  # Create dataset instance with custom folder
    sequences = dataset.load_sequences()  # Load and parse sequences from the dataset
    print(sequences)  # Print the sequence information