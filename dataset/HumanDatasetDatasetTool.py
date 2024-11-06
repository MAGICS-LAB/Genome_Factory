import os
from Bio import SeqIO
from NcbiDatasetCli import NCBIDownloader  # Ensure this module is properly imported

class HumanDataset:
    def __init__(self, taxon_id="9606", download_folder=None, download=True):
        """
        Initializes the HumanDataset class to download and load data for a specified organism.

        Parameters:
        - taxon_id: NCBI Taxonomy ID (e.g., 9606 for Homo sapiens)
        - download_folder: Directory path to store the downloaded data, defaults to "./human"
        - download: Whether to download data; if already downloaded, set to False
        """
        self.taxon_id = taxon_id
        self.download_folder = download_folder if download_folder else "./human"

        # Ensure the download directory exists
        os.makedirs(self.download_folder, exist_ok=True)

        # If download is required, initiate NCBIDownloader
        if download:
            downloader = NCBIDownloader(
                data_type="genome",
                index_type="taxon",
                identifier=self.taxon_id,  # Use NCBI Taxonomy ID 9606 for Homo sapiens
                output_dir=self.download_folder,
                assembly_source="RefSeq",
                include="genome"
            )
            downloader.download_and_extract()

        # Find all .fna files in the download directory
        self.fna_files = self.find_fna_files()

    def find_fna_files(self):
        """
        Finds all .fna file paths in the download directory.

        Returns:
        - A list of all .fna file paths
        """
        fna_files = []
        for root, dirs, files in os.walk(self.download_folder):
            for file in files:
                if file.endswith(".fna"):
                    fna_files.append(os.path.join(root, file))

        if not fna_files:
            raise FileNotFoundError("No .fna files found. Please check if the download was successful.")
        
        return fna_files

    def load_sequences(self):
        """
        Loads genome data from all .fna files and returns brief information.

        Returns:
        - A list of dictionaries, each containing sequence information including id, description, length, and a truncated sequence.
        """
        all_sequences = []

        # Iterate over each .fna file and parse its sequences
        for fna_file in self.fna_files:
            with open(fna_file, "rt") as file:
                for record in SeqIO.parse(file, "fasta"):
                    sequence_info = {
                        "id": record.id,
                        "description": record.description,
                        "length": len(record.seq),
                        "sequence": str(record.seq[:100]) + "..."  # Display the first 100 bases
                    }
                    all_sequences.append(sequence_info)
        
        return all_sequences

# Example usage
if __name__ == "__main__":
    # Initialize the HumanDataset class
    dataset = HumanDataset(taxon_id="9606", download_folder="./human_data", download=True)
    
    # Load and display genome sequence information
    sequences = dataset.load_sequences()
    print(sequences)