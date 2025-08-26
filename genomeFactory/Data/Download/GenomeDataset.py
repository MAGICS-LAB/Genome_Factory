import os
import json
from Bio import SeqIO
from genomeFactory.Data.Download.NcbiDatasetCli import NCBIDownloader  # Ensure this module is properly installed

class GenomeDataset:
    def __init__(self, species: str, download_folder: str = None, download: bool = True):
        """
        Initializes the GenomeDataset class to download and load data for a specified organism.
        Parameters:
        - species: Species name (e.g., "Homo sapiens")
        - download_folder: Directory path to store the downloaded data, defaults to "./{species}"
        - download: Whether to download data; if already downloaded, set to False
        """
        # Build an absolute path to Datasets_species_taxonid_dict.json
        dir_here = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(dir_here, "Datasets_species_taxonid_dict.json")

        # Load species-to-taxon_id mapping from JSON file
        with open(json_path, "r", encoding="utf-8") as f:
            species_taxon_dict = json.load(f)

        # Map species to taxon_id
        self.taxon_id = species_taxon_dict.get(species)
        if not self.taxon_id:
            raise ValueError(f"Species '{species}' not found in the dataset JSON file.")

        # Set the download folder to the species name if not specified
        self.download_folder = download_folder if download_folder else f"./{species.replace(' ', '_')}"
        # Ensure the download directory exists
        os.makedirs(self.download_folder, exist_ok=True)

        # If download is required, initiate NCBIDownloader
        if download:
            downloader = NCBIDownloader(
                data_type="genome",
                index_type="taxon",
                identifier=self.taxon_id,  # Use the mapped taxon_id
                output_dir=self.download_folder,
                assembly_source="RefSeq",
                include="genome"
            )
            downloader.download_and_extract()

        # Find all .fna files in the download directory
        self.fna_files = self.find_fna_files()

    def find_fna_files(self) -> list[str]:
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