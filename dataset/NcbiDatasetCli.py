import hashlib
import logging
import os
import subprocess
import zipfile
from typing import Dict, List, Optional


class NCBIDownloader:
    def __init__(self,data_type: str,index_type: str,identifier: str,output_dir: str,assembly_source: str = "RefSeq",include: str = "genome",) -> None:
        """
        Initializes the download class, allowing users to control download commands by passing parameters.

        Parameters:
        - data_type: Data type (e.g., genome, gene, virus)
        - index_type: Index type (e.g., taxon, accession, gene-id)
        - identifier: Specific value of the index (e.g., species name, gene ID, accession number)
        - output_dir: Directory to store downloaded files
        - assembly_source: Data source (default is RefSeq)
        - include: Content to include (default is genome)
        """
        self.data_type = data_type
        self.index_type = index_type
        self.identifier = identifier
        self.output_dir = output_dir
        self.assembly_source = assembly_source
        self.include = include

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Build a secure filename and path
        self.zip_filename = f"{self.identifier}.zip"
        self.zip_filepath = os.path.join(output_dir, self.zip_filename)

        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def build_download_command(self) -> str:
        """
        Builds the download command, allowing flexible parameter input to control download behavior.

        Returns:
        - Complete download command string
        """
        if self.data_type == "gene":
            command = (
                f"datasets download {self.data_type} {self.index_type} {self.identifier} "
                f"--include {self.include} "
                f"--filename {str(self.zip_filepath)} "
                "--no-progressbar "
            )
        elif self.data_type == "genome":
            command = (
                f"datasets download {self.data_type} {self.index_type} {self.identifier} "
                f"--assembly-source {self.assembly_source} "
                f"--include {self.include} "
                f"--filename {str(self.zip_filepath)} "
                "--no-progressbar "
            )
        else:
            self.logger.error("Unsupported data type")
            raise ValueError("Unsupported data type")

        return command

    def download_and_extract(self) -> None:
        """
        Downloads and extracts content for the specified data type.
        """
        try:
            # Get the constructed download command
            download_command = self.build_download_command()

            # Execute download command
            self.logger.info(f"Running command: {download_command}")
            subprocess.run(download_command, shell=True, check=True)
            self.logger.info(f"Download of {self.identifier} successful!")

            # Unzip file
            with zipfile.ZipFile(self.zip_filepath, "r") as zip_ref:
                zip_ref.extractall(self.output_dir)
            self.logger.info(f"File extraction for {self.identifier} complete.")

            # Delete zip file after extraction
            if os.path.exists(self.zip_filepath):
                os.remove(self.zip_filepath)
                self.logger.info(f"{self.zip_filename} deleted after extraction.")

            # Verify MD5 checksum
            md5_file = os.path.join(self.output_dir, "md5sum.txt")
            if os.path.exists(md5_file):
                self.validate_md5(md5_file)
            else:
                self.logger.warning(
                    "MD5 checksum file not found, skipping MD5 validation."
                )

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Download of {self.identifier} failed. Error: {e}")
        except zipfile.BadZipFile:
            self.logger.error(
                f"Failed to extract {self.zip_filepath}. The file may be corrupted."
            )
        except Exception as e:
            self.logger.exception(
                f"An error occurred while processing {self.identifier}: {e}"
            )

    def validate_md5(self, md5_file: str) -> None:
        """
        Validates the MD5 checksum of files.

        Parameters:
        - md5_file: Path to the file containing MD5 checksums
        """
        self.logger.info(f"Starting MD5 validation using file: {md5_file}")

        # Read md5sum.txt line by line and validate each file
        with open(md5_file, "r") as f:
            for line in f:
                md5_hash, relative_path = line.strip().split("  ")
                file_path = os.path.join(self.output_dir, relative_path)

                if os.path.exists(file_path):
                    calculated_md5 = self.calculate_md5(file_path)
                    if calculated_md5 == md5_hash:
                        self.logger.info(f"{relative_path}: OK")
                    else:
                        self.logger.error(f"{relative_path}: MD5 checksum mismatch")
                        exit(1)
                else:
                    self.logger.error(f"{relative_path}: File not found")
                    exit(1)

        self.logger.info("MD5 validation complete, all files verified.")

    @staticmethod
    def calculate_md5(file_path: str) -> str:
        """
        Calculates the MD5 checksum of a file.

        Parameters:
        - file_path: Path to the file

        Returns:
        - MD5 value of the file
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()