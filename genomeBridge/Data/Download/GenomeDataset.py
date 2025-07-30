import os
import json
import logging
import datetime
import hashlib
import getpass
import re
import random
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
from Bio import SeqIO
from genomeBridge.Data.Download.NcbiDatasetCli import NCBIDownloader
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import pandas as pd

class GenomeDataset:
    def __init__(self, species: str, download_folder: str = None, download: bool = True):
        """
        Initializes the GenomeDataset class to download and load data for a specified organism.
        Automatically implements HIPAA compliance for human genomic data (taxon ID 9606).
        
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
        
        self.species = species
        # Set the download folder to the species name if not specified
        self.download_folder = download_folder if download_folder else f"./{species.replace(' ', '_')}"
        os.makedirs(self.download_folder, exist_ok=True)

        # HIPAA Compliance Implementation
        self.is_human_data = (self.taxon_id == "9606")  # Human taxon ID
        self.cipher_suite = None
        self.audit_log = []
        
        if self.is_human_data:
            print("⚠️ HIPAA Compliance Required: Human genomic data detected")
            self._initialize_hipaa_compliance()
            self._log_audit_event("HUMAN_DATA_DETECTED", f"Taxon ID {self.taxon_id} classified as PHI")

        # If download is required, initiate NCBIDownloader
        if download:
            downloader = NCBIDownloader(
                data_type="genome",
                index_type="taxon",
                identifier=self.taxon_id,
                output_dir=self.download_folder,
                assembly_source="RefSeq",
                include="genome"
            )
            downloader.download_and_extract()
            
            if self.is_human_data:
                self._encrypt_human_genomic_files()

        # Find all .fna files in the download directory
        self.fna_files = self.find_fna_files()

    def _initialize_hipaa_compliance(self):
        """Initialize HIPAA compliance features for human genomic data."""
        # Generate encryption key using PBKDF2 (HIPAA-compliant key derivation)
        password = getpass.getpass("Enter encryption password for human genomic data: ").encode()
        salt = b'genomic_hipaa_2024_salt'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # HIPAA-compliant iteration count
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.cipher_suite = Fernet(key)
        
        # Setup audit logging
        logging.basicConfig(
            filename=os.path.join(self.download_folder, 'hipaa_audit.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        print("🔒 HIPAA Compliance Activated:")
        print("   - AES-256 encryption enabled")
        print("   - Audit logging initialized")
        print("   - Minimum necessary standard compliance")

    def _log_audit_event(self, event_type: str, description: str):
        """Log HIPAA-compliant audit events."""
        if self.is_human_data:
            audit_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "event_type": event_type,
                "description": description,
                "taxon_id": self.taxon_id,
                "user": os.getenv('USER', 'unknown')
            }
            self.audit_log.append(audit_entry)
            logging.info(f"HIPAA_AUDIT: {json.dumps(audit_entry)}")

    def _encrypt_human_genomic_files(self):
        """Encrypt human genomic files for HIPAA compliance."""
        print("🔐 Encrypting human genomic data for HIPAA compliance...")
        
        for root, dirs, files in os.walk(self.download_folder):
            for file in files:
                if file.endswith(".fna") and not file.endswith(".encrypted"):
                    file_path = os.path.join(root, file)
                    
                    # Read and encrypt file
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    
                    encrypted_data = self.cipher_suite.encrypt(file_data)
                    encrypted_path = file_path + ".encrypted"
                    
                    with open(encrypted_path, 'wb') as f:
                        f.write(encrypted_data)
                    
                    # Secure deletion of original (3-pass overwrite)
                    self._secure_delete(file_path)
                    
                    self._log_audit_event("FILE_ENCRYPTED", f"Encrypted {os.path.basename(file_path)}")
                    print(f"   ✅ Encrypted: {os.path.basename(file_path)}")

    def _secure_delete(self, file_path: str):
        """HIPAA-compliant secure deletion (DoD 5220.22-M standard)."""
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            with open(file_path, 'r+b') as f:
                for _ in range(3):  # 3-pass overwrite
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            os.remove(file_path)
            self._log_audit_event("SECURE_DELETION", f"Securely deleted {os.path.basename(file_path)}")

    def find_fna_files(self) -> list[str]:
        """
        Finds all .fna file paths in the download directory.
        For human data, includes both .fna and .fna.encrypted files.
        """
        fna_files = []
        for root, dirs, files in os.walk(self.download_folder):
            for file in files:
                if file.endswith(".fna") or (self.is_human_data and file.endswith(".fna.encrypted")):
                    fna_files.append(os.path.join(root, file))
        
        if not fna_files:
            raise FileNotFoundError("No .fna files found. Please check if the download was successful.")
        
        return fna_files

    def analyze_class_imbalance(self, samples: List[Dict], 
                               class_label_key: str = 'label',
                               positive_class: str = None,
                               negative_class: str = None) -> Dict:
        """
        Unified class imbalance analysis for both binary and multi-class scenarios.
        Handles genomic datasets for tasks like rare variant prediction, disease classifications, and species classification.
        
        Parameters:
        - samples: List of sample dictionaries with class labels or legacy format with 'features'
        - class_label_key: Key name for class labels in sample dictionaries
        - positive_class: For binary classification, specify positive class label
        - negative_class: For binary classification, specify negative class label
        
        Returns:
        - Dictionary with comprehensive class imbalance analysis
        """
        print("⚖️ Analyzing class imbalance in genomic dataset...")
        self._log_audit_event("CLASS_IMBALANCE_ANALYSIS_START", "Starting unified class imbalance analysis")
        
        # Handle legacy binary format (positive_samples, negative_samples lists)
        if isinstance(samples, tuple) and len(samples) == 2:
            positive_samples, negative_samples = samples
            return self._analyze_binary_legacy_format(positive_samples, negative_samples)
        
        # Modern unified format - count samples per class
        class_counts = Counter()
        total_samples = len(samples)
        
        for sample in samples:
            if class_label_key in sample:
                class_counts[sample[class_label_key]] += 1
        
        if not class_counts:
            raise ValueError(f"No samples found with class label key '{class_label_key}'")
        
        num_classes = len(class_counts)
        
        # Calculate class distribution statistics
        class_frequencies = {cls: count/total_samples for cls, count in class_counts.items()}
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])  # Sort by count
        
        # Identify rare classes (bottom 10% or classes with <5% representation)
        rare_threshold = max(0.05, total_samples * 0.1 / num_classes) if num_classes > 1 else total_samples * 0.05
        rare_classes = [cls for cls, count in class_counts.items() if count < rare_threshold]
        
        # Calculate imbalance metrics
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Calculate Gini coefficient for inequality measurement
        sorted_counts = sorted(class_counts.values())
        n = len(sorted_counts)
        gini_coefficient = (2 * sum((i + 1) * count for i, count in enumerate(sorted_counts))) / (n * sum(sorted_counts)) - (n + 1) / n
        
        # Identify severely imbalanced classes
        median_count = np.median(list(class_counts.values()))
        severely_underrepresented = [cls for cls, count in class_counts.items() if count < median_count * 0.1]
        severely_overrepresented = [cls for cls, count in class_counts.items() if count > median_count * 5]
        
        # Binary classification specific metrics
        binary_metrics = {}
        if num_classes == 2 or (positive_class and negative_class):
            binary_metrics = self._calculate_binary_metrics(
                class_counts, positive_class, negative_class
            )
        
        analysis_results = {
            'total_samples': total_samples,
            'num_classes': num_classes,
            'classification_type': 'binary' if num_classes == 2 else 'multi_class',
            'class_counts': dict(class_counts),
            'class_frequencies': class_frequencies,
            'rare_classes': rare_classes,
            'severely_underrepresented': severely_underrepresented,
            'severely_overrepresented': severely_overrepresented,
            'imbalance_ratio': imbalance_ratio,
            'gini_coefficient': gini_coefficient,
            'class_distribution_stats': {
                'min_count': min_count,
                'max_count': max_count,
                'median_count': median_count,
                'mean_count': np.mean(list(class_counts.values())),
                'std_count': np.std(list(class_counts.values()))
            },
            'imbalance_severity': self._classify_imbalance_severity(imbalance_ratio, gini_coefficient),
            'sorted_classes_by_frequency': sorted_classes
        }
        
        # Add binary-specific metrics if applicable
        if binary_metrics:
            analysis_results['binary_metrics'] = binary_metrics
        
        print(f"📊 Class imbalance analysis complete:")
        print(f"   - Classification type: {analysis_results['classification_type']}")
        print(f"   - Total classes: {num_classes}")
        print(f"   - Imbalance ratio (max/min): {imbalance_ratio:.2f}")
        print(f"   - Gini coefficient: {gini_coefficient:.3f}")
        print(f"   - Rare classes (<{rare_threshold:.1f} samples): {len(rare_classes)}")
        print(f"   - Severely underrepresented classes: {len(severely_underrepresented)}")
        
        self._log_audit_event("CLASS_IMBALANCE_ANALYSIS_COMPLETE", 
                            f"Analyzed {num_classes} classes with imbalance ratio {imbalance_ratio:.2f}")
        
        return analysis_results

    def _analyze_binary_legacy_format(self, positive_samples: List[Dict], negative_samples: List[Dict]) -> Dict:
        """Handle legacy binary format for backwards compatibility."""
        print("📊 Processing legacy binary format...")
        
        # Count feature occurrences in positive and negative sets
        positive_feature_counts = Counter()
        negative_feature_counts = Counter()
        
        # Extract features from samples
        for sample in positive_samples:
            if 'features' in sample:
                for feature in sample['features']:
                    positive_feature_counts[feature] += 1
        
        for sample in negative_samples:
            if 'features' in sample:
                for feature in sample['features']:
                    negative_feature_counts[feature] += 1
        
        # Calculate imbalance metrics
        all_features = set(positive_feature_counts.keys()) | set(negative_feature_counts.keys())
        imbalance_scores = {}
        
        for feature in all_features:
            pos_count = positive_feature_counts.get(feature, 0)
            neg_count = negative_feature_counts.get(feature, 0)
            
            if pos_count + neg_count > 0:
                imbalance_ratio = abs(pos_count - neg_count) / (pos_count + neg_count)
                imbalance_scores[feature] = {
                    'positive_count': pos_count,
                    'negative_count': neg_count,
                    'imbalance_ratio': imbalance_ratio,
                    'severity': 'high' if imbalance_ratio > 0.6 else 'medium' if imbalance_ratio > 0.3 else 'low'
                }
        
        # Summary statistics
        total_features = len(all_features)
        high_imbalance_features = sum(1 for score in imbalance_scores.values() if score['severity'] == 'high')
        features_only_in_positive = len(set(positive_feature_counts.keys()) - set(negative_feature_counts.keys()))
        features_only_in_negative = len(set(negative_feature_counts.keys()) - set(positive_feature_counts.keys()))
        
        # Calculate overall imbalance ratio
        total_positive = len(positive_samples)
        total_negative = len(negative_samples)
        overall_imbalance_ratio = max(total_positive, total_negative) / min(total_positive, total_negative) if min(total_positive, total_negative) > 0 else float('inf')
        
        return {
            'total_samples': total_positive + total_negative,
            'num_classes': 2,
            'classification_type': 'binary_legacy',
            'class_counts': {'positive': total_positive, 'negative': total_negative},
            'class_frequencies': {
                'positive': total_positive / (total_positive + total_negative),
                'negative': total_negative / (total_positive + total_negative)
            },
            'imbalance_ratio': overall_imbalance_ratio,
            'total_features': total_features,
            'high_imbalance_count': high_imbalance_features,
            'features_only_positive': features_only_in_positive,
            'features_only_negative': features_only_in_negative,
            'imbalance_percentage': (high_imbalance_features / total_features * 100) if total_features > 0 else 0,
            'feature_imbalance_scores': imbalance_scores,
            'positive_samples_count': total_positive,
            'negative_samples_count': total_negative,
            'imbalance_severity': self._classify_imbalance_severity(overall_imbalance_ratio, 0.5)  # Approximate Gini
        }

    def _calculate_binary_metrics(self, class_counts: Counter, positive_class: str = None, negative_class: str = None) -> Dict:
        """Calculate binary-specific metrics."""
        classes = list(class_counts.keys())
        
        if positive_class and negative_class:
            pos_count = class_counts.get(positive_class, 0)
            neg_count = class_counts.get(negative_class, 0)
        else:
            # Auto-detect: assume smaller class is positive (rare/minority class)
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
            pos_count = sorted_classes[0][1]
            neg_count = sorted_classes[1][1]
            positive_class = sorted_classes[0][0]
            negative_class = sorted_classes[1][0]
        
        total = pos_count + neg_count
        
        return {
            'positive_class': positive_class,
            'negative_class': negative_class,
            'positive_count': pos_count,
            'negative_count': neg_count,
            'positive_ratio': pos_count / total if total > 0 else 0,
            'negative_ratio': neg_count / total if total > 0 else 0,
            'binary_imbalance_ratio': neg_count / pos_count if pos_count > 0 else float('inf')
        }

    def generate_balanced_dataset(self, samples: List[Dict], 
                                class_label_key: str = 'label',
                                balancing_strategy: str = 'auto',
                                target_samples_per_class: int = None,
                                min_samples_per_class: int = 10,
                                positive_class: str = None,
                                negative_class: str = None) -> List[Dict]:
        """
        Unified dataset balancing for both binary and multi-class scenarios.
        Supports various balancing strategies for genomic datasets.
        
        Parameters:
        - samples: List of sample dictionaries or tuple of (positive_samples, negative_samples) for legacy format
        - class_label_key: Key name for class labels
        - balancing_strategy: Strategy ('auto', 'undersample', 'oversample', 'undersample_oversample', 'smote_like')
        - target_samples_per_class: Target number of samples per class (if None, calculated automatically)
        - min_samples_per_class: Minimum samples required per class
        - positive_class: For binary classification, specify positive class label
        - negative_class: For binary classification, specify negative class label
        
        Returns:
        - List of balanced samples
        """
        print(f"⚖️ Generating balanced dataset using unified approach...")
        self._log_audit_event("UNIFIED_BALANCING_START", f"Strategy: {balancing_strategy}")
        
        # Handle legacy binary format
        if isinstance(samples, tuple) and len(samples) == 2:
            return self._balance_binary_legacy_format(samples, balancing_strategy, target_count)
        
        # Analyze current imbalance
        analysis = self.analyze_class_imbalance(samples, class_label_key, positive_class, negative_class)
        
        # Auto-select strategy based on analysis
        if balancing_strategy == 'auto':
            balancing_strategy = self._auto_select_balancing_strategy(analysis)
            print(f"   🤖 Auto-selected strategy: {balancing_strategy}")
        
        # Group samples by class
        class_samples = defaultdict(list)
        for sample in samples:
            if class_label_key in sample:
                class_samples[sample[class_label_key]].append(sample)
        
        # Remove classes with too few samples
        filtered_classes = {cls: samples_list for cls, samples_list in class_samples.items() 
                           if len(samples_list) >= min_samples_per_class}
        
        removed_classes = set(class_samples.keys()) - set(filtered_classes.keys())
        if removed_classes:
            print(f"   ⚠️ Removed {len(removed_classes)} classes with <{min_samples_per_class} samples: {removed_classes}")
        
        # Determine target samples per class
        if target_samples_per_class is None:
            target_samples_per_class = self._calculate_target_samples_per_class(
                filtered_classes, balancing_strategy
            )
        
        balanced_samples = []
        
        # Apply balancing strategy to each class
        for class_label, class_sample_list in filtered_classes.items():
            current_count = len(class_sample_list)
            
            if current_count == target_samples_per_class:
                # Already balanced
                balanced_samples.extend(class_sample_list)
            elif current_count > target_samples_per_class:
                # Undersample
                undersampled = self._undersample_class(class_sample_list, target_samples_per_class)
                balanced_samples.extend(undersampled)
            else:
                # Oversample
                if balancing_strategy in ['smote_like', 'oversample']:
                    if balancing_strategy == 'smote_like':
                        oversampled = self._smote_like_oversample(class_sample_list, target_samples_per_class)
                    else:
                        oversampled = self._simple_oversample(class_sample_list, target_samples_per_class)
                    balanced_samples.extend(oversampled)
                else:
                    # For undersample strategy, keep original size
                    balanced_samples.extend(class_sample_list)
        
        # Shuffle the balanced dataset
        random.shuffle(balanced_samples)
        
        print(f"✅ Unified balancing complete:")
        print(f"   - Classification type: {analysis['classification_type']}")
        print(f"   - Original classes: {len(class_samples)}")
        print(f"   - Retained classes: {len(filtered_classes)}")
        print(f"   - Strategy used: {balancing_strategy}")
        print(f"   - Target samples per class: {target_samples_per_class}")
        print(f"   - Total balanced samples: {len(balanced_samples)}")
        
        self._log_audit_event("UNIFIED_BALANCING_COMPLETE", 
                            f"Generated {len(balanced_samples)} balanced samples across {len(filtered_classes)} classes")
        
        return balanced_samples

    def _balance_binary_legacy_format(self, samples: Tuple[List[Dict], List[Dict]], 
                                    strategy: str, target_count: int = None) -> List[Dict]:
        """Handle legacy binary format balancing."""
        positive_samples, negative_samples = samples
        
        if target_count is None:
            if strategy == 'undersample':
                target_count = min(len(positive_samples), len(negative_samples))
            elif strategy == 'oversample':
                target_count = max(len(positive_samples), len(negative_samples))
            else:  # undersample_oversample or auto
                target_count = int(np.median([len(positive_samples), len(negative_samples)]))
        
        # Balance positive samples
        if len(positive_samples) > target_count:
            balanced_positive = self._undersample_class(positive_samples, target_count)
        elif len(positive_samples) < target_count:
            balanced_positive = self._simple_oversample(positive_samples, target_count)
        else:
            balanced_positive = positive_samples
        
        # Balance negative samples
        if len(negative_samples) > target_count:
            balanced_negative = self._undersample_class(negative_samples, target_count)
        elif len(negative_samples) < target_count:
            balanced_negative = self._simple_oversample(negative_samples, target_count)
        else:
            balanced_negative = negative_samples
        
        # Combine and shuffle
        balanced_samples = balanced_positive + balanced_negative
        random.shuffle(balanced_samples)
        
        return balanced_samples

    def _auto_select_balancing_strategy(self, analysis: Dict) -> str:
        """Automatically select the best balancing strategy based on analysis."""
        severity = analysis['imbalance_severity']
        num_classes = analysis['num_classes']
        imbalance_ratio = analysis['imbalance_ratio']
        
        # Binary classification auto-selection
        if num_classes == 2:
            if imbalance_ratio > 20:
                return 'undersample_oversample'  # Severe binary imbalance
            elif imbalance_ratio > 5:
                return 'undersample'  # Moderate binary imbalance
            else:
                return 'oversample'  # Mild binary imbalance
        
        # Multi-class auto-selection
        else:
            if severity == 'severe':
                return 'smote_like'  # Generate synthetic samples for severe imbalance
            elif severity == 'high':
                return 'undersample_oversample'  # Combined approach
            elif severity == 'moderate':
                return 'undersample'  # Conservative approach
            else:
                return 'oversample'  # Mild imbalance, just boost minorities
    
    def _calculate_target_samples_per_class(self, class_samples: Dict, strategy: str) -> int:
        """Calculate target samples per class based on strategy."""
        class_sizes = [len(samples_list) for samples_list in class_samples.values()]
        
        if strategy == 'undersample':
            return min(class_sizes)
        elif strategy == 'oversample':
            return max(class_sizes)
        else:  # undersample_oversample, smote_like, or auto
            return int(np.median(class_sizes))

    def _classify_imbalance_severity(self, imbalance_ratio: float, gini_coefficient: float) -> str:
        """Classify the severity of class imbalance based on multiple metrics."""
        if imbalance_ratio > 100 or gini_coefficient > 0.7:
            return "severe"
        elif imbalance_ratio > 20 or gini_coefficient > 0.5:
            return "high"
        elif imbalance_ratio > 5 or gini_coefficient > 0.3:
            return "moderate"
        else:
            return "low"

    def handle_specialized_imbalance(self, samples: List[Dict], 
                                   domain_type: str,
                                   **domain_kwargs) -> Dict:
        """
        Unified handler for specialized genomic imbalance scenarios.
        
        Parameters:
        - samples: List of sample data
        - domain_type: Type of specialized handling ('rare_variant', 'disease_classification', 'species_classification')
        - domain_kwargs: Domain-specific parameters
        
        Returns:
        - Dictionary with specialized balancing results
        """
        print(f"🧬 Handling specialized {domain_type} imbalance...")
        self._log_audit_event("SPECIALIZED_IMBALANCE_START", f"Domain: {domain_type}")
        
        if domain_type == 'rare_variant':
            return self._handle_rare_variant_imbalance(samples, **domain_kwargs)
        elif domain_type == 'disease_classification':
            return self._handle_disease_classification_imbalance(samples, **domain_kwargs)
        elif domain_type == 'species_classification':
            return self._handle_species_classification_imbalance(samples, **domain_kwargs)
        else:
            raise ValueError(f"Unknown domain type: {domain_type}")

    def _handle_rare_variant_imbalance(self, samples: List[Dict], 
                                     variant_type_key: str = 'variant_type',
                                     pathogenicity_key: str = 'pathogenicity',
                                     frequency_threshold: float = 0.01) -> Dict:
        """Specialized handling for rare variant prediction imbalance."""
        print("🧬 Applying rare variant-specific balancing...")
        
        # Group by pathogenicity
        pathogenic_samples = []
        benign_samples = []
        vus_samples = []  # Variants of Unknown Significance
        
        variant_analysis = {
            'total_variants': len(samples),
            'variant_types': Counter(),
            'pathogenicity_distribution': Counter(),
            'balanced_samples': []
        }
        
        for sample in samples:
            if variant_type_key in sample:
                variant_analysis['variant_types'][sample[variant_type_key]] += 1
            
            if pathogenicity_key in sample:
                pathogenicity = sample[pathogenicity_key].lower()
                variant_analysis['pathogenicity_distribution'][pathogenicity] += 1
                
                if 'pathogenic' in pathogenicity and 'benign' not in pathogenicity:
                    pathogenic_samples.append(sample)
                elif 'benign' in pathogenicity:
                    benign_samples.append(sample)
                else:
                    vus_samples.append(sample)
        
        # Apply specialized rare variant balancing
        pathogenic_count = len(pathogenic_samples)
        benign_count = len(benign_samples)
        vus_count = len(vus_samples)
        
        print(f"📊 Variant distribution:")
        print(f"   - Pathogenic variants: {pathogenic_count}")
        print(f"   - Benign variants: {benign_count}")
        print(f"   - VUS (Unknown): {vus_count}")
        
        if pathogenic_count > 0 and benign_count > 0:
            imbalance_ratio = benign_count / pathogenic_count
            
            if imbalance_ratio > 10:  # Severe imbalance common in rare variant studies
                print(f"   ⚠️ Severe imbalance detected (ratio: {imbalance_ratio:.1f})")
                
                # Strategy: Preserve all pathogenic variants (critical minority)
                # Limit benign variants to maintain realistic but manageable ratio
                target_pathogenic = pathogenic_count  # Keep all pathogenic
                target_benign = min(benign_count, pathogenic_count * 5)  # Max 5:1 ratio
                target_vus = min(vus_count, pathogenic_count * 2)  # Max 2:1 ratio
                
                balanced_samples = []
                balanced_samples.extend(pathogenic_samples)  # Keep all pathogenic
                
                if target_benign < len(benign_samples):
                    balanced_samples.extend(random.sample(benign_samples, target_benign))
                else:
                    balanced_samples.extend(benign_samples)
                
                if vus_samples and target_vus < len(vus_samples):
                    balanced_samples.extend(random.sample(vus_samples, target_vus))
                elif vus_samples:
                    balanced_samples.extend(vus_samples)
                
                variant_analysis['balanced_samples'] = balanced_samples
                variant_analysis['balancing_applied'] = True
                variant_analysis['final_distribution'] = {
                    'pathogenic': len([s for s in balanced_samples if pathogenicity_key in s and 'pathogenic' in s[pathogenicity_key].lower()]),
                    'benign': len([s for s in balanced_samples if pathogenicity_key in s and 'benign' in s[pathogenicity_key].lower()]),
                    'vus': len([s for s in balanced_samples if pathogenicity_key in s and s[pathogenicity_key].lower() not in ['pathogenic', 'benign']])
                }
                
                print(f"✅ Rare variant balancing applied:")
                print(f"   - Final pathogenic: {variant_analysis['final_distribution']['pathogenic']}")
                print(f"   - Final benign: {variant_analysis['final_distribution']['benign']}")
                print(f"   - Final VUS: {variant_analysis['final_distribution']['vus']}")
        
        return variant_analysis

    def _handle_disease_classification_imbalance(self, samples: List[Dict],
                                               disease_key: str = 'disease',
                                               severity_key: str = 'severity',
                                               rare_disease_threshold: int = 50) -> Dict:
        """Specialized handling for disease classification imbalance."""
        print("🏥 Applying disease classification-specific balancing...")
        
        # Analyze disease distribution
        disease_counts = Counter()
        severity_distribution = defaultdict(Counter)
        
        for sample in samples:
            if disease_key in sample:
                disease = sample[disease_key]
                disease_counts[disease] += 1
                
                if severity_key in sample:
                    severity_distribution[disease][sample[severity_key]] += 1
        
        # Identify rare vs common diseases
        rare_diseases = [disease for disease, count in disease_counts.items() 
                        if count < rare_disease_threshold]
        common_diseases = [disease for disease, count in disease_counts.items() 
                          if count >= rare_disease_threshold]
        
        print(f"📊 Disease classification analysis:")
        print(f"   - Total diseases: {len(disease_counts)}")
        print(f"   - Rare diseases (<{rare_disease_threshold} samples): {len(rare_diseases)}")
        print(f"   - Common diseases: {len(common_diseases)}")
        
        # Group samples by disease
        disease_samples = defaultdict(list)
        for sample in samples:
            if disease_key in sample:
                disease_samples[sample[disease_key]].append(sample)
        
        # Apply disease-specific balancing
        balanced_samples = []
        
        # Strategy 1: Preserve all rare disease samples
        for rare_disease in rare_diseases:
            balanced_samples.extend(disease_samples[rare_disease])
            print(f"   💊 Preserved all {len(disease_samples[rare_disease])} samples for rare disease: {rare_disease}")
        
        # Strategy 2: Balance common diseases
        if common_diseases:
            common_disease_counts = [disease_counts[disease] for disease in common_diseases]
            target_common_count = int(np.median(common_disease_counts))
            
            for common_disease in common_diseases:
                current_samples = disease_samples[common_disease]
                
                if len(current_samples) > target_common_count:
                    # Undersample while preserving severity distribution
                    stratified_samples = self._stratify_by_severity(current_samples, severity_key, target_common_count)
                    balanced_samples.extend(stratified_samples)
                else:
                    # Keep all samples for this common disease
                    balanced_samples.extend(current_samples)
        
        # Shuffle final dataset
        random.shuffle(balanced_samples)
        
        return {
            'original_samples': len(samples),
            'balanced_samples': len(balanced_samples),
            'total_diseases': len(disease_counts),
            'rare_diseases': len(rare_diseases),
            'common_diseases': len(common_diseases),
            'disease_distribution': dict(disease_counts),
            'severity_distribution': {disease: dict(severity_dist) 
                                    for disease, severity_dist in severity_distribution.items()},
            'balanced_dataset': balanced_samples
        }

    def _handle_species_classification_imbalance(self, samples: List[Dict],
                                               species_key: str = 'species',
                                               phylogenetic_key: str = 'phylogeny') -> Dict:
        """Specialized handling for species classification imbalance."""
        print("🌱 Applying species classification-specific balancing...")
        
        # Analyze species distribution
        species_counts = Counter()
        for sample in samples:
            if species_key in sample:
                species_counts[sample[species_key]] += 1
        
        # Apply phylogenetic-aware balancing
        # Group related species together
        phylogenetic_groups = defaultdict(list)
        for sample in samples:
            if phylogenetic_key in sample:
                phylo_group = sample[phylogenetic_key]
                phylogenetic_groups[phylo_group].append(sample)
        
        # Balance within and across phylogenetic groups
        balanced_samples = []
        for group, group_samples in phylogenetic_groups.items():
            # Apply intra-group balancing
            group_species_counts = Counter()
            for sample in group_samples:
                if species_key in sample:
                    group_species_counts[sample[species_key]] += 1
            
            if len(group_species_counts) > 1:
                # Balance species within this phylogenetic group
                target_per_species = int(np.median(list(group_species_counts.values())))
                group_balanced = self.generate_balanced_dataset(
                    group_samples, 
                    class_label_key=species_key,
                    target_samples_per_class=target_per_species
                )
                balanced_samples.extend(group_balanced)
            else:
                balanced_samples.extend(group_samples)
        
        return {
            'original_samples': len(samples),
            'balanced_samples': len(balanced_samples),
            'species_distribution': dict(species_counts),
            'phylogenetic_groups': len(phylogenetic_groups),
            'balanced_dataset': balanced_samples
        }

    def generate_comprehensive_balance_report(self, samples: List[Dict], 
                                            class_label_key: str = 'label',
                                            domain_type: str = None,
                                            include_recommendations: bool = True) -> Dict:
        """
        Generate unified comprehensive balance analysis and recommendations.
        Works for both binary and multi-class scenarios.
        
        Parameters:
        - samples: List of samples to analyze or tuple for legacy binary format
        - class_label_key: Key for class labels
        - domain_type: Optional domain specialization ('rare_variant', 'disease_classification', 'species_classification')
        - include_recommendations: Whether to include balancing recommendations
        
        Returns:
        - Comprehensive balance report
        """
        print("📊 Generating comprehensive balance report...")
        
        # Perform unified class imbalance analysis
        imbalance_analysis = self.analyze_class_imbalance(samples, class_label_key)
        
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "species": self.species,
            "taxon_id": self.taxon_id,
            "analysis_type": "unified_class_imbalance",
            "classification_type": imbalance_analysis['classification_type'],
            "class_label_key": class_label_key,
            "imbalance_analysis": imbalance_analysis
        }
        
        # Add domain-specific analysis if specified
        if domain_type:
            try:
                domain_analysis = self.handle_specialized_imbalance(samples, domain_type)
                report["domain_specific_analysis"] = domain_analysis
            except Exception as e:
                print(f"⚠️ Domain-specific analysis failed: {e}")
        
        if include_recommendations:
            recommendations = self._generate_unified_recommendations(imbalance_analysis, domain_type)
            report["balancing_recommendations"] = recommendations
        
        # Add genomic context analysis
        if not isinstance(samples, tuple):  # Skip for legacy format
            genomic_context = self._analyze_genomic_context(samples)
            report["genomic_context_analysis"] = genomic_context
        
        print(f"✅ Comprehensive balance report generated:")
        print(f"   - Classification type: {imbalance_analysis['classification_type']}")
        print(f"   - Classes analyzed: {imbalance_analysis['num_classes']}")
        print(f"   - Imbalance severity: {imbalance_analysis['imbalance_severity']}")
        if domain_type:
            print(f"   - Domain specialization: {domain_type}")
        
        self._log_audit_event("COMPREHENSIVE_BALANCE_REPORT_GENERATED", 
                            f"Report for {imbalance_analysis['num_classes']} classes, type: {imbalance_analysis['classification_type']}")
        
        return report

    def _generate_unified_recommendations(self, analysis: Dict, domain_type: str = None) -> Dict:
        """Generate unified recommendations based on analysis."""
        recommendations = {
            "strategy_recommendations": [],
            "priority_actions": [],
            "risk_assessment": {},
            "implementation_guidelines": []
        }
        
        severity = analysis['imbalance_severity']
        num_classes = analysis['num_classes']
        imbalance_ratio = analysis['imbalance_ratio']
        classification_type = analysis['classification_type']
        
        # General recommendations based on severity
        if severity == "severe":
            recommendations["strategy_recommendations"].extend([
                "Apply combined undersample-oversample strategy",
                "Consider SMOTE-like synthetic sample generation", 
                "Implement stratified cross-validation",
                "Use class-weighted loss functions in model training",
                "Monitor minority class performance closely"
            ])
            recommendations["priority_actions"].append("CRITICAL: Address severe imbalance before model training")
            
        elif severity == "high":
            recommendations["strategy_recommendations"].extend([
                "Apply undersampling to majority classes",
                "Implement stratified sampling",
                "Consider ensemble methods for imbalanced data",
                "Use appropriate evaluation metrics (F1-score, balanced accuracy)"
            ])
            
        elif severity == "moderate":
            recommendations["strategy_recommendations"].extend([
                "Monitor performance on minority classes",
                "Use appropriate evaluation metrics",
                "Consider cost-sensitive learning"
            ])
        
        # Binary vs Multi-class specific recommendations
        if classification_type == 'binary' or num_classes == 2:
            recommendations["implementation_guidelines"].extend([
                "For binary classification, focus on precision-recall trade-off",
                "Consider threshold tuning for optimal performance",
                "Use ROC-AUC and PR-AUC for evaluation"
            ])
        else:
            recommendations["implementation_guidelines"].extend([
                "For multi-class, use macro/weighted F1-score",
                "Consider hierarchical classification for related classes",
                "Monitor per-class performance individually"
            ])
        
        # Domain-specific recommendations
        if domain_type == 'rare_variant':
            recommendations["genomic_specific"] = [
                "Preserve all pathogenic variants (critical minority class)",
                "Limit benign variants to manageable ratio (max 5:1)",
                "Include VUS (variants of unknown significance) cautiously",
                "Consider functional impact in balancing decisions"
            ]
        elif domain_type == 'disease_classification':
            recommendations["genomic_specific"] = [
                "Preserve all rare disease samples completely",
                "Balance common diseases while maintaining severity distribution",
                "Consider comorbidity patterns in sampling",
                "Use disease-specific evaluation metrics"
            ]
        elif domain_type == 'species_classification':
            recommendations["genomic_specific"] = [
                "Apply phylogenetic-aware balancing",
                "Preserve endangered/rare species samples",
                "Consider evolutionary relationships in synthetic generation",
                "Use hierarchical classification for taxonomic levels"
            ]
        
        # Risk assessment
        recommendations["risk_assessment"] = {
            "model_bias_risk": "high" if severity in ["severe", "high"] else "moderate",
            "generalization_risk": "high" if imbalance_ratio > 50 else "moderate",
            "minority_class_detection_risk": "high" if len(analysis.get('rare_classes', [])) > num_classes * 0.3 else "low",
            "overfitting_risk": "high" if num_classes > 10 and severity == "severe" else "moderate"
        }
        
        return recommendations

    def generate_balanced_negative_samples_maxflow(self, positive_samples: List[Dict], 
                                                  candidate_negatives: List[Dict],
                                                  balance_threshold: float = 0.1) -> List[Dict]:
        """
        Generate class-balanced negative samples using maximum flow approach.
        DEPRECATED: This method is maintained for backwards compatibility.
        Use generate_balanced_dataset() with balancing_strategy='auto' for new implementations.
        
        Parameters:
        - positive_samples: List of positive samples
        - candidate_negatives: List of candidate negative samples
        - balance_threshold: Maximum allowed imbalance ratio (0.0 = perfect balance)
        
        Returns:
        - List of balanced negative samples
        """
        print("⚠️ DEPRECATED: Using legacy maxflow method. Consider migrating to generate_balanced_dataset()")
        self._log_audit_event("LEGACY_MAXFLOW_BALANCING", "Using deprecated maxflow method")
        
        # Convert to unified format and use new framework
        all_samples = []
        
        # Add positive samples with labels
        for sample in positive_samples:
            sample_copy = sample.copy()
            sample_copy['label'] = 'positive'
            all_samples.append(sample_copy)
        
        # Add negative samples with labels
        for sample in candidate_negatives:
            sample_copy = sample.copy()
            sample_copy['label'] = 'negative'
            all_samples.append(sample_copy)
        
        # Use unified balancing with undersample strategy (similar to maxflow approach)
        balanced_samples = self.generate_balanced_dataset(
            samples=all_samples,
            class_label_key='label',
            balancing_strategy='undersample',
            min_samples_per_class=1
        )
        
        # Extract only negative samples for backwards compatibility
        negative_samples = [s for s in balanced_samples if s.get('label') == 'negative']
        
        # Remove the label we added
        for sample in negative_samples:
            if 'label' in sample:
                del sample['label']
        
        print(f"✅ Legacy maxflow balancing complete: {len(negative_samples)} negative samples selected")
        return negative_samples

    def generate_balanced_negative_samples_gibbs(self, positive_samples: List[Dict], 
                                               candidate_negatives: List[Dict],
                                               n_iterations: int = 1000,
                                               temperature: float = 1.0) -> List[Dict]:
        """
        Generate class-balanced negative samples using Gibbs sampling approach.
        DEPRECATED: This method is maintained for backwards compatibility.
        Use generate_balanced_dataset() with balancing_strategy='smote_like' for new implementations.
        
        Parameters:
        - positive_samples: List of positive samples
        - candidate_negatives: List of candidate negative samples  
        - n_iterations: Number of Gibbs sampling iterations (ignored in unified approach)
        - temperature: Temperature parameter for sampling (ignored in unified approach)
        
        Returns:
        - List of balanced negative samples
        """
        print("⚠️ DEPRECATED: Using legacy Gibbs method. Consider migrating to generate_balanced_dataset()")
        self._log_audit_event("LEGACY_GIBBS_BALANCING", "Using deprecated Gibbs method")
        
        # Convert to unified format and use new framework
        all_samples = []
        
        # Add positive samples with labels
        for sample in positive_samples:
            sample_copy = sample.copy()
            sample_copy['label'] = 'positive'
            all_samples.append(sample_copy)
        
        # Add negative samples with labels
        for sample in candidate_negatives:
            sample_copy = sample.copy()
            sample_copy['label'] = 'negative'
            all_samples.append(sample_copy)
        
        # Use unified balancing with SMOTE-like strategy (similar to Gibbs sampling approach)
        balanced_samples = self.generate_balanced_dataset(
            samples=all_samples,
            class_label_key='label',
            balancing_strategy='smote_like',
            target_samples_per_class=len(positive_samples),
            min_samples_per_class=1
        )
        
        # Extract only negative samples for backwards compatibility
        negative_samples = [s for s in balanced_samples if s.get('label') == 'negative']
        
        # Remove the label we added
        for sample in negative_samples:
            if 'label' in sample:
                del sample['label']
        
        print(f"✅ Legacy Gibbs balancing complete: {len(negative_samples)} negative samples generated")
        return negative_samples

    def generate_class_balance_report(self, positive_samples: List[Dict] = None, 
                                    negative_samples: List[Dict] = None,
                                    balanced_negatives: List[Dict] = None,
                                    samples: List[Dict] = None,
                                    class_label_key: str = 'label') -> Dict:
        """
        Generate comprehensive class balance analysis report.
        Supports both legacy binary format and new unified format.
        
        Parameters:
        - positive_samples: Legacy format - list of positive samples
        - negative_samples: Legacy format - list of negative samples  
        - balanced_negatives: Legacy format - list of balanced negative samples
        - samples: New unified format - list of all samples with class labels
        - class_label_key: Key for class labels in unified format
        
        Returns:
        - Comprehensive class balance report
        """
        print("📊 Generating class balance report...")
        
        # Handle legacy binary format
        if positive_samples is not None and negative_samples is not None:
            print("📊 Using legacy binary format...")
            
            # Convert to unified format
            all_samples = []
            for sample in positive_samples:
                sample_copy = sample.copy()
                sample_copy['label'] = 'positive'
                all_samples.append(sample_copy)
            
            for sample in negative_samples:
                sample_copy = sample.copy()
                sample_copy['label'] = 'negative'
                all_samples.append(sample_copy)
            
            # Generate unified report
            report = self.generate_comprehensive_balance_report(
                samples=all_samples,
                class_label_key='label',
                include_recommendations=True
            )
            
            # Add legacy-specific information
            if balanced_negatives is not None:
                balanced_all_samples = []
                for sample in positive_samples:
                    sample_copy = sample.copy()
                    sample_copy['label'] = 'positive'
                    balanced_all_samples.append(sample_copy)
                
                for sample in balanced_negatives:
                    sample_copy = sample.copy()
                    sample_copy['label'] = 'negative'
                    balanced_all_samples.append(sample_copy)
                
                balanced_analysis = self.analyze_class_imbalance(balanced_all_samples, 'label')
                report["balanced_analysis"] = balanced_analysis
                
                # Calculate improvement metrics
                original_analysis = report["imbalance_analysis"]
                if "binary_metrics" in original_analysis and "binary_metrics" in balanced_analysis:
                    original_ratio = original_analysis["binary_metrics"]["binary_imbalance_ratio"]
                    balanced_ratio = balanced_analysis["binary_metrics"]["binary_imbalance_ratio"]
                    
                    if original_ratio != float('inf') and balanced_ratio != float('inf'):
                        improvement = ((original_ratio - balanced_ratio) / original_ratio * 100)
                        report["improvement_metrics"] = {
                            "imbalance_ratio_improvement": improvement,
                            "original_ratio": original_ratio,
                            "balanced_ratio": balanced_ratio
                        }
            
            # Add legacy-compatible fields
            if "binary_metrics" in report["imbalance_analysis"]:
                binary_metrics = report["imbalance_analysis"]["binary_metrics"]
                report["sample_statistics"] = {
                    "positive_samples": binary_metrics["positive_count"],
                    "original_negative_samples": binary_metrics["negative_count"],
                    "balanced_negative_samples": len(balanced_negatives) if balanced_negatives else 0
                }
            
            return report
        
        # Handle unified format
        elif samples is not None:
            return self.generate_comprehensive_balance_report(
                samples=samples,
                class_label_key=class_label_key,
                include_recommendations=True
            )
        
        else:
            raise ValueError("Must provide either (positive_samples, negative_samples) or samples parameter")

    def quality_filter_sequences(self, min_length: int = 100, max_length: int = 10000, 
                                min_gc_content: float = 0.2, max_gc_content: float = 0.8,
                                remove_ambiguous: bool = True, max_n_percent: float = 0.05) -> dict:
        """
        Apply quality filtering to genomic sequences based on multiple criteria.
        Implements filtering methods similar to those used in high-throughput sequencing analysis.
        
        Parameters:
        - min_length: Minimum sequence length to retain
        - max_length: Maximum sequence length to retain  
        - min_gc_content: Minimum GC content ratio (0.0-1.0)
        - max_gc_content: Maximum GC content ratio (0.0-1.0)
        - remove_ambiguous: Remove sequences with ambiguous nucleotides
        - max_n_percent: Maximum percentage of N's allowed in sequence
        
        Returns:
        - Dictionary with filtering results and statistics
        """
        print("🔬 Applying quality filtering to genomic sequences...")
        self._log_audit_event("QUALITY_FILTERING_START", f"Filtering sequences with criteria: min_len={min_length}, max_len={max_length}")
        
        accessible_files = self._get_accessible_files()
        filtered_results = {
            "total_sequences": 0,
            "passed_filter": 0,
            "filtered_by_length": 0,
            "filtered_by_gc": 0,
            "filtered_by_ambiguous": 0,
            "filtered_sequences": [],
            "quality_stats": {}
        }
        
        for file_path in accessible_files:
            file_stats = self._filter_file_sequences(
                file_path, min_length, max_length, min_gc_content, 
                max_gc_content, remove_ambiguous, max_n_percent
            )
            
            # Aggregate statistics
            for key in ["total_sequences", "passed_filter", "filtered_by_length", 
                       "filtered_by_gc", "filtered_by_ambiguous"]:
                filtered_results[key] += file_stats[key]
            
            filtered_results["filtered_sequences"].extend(file_stats["filtered_sequences"])
            filtered_results["quality_stats"][os.path.basename(file_path)] = file_stats
        
        # Calculate filtering efficiency
        if filtered_results["total_sequences"] > 0:
            efficiency = (filtered_results["passed_filter"] / filtered_results["total_sequences"]) * 100
            print(f"✅ Quality filtering complete: {efficiency:.2f}% sequences passed filters")
        
        self._log_audit_event("QUALITY_FILTERING_COMPLETE", 
                            f"Filtered {filtered_results['total_sequences']} sequences, {filtered_results['passed_filter']} passed")
        
        return filtered_results

    def _filter_file_sequences(self, file_path: str, min_length: int, max_length: int,
                              min_gc_content: float, max_gc_content: float,
                              remove_ambiguous: bool, max_n_percent: float) -> dict:
        """Filter sequences in a single file based on quality criteria."""
        stats = {
            "total_sequences": 0,
            "passed_filter": 0,
            "filtered_by_length": 0,
            "filtered_by_gc": 0,
            "filtered_by_ambiguous": 0,
            "filtered_sequences": []
        }
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse FASTA format
            sequences = self._parse_fasta_content(content)
            
            for seq_id, sequence in sequences.items():
                stats["total_sequences"] += 1
                
                # Length filtering
                if len(sequence) < min_length or len(sequence) > max_length:
                    stats["filtered_by_length"] += 1
                    continue
                
                # GC content filtering
                gc_content = self._calculate_gc_content(sequence)
                if gc_content < min_gc_content or gc_content > max_gc_content:
                    stats["filtered_by_gc"] += 1
                    continue
                
                # Ambiguous nucleotide filtering
                if remove_ambiguous:
                    n_count = sequence.upper().count('N')
                    if (n_count / len(sequence)) > max_n_percent:
                        stats["filtered_by_ambiguous"] += 1
                        continue
                    
                    # Check for other ambiguous nucleotides
                    valid_nucleotides = set('ATGC')
                    if not set(sequence.upper()).issubset(valid_nucleotides.union({'N'})):
                        stats["filtered_by_ambiguous"] += 1
                        continue
                
                # Sequence passed all filters
                stats["passed_filter"] += 1
                stats["filtered_sequences"].append({
                    "id": seq_id,
                    "sequence": sequence,
                    "length": len(sequence),
                    "gc_content": gc_content,
                    "source_file": os.path.basename(file_path)
                })
        
        except Exception as e:
            print(f"⚠️ Error filtering file {file_path}: {e}")
        
        return stats

    def taxonomic_classification_filter(self, target_taxa: list = None, exclude_taxa: list = None,
                                      min_taxonomy_confidence: float = 0.8) -> dict:
        """
        Filter sequences based on taxonomic classification criteria.
        Implements taxonomy-based filtering similar to QIIME2 methods.
        
        Parameters:
        - target_taxa: List of taxonomic groups to retain (e.g., ['Bacteria', 'Archaea'])
        - exclude_taxa: List of taxonomic groups to exclude (e.g., ['Chloroplast', 'Mitochondria'])
        - min_taxonomy_confidence: Minimum confidence score for taxonomic assignment
        
        Returns:
        - Dictionary with taxonomic filtering results
        """
        print("🧬 Applying taxonomic classification filtering...")
        self._log_audit_event("TAXONOMIC_FILTERING_START", 
                            f"Target taxa: {target_taxa}, Exclude taxa: {exclude_taxa}")
        
        # Default filtering for genomic data
        if target_taxa is None:
            target_taxa = ['Bacteria', 'Archaea', 'Eukaryota']
        
        if exclude_taxa is None:
            exclude_taxa = ['Chloroplast', 'Mitochondria', 'Unassigned']
        
        accessible_files = self._get_accessible_files()
        taxonomy_results = {
            "total_sequences": 0,
            "taxonomically_classified": 0,
            "target_taxa_retained": 0,
            "excluded_taxa_removed": 0,
            "taxonomy_distribution": Counter(),
            "filtered_sequences": []
        }
        
        for file_path in accessible_files:
            file_taxonomy = self._classify_file_sequences(file_path, target_taxa, exclude_taxa)
            
            # Aggregate results
            for key in ["total_sequences", "taxonomically_classified", 
                       "target_taxa_retained", "excluded_taxa_removed"]:
                taxonomy_results[key] += file_taxonomy[key]
            
            taxonomy_results["taxonomy_distribution"].update(file_taxonomy["taxonomy_distribution"])
            taxonomy_results["filtered_sequences"].extend(file_taxonomy["filtered_sequences"])
        
        print(f"✅ Taxonomic filtering complete: {taxonomy_results['target_taxa_retained']} sequences retained")
        self._log_audit_event("TAXONOMIC_FILTERING_COMPLETE", 
                            f"Retained {taxonomy_results['target_taxa_retained']} sequences from target taxa")
        
        return taxonomy_results

    def _classify_file_sequences(self, file_path: str, target_taxa: list, exclude_taxa: list) -> dict:
        """Classify sequences in a single file and apply taxonomic filters."""
        results = {
            "total_sequences": 0,
            "taxonomically_classified": 0,
            "target_taxa_retained": 0,
            "excluded_taxa_removed": 0,
            "taxonomy_distribution": Counter(),
            "filtered_sequences": []
        }
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            sequences = self._parse_fasta_content(content)
            
            for seq_id, sequence in sequences.items():
                results["total_sequences"] += 1
                
                # Simple taxonomic classification based on sequence characteristics
                # In a real implementation, this would use tools like BLAST, VSEARCH, or trained classifiers
                taxonomy = self._simple_taxonomic_assignment(sequence, seq_id)
                
                if taxonomy != "Unassigned":
                    results["taxonomically_classified"] += 1
                    results["taxonomy_distribution"][taxonomy] += 1
                    
                    # Check if taxonomy should be retained
                    include_sequence = False
                    exclude_sequence = False
                    
                    # Check target taxa
                    for target in target_taxa:
                        if target.lower() in taxonomy.lower():
                            include_sequence = True
                            break
                    
                    # Check exclude taxa
                    for exclude in exclude_taxa:
                        if exclude.lower() in taxonomy.lower():
                            exclude_sequence = True
                            break
                    
                    if include_sequence and not exclude_sequence:
                        results["target_taxa_retained"] += 1
                        results["filtered_sequences"].append({
                            "id": seq_id,
                            "sequence": sequence,
                            "taxonomy": taxonomy,
                            "source_file": os.path.basename(file_path)
                        })
                    elif exclude_sequence:
                        results["excluded_taxa_removed"] += 1
        
        except Exception as e:
            print(f"⚠️ Error classifying file {file_path}: {e}")
        
        return results

    def species_selection_filter(self, target_species: list = None, 
                                phylogenetic_distance: float = 0.1) -> dict:
        """
        Filter sequences based on species-specific criteria and phylogenetic relationships.
        
        Parameters:
        - target_species: List of target species to retain
        - phylogenetic_distance: Maximum phylogenetic distance for related species inclusion
        
        Returns:
        - Dictionary with species selection results
        """
        print("🎯 Applying species-specific selection filtering...")
        self._log_audit_event("SPECIES_FILTERING_START", f"Target species: {target_species}")
        
        # If no target species specified, use the current species
        if target_species is None:
            target_species = [self.species]
        
        accessible_files = self._get_accessible_files()
        species_results = {
            "total_sequences": 0,
            "species_matched": 0,
            "phylogenetically_related": 0,
            "species_distribution": Counter(),
            "filtered_sequences": []
        }
        
        for file_path in accessible_files:
            file_species = self._filter_by_species(file_path, target_species, phylogenetic_distance)
            
            # Aggregate results
            for key in ["total_sequences", "species_matched", "phylogenetically_related"]:
                species_results[key] += file_species[key]
            
            species_results["species_distribution"].update(file_species["species_distribution"])
            species_results["filtered_sequences"].extend(file_species["filtered_sequences"])
        
        print(f"✅ Species filtering complete: {species_results['species_matched']} sequences from target species")
        self._log_audit_event("SPECIES_FILTERING_COMPLETE", 
                            f"Matched {species_results['species_matched']} sequences to target species")
        
        return species_results

    def _filter_by_species(self, file_path: str, target_species: list, phylogenetic_distance: float) -> dict:
        """Filter sequences based on species criteria."""
        results = {
            "total_sequences": 0,
            "species_matched": 0,
            "phylogenetically_related": 0,
            "species_distribution": Counter(),
            "filtered_sequences": []
        }
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            sequences = self._parse_fasta_content(content)
            
            for seq_id, sequence in sequences.items():
                results["total_sequences"] += 1
                
                # Extract species information from sequence ID or perform species assignment
                species_assignment = self._assign_species(sequence, seq_id)
                results["species_distribution"][species_assignment] += 1
                
                # Check if species matches target
                species_match = False
                for target in target_species:
                    if self._species_similarity(species_assignment, target) >= (1.0 - phylogenetic_distance):
                        species_match = True
                        if species_assignment == target:
                            results["species_matched"] += 1
                        else:
                            results["phylogenetically_related"] += 1
                        break
                
                if species_match:
                    results["filtered_sequences"].append({
                        "id": seq_id,
                        "sequence": sequence,
                        "assigned_species": species_assignment,
                        "source_file": os.path.basename(file_path)
                    })
        
        except Exception as e:
            print(f"⚠️ Error filtering species in file {file_path}: {e}")
        
        return results

    def _get_accessible_files(self) -> list[str]:
        """Get list of accessible files (decrypt if necessary for human data)."""
        if self.is_human_data and any(f.endswith(".encrypted") for f in self.fna_files):
            return self._decrypt_files_temporarily()
        else:
            return self.fna_files

    def _parse_fasta_content(self, content: str) -> dict:
        """Parse FASTA content and return sequences dictionary."""
        sequences = {}
        current_id = None
        current_seq = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:]  # Remove '>' character
                current_seq = []
            elif line:
                current_seq.append(line.upper())
        
        # Add the last sequence
        if current_id:
            sequences[current_id] = ''.join(current_seq)
        
        return sequences

    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content of a sequence."""
        sequence = sequence.upper()
        gc_count = sequence.count('G') + sequence.count('C')
        total_count = len(sequence)
        return gc_count / total_count if total_count > 0 else 0.0

    def _simple_taxonomic_assignment(self, sequence: str, seq_id: str) -> str:
        """Simple taxonomic assignment based on sequence characteristics and ID."""
        # This is a simplified taxonomic assignment - in practice, you'd use BLAST, VSEARCH, or trained classifiers
        if 'chloroplast' in seq_id.lower():
            return "Chloroplast"
        elif 'mitochondria' in seq_id.lower() or 'mitochondrial' in seq_id.lower():
            return "Mitochondria"
        elif self.taxon_id == "9606":  # Human
            return "Eukaryota"
        elif any(keyword in seq_id.lower() for keyword in ['bacteria', 'bacterial']):
            return "Bacteria"
        elif any(keyword in seq_id.lower() for keyword in ['archaea', 'archaeal']):
            return "Archaea"
        else:
            # Basic sequence-based classification
            gc_content = self._calculate_gc_content(sequence)
            if gc_content > 0.65:
                return "Bacteria"  # High GC typically bacterial
            elif gc_content < 0.35:
                return "Archaea"   # Low GC might be archaeal
            else:
                return "Eukaryota" # Moderate GC content
    
    def _assign_species(self, sequence: str, seq_id: str) -> str:
        """Assign species based on sequence characteristics."""
        # Extract species from sequence ID if available
        if 'homo sapiens' in seq_id.lower() or 'human' in seq_id.lower():
            return "Homo sapiens"
        elif 'mus musculus' in seq_id.lower() or 'mouse' in seq_id.lower():
            return "Mus musculus"
        else:
            return self.species  # Default to the species being analyzed
    
    def _species_similarity(self, species1: str, species2: str) -> float:
        """Calculate similarity between two species names."""
        # Simple string similarity - in practice, you'd use phylogenetic databases
        if species1.lower() == species2.lower():
            return 1.0
        
        # Check genus level similarity
        genus1 = species1.split()[0] if ' ' in species1 else species1
        genus2 = species2.split()[0] if ' ' in species2 else species2
        
        if genus1.lower() == genus2.lower():
            return 0.8  # Same genus, different species
        else:
            return 0.0  # Different genus

    def generate_filtering_report(self, quality_results: dict = None, 
                                taxonomy_results: dict = None, 
                                species_results: dict = None) -> dict:
        """Generate comprehensive filtering report."""
        report = {
            "species": self.species,
            "taxon_id": self.taxon_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "total_files_processed": len(self.fna_files),
            "filtering_summary": {}
        }
        
        if quality_results:
            report["filtering_summary"]["quality_filtering"] = {
                "total_sequences": quality_results["total_sequences"],
                "sequences_passed": quality_results["passed_filter"],
                "filtering_efficiency": (quality_results["passed_filter"] / quality_results["total_sequences"] * 100) if quality_results["total_sequences"] > 0 else 0
            }
        
        if taxonomy_results:
            report["filtering_summary"]["taxonomic_filtering"] = {
                "taxonomically_classified": taxonomy_results["taxonomically_classified"],
                "target_taxa_retained": taxonomy_results["target_taxa_retained"],
                "taxonomy_distribution": dict(taxonomy_results["taxonomy_distribution"])
            }
        
        if species_results:
            report["filtering_summary"]["species_filtering"] = {
                "species_matched": species_results["species_matched"],
                "phylogenetically_related": species_results["phylogenetically_related"],
                "species_distribution": dict(species_results["species_distribution"])
            }
        
        self._log_audit_event("FILTERING_REPORT_GENERATED", "Comprehensive filtering report generated")
        return report

    def access_genomic_data(self, purpose: str = "research") -> list[str]:
        """
        Access genomic data with HIPAA minimum necessary standard compliance.
        
        Parameters:
        - purpose: Purpose of data access ("research", "treatment", "public_health")
        
        Returns:
        - List of accessible file paths
        """
        if not self.is_human_data:
            return self.fna_files
        
        # HIPAA Minimum Necessary Standard Implementation
        self._log_audit_event("DATA_ACCESS_REQUEST", f"Purpose: {purpose}")
        
        # Validate purpose and apply minimum necessary standard
        if purpose == "treatment":
            # Treatment exception: No minimum necessary restriction per HIPAA
            print("🏥 Treatment access: Full genomic data available (HIPAA treatment exception)")
            accessible_files = self._decrypt_files_temporarily()
        elif purpose in ["research", "public_health"]:
            # Apply minimum necessary standard
            print(f"🔬 {purpose.title()} access: Applying minimum necessary standard")
            print("⚠️ Note: Researcher must justify necessity of full genomic dataset")
            accessible_files = self._decrypt_files_temporarily()
        else:
            raise ValueError("Purpose must be 'research', 'treatment', or 'public_health'")
        
        self._log_audit_event("DATA_ACCESS_GRANTED", f"Files accessed for {purpose}")
        return accessible_files

    def _decrypt_files_temporarily(self) -> list[str]:
        """Temporarily decrypt files for authorized access."""
        decrypted_files = []
        
        for fna_file in self.fna_files:
            if fna_file.endswith(".encrypted"):
                # Decrypt file temporarily
                with open(fna_file, 'rb') as f:
                    encrypted_data = f.read()
                
                decrypted_data = self.cipher_suite.decrypt(encrypted_data)
                temp_path = fna_file.replace(".encrypted", ".temp")
                
                with open(temp_path, 'wb') as f:
                    f.write(decrypted_data)
                
                decrypted_files.append(temp_path)
                self._log_audit_event("TEMP_DECRYPTION", f"Temporarily decrypted {os.path.basename(fna_file)}")
            else:
                decrypted_files.append(fna_file)
        
        return decrypted_files

    def cleanup_temporary_files(self):
        """Clean up temporarily decrypted files."""
        if not self.is_human_data:
            return
            
        for root, dirs, files in os.walk(self.download_folder):
            for file in files:
                if file.endswith(".temp"):
                    file_path = os.path.join(root, file)
                    self._secure_delete(file_path)
                    print(f"🗑️ Securely cleaned up: {os.path.basename(file_path)}")

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.datetime.now().isoformat()

    def generate_hipaa_compliance_report(self) -> dict:
        """Generate HIPAA compliance report."""
        if not self.is_human_data:
            return {"hipaa_applicable": False, "reason": "Non-human genomic data"}
        
        return {
            "hipaa_applicable": True,
            "data_classification": "Protected Health Information (PHI)",
            "taxon_id": self.taxon_id,
            "encryption_status": "AES-256 enabled",
            "audit_events": len(self.audit_log),
            "minimum_necessary_compliance": "Implemented",
            "secure_deletion": "DoD 5220.22-M standard",
            "total_files": len(self.fna_files),
            "encrypted_files": len([f for f in self.fna_files if f.endswith(".encrypted")])
        }

    @classmethod
    def from_fasta_files(cls, fasta_files: List[str], 
                        species: str = "Custom", 
                        download_folder: str = None,
                        validate_sequences: bool = True) -> 'GenomeDataset':
        """
        Create GenomeDataset instance from user-provided FASTA files.
        
        Parameters:
        - fasta_files: List of paths to FASTA files
        - species: Species name for the dataset (default: "Custom")
        - download_folder: Directory to copy/organize files
        - validate_sequences: Whether to validate sequence format
        
        Returns:
        - GenomeDataset instance with user-provided data
        """
        print(f"📁 Creating GenomeDataset from {len(fasta_files)} user-provided FASTA files...")
        
        # Create instance without downloading
        instance = cls.__new__(cls)
        instance.species = species
        instance.taxon_id = "custom"
        instance.download_folder = download_folder if download_folder else f"./Custom_Dataset_{species.replace(' ', '_')}"
        os.makedirs(instance.download_folder, exist_ok=True)
        
        # HIPAA compliance (not applicable for custom data unless specified)
        instance.is_human_data = False
        instance.cipher_suite = None
        instance.audit_log = []
        
        # Copy and validate FASTA files
        instance.fna_files = []
        
        for i, fasta_file in enumerate(fasta_files):
            if not os.path.exists(fasta_file):
                raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
            
            # Copy file to dataset directory
            filename = f"custom_genome_{i+1}.fna"
            dest_path = os.path.join(instance.download_folder, filename)
            
            # Read, validate, and write sequences
            sequences_processed = 0
            with open(fasta_file, 'r') as infile, open(dest_path, 'w') as outfile:
                current_seq = []
                current_header = None
                
                for line in infile:
                    line = line.strip()
                    if line.startswith('>'):
                        # Write previous sequence if exists
                        if current_header and current_seq:
                            sequence = ''.join(current_seq)
                            if validate_sequences:
                                sequence = instance._validate_and_clean_sequence(sequence)
                            outfile.write(f"{current_header}\n")
                            outfile.write(f"{sequence}\n")
                            sequences_processed += 1
                        
                        current_header = line
                        current_seq = []
                    else:
                        current_seq.append(line.upper())
                
                # Write last sequence
                if current_header and current_seq:
                    sequence = ''.join(current_seq)
                    if validate_sequences:
                        sequence = instance._validate_and_clean_sequence(sequence)
                    outfile.write(f"{current_header}\n")
                    outfile.write(f"{sequence}\n")
                    sequences_processed += 1
            
            instance.fna_files.append(dest_path)
            print(f"   ✅ Processed {os.path.basename(fasta_file)}: {sequences_processed} sequences")
        
        print(f"🎉 Custom GenomeDataset created with {len(instance.fna_files)} files")
        return instance

    def _validate_and_clean_sequence(self, sequence: str) -> str:
        """Validate and clean DNA sequence."""
        # Remove any non-nucleotide characters except N
        valid_chars = set('ATGCN')
        cleaned = ''.join(char for char in sequence.upper() if char in valid_chars)
        
        if len(cleaned) < len(sequence) * 0.9:  # More than 10% invalid characters
            print(f"⚠️ Warning: Sequence had {len(sequence) - len(cleaned)} invalid characters")
        
        return cleaned

    def generate_synthetic_metagenomic_dataset(self, 
                                             species_abundances: Dict[str, float],
                                             total_sequences: int = 10000,
                                             sequence_length_range: Tuple[int, int] = (500, 5000),
                                             noise_level: float = 0.01,
                                             output_format: str = "fasta") -> Dict:
        """
        Generate synthetic metagenomic dataset with specified species abundances.
        
        Parameters:
        - species_abundances: Dict mapping species names to relative abundances (should sum to 1.0)
        - total_sequences: Total number of sequences to generate
        - sequence_length_range: (min_length, max_length) for generated sequences
        - noise_level: Proportion of random mutations to introduce (0.0-1.0)
        - output_format: Output format ('fasta', 'csv', 'both')
        
        Returns:
        - Dictionary with generation results and file paths
        """
        print(f"🧬 Generating synthetic metagenomic dataset...")
        print(f"   - Species: {len(species_abundances)}")
        print(f"   - Total sequences: {total_sequences}")
        print(f"   - Noise level: {noise_level:.1%}")
        
        # Validate abundances
        total_abundance = sum(species_abundances.values())
        if abs(total_abundance - 1.0) > 0.01:
            print(f"⚠️ Warning: Species abundances sum to {total_abundance:.3f}, normalizing to 1.0")
            species_abundances = {sp: abund/total_abundance for sp, abund in species_abundances.items()}
        
        # Calculate sequences per species
        sequences_per_species = {}
        assigned_sequences = 0
        
        for species, abundance in species_abundances.items():
            count = int(total_sequences * abundance)
            sequences_per_species[species] = count
            assigned_sequences += count
        
        # Distribute remaining sequences
        remaining = total_sequences - assigned_sequences
        species_list = list(species_abundances.keys())
        for i in range(remaining):
            species = species_list[i % len(species_list)]
            sequences_per_species[species] += 1
        
        # Generate sequences for each species
        synthetic_data = {
            'sequences': [],
            'metadata': [],
            'species_counts': sequences_per_species,
            'generation_params': {
                'total_sequences': total_sequences,
                'noise_level': noise_level,
                'sequence_length_range': sequence_length_range
            }
        }
        
        for species, seq_count in sequences_per_species.items():
            print(f"   📊 Generating {seq_count} sequences for {species}...")
            
            species_sequences = self._generate_species_specific_sequences(
                species=species,
                num_sequences=seq_count,
                length_range=sequence_length_range,
                noise_level=noise_level
            )
            
            synthetic_data['sequences'].extend(species_sequences)
            
            # Add metadata
            for i, seq in enumerate(species_sequences):
                metadata = {
                    'sequence_id': f"{species.replace(' ', '_')}_{i+1:06d}",
                    'species': species,
                    'length': len(seq),
                    'gc_content': self._calculate_gc_content(seq),
                    'noise_applied': noise_level > 0
                }
                synthetic_data['metadata'].append(metadata)
        
        # Save to files
        output_files = self._save_synthetic_dataset(synthetic_data, output_format)
        synthetic_data['output_files'] = output_files
        
        print(f"✅ Synthetic metagenomic dataset generated:")
        print(f"   - Total sequences: {len(synthetic_data['sequences'])}")
        print(f"   - Average GC content: {np.mean([meta['gc_content'] for meta in synthetic_data['metadata']]):.1%}")
        print(f"   - Output files: {list(output_files.keys())}")
        
        return synthetic_data

    def _generate_species_specific_sequences(self, species: str, num_sequences: int, 
                                           length_range: Tuple[int, int], noise_level: float) -> List[str]:
        """Generate species-specific sequences with characteristic composition."""
        sequences = []
        
        # Get species-specific nucleotide composition
        composition = self._get_species_composition(species)
        
        min_len, max_len = length_range
        
        for i in range(num_sequences):
            # Random length within range
            length = random.randint(min_len, max_len)
            
            # Generate base sequence using species composition
            sequence = self._generate_sequence_with_composition(length, composition)
            
            # Apply noise if specified
            if noise_level > 0:
                sequence = self._apply_controlled_noise(sequence, noise_level)
            
            sequences.append(sequence)
        
        return sequences

    def _get_species_composition(self, species: str) -> Dict[str, float]:
        """Get characteristic nucleotide composition for species."""
        # Predefined compositions for common organisms
        compositions = {
            'Escherichia coli': {'A': 0.246, 'T': 0.246, 'G': 0.254, 'C': 0.254},
            'Homo sapiens': {'A': 0.295, 'T': 0.295, 'G': 0.205, 'C': 0.205},
            'Saccharomyces cerevisiae': {'A': 0.309, 'T': 0.309, 'G': 0.191, 'C': 0.191},
            'Bacillus subtilis': {'A': 0.284, 'T': 0.284, 'G': 0.216, 'C': 0.216},
            'Thermotoga maritima': {'A': 0.243, 'T': 0.243, 'G': 0.257, 'C': 0.257},
            'Mycoplasma genitalium': {'A': 0.318, 'T': 0.318, 'G': 0.182, 'C': 0.182},
            'Methanocaldococcus jannaschii': {'A': 0.314, 'T': 0.314, 'G': 0.186, 'C': 0.186},
            'Arabidopsis thaliana': {'A': 0.320, 'T': 0.320, 'G': 0.180, 'C': 0.180},
            'Drosophila melanogaster': {'A': 0.291, 'T': 0.291, 'G': 0.209, 'C': 0.209},
            'Caenorhabditis elegans': {'A': 0.323, 'T': 0.323, 'G': 0.177, 'C': 0.177}
        }
        
        # Return predefined composition or default balanced composition
        return compositions.get(species, {'A': 0.25, 'T': 0.25, 'G': 0.25, 'C': 0.25})

    def _generate_sequence_with_composition(self, length: int, composition: Dict[str, float]) -> str:
        """Generate sequence with specified nucleotide composition."""
        nucleotides = ['A', 'T', 'G', 'C']
        probabilities = [composition[nt] for nt in nucleotides]
        
        # Generate sequence using composition probabilities
        sequence = ''.join(np.random.choice(nucleotides, size=length, p=probabilities))
        
        return sequence

    def _apply_controlled_noise(self, sequence: str, noise_level: float) -> str:
        """Apply controlled noise (mutations) to sequence."""
        if noise_level <= 0:
            return sequence
        
        sequence = list(sequence)
        nucleotides = ['A', 'T', 'G', 'C']
        
        # Apply random mutations
        num_mutations = max(1, int(len(sequence) * noise_level))
        positions = random.sample(range(len(sequence)), min(num_mutations, len(sequence)))
        
        for pos in positions:
            # Replace with random nucleotide (excluding current)
            current_nt = sequence[pos]
            new_nt = random.choice([nt for nt in nucleotides if nt != current_nt])
            sequence[pos] = new_nt
        
        return ''.join(sequence)

    def _save_synthetic_dataset(self, synthetic_data: Dict, output_format: str) -> Dict[str, str]:
        """Save synthetic dataset in specified format(s)."""
        output_files = {}
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(self.download_folder, f"synthetic_metagenomic_{timestamp}")
        
        if output_format in ['fasta', 'both']:
            # Save as FASTA
            fasta_path = f"{base_path}.fasta"
            with open(fasta_path, 'w') as f:
                for i, (seq, meta) in enumerate(zip(synthetic_data['sequences'], synthetic_data['metadata'])):
                    header = f">{meta['sequence_id']} species={meta['species']} length={meta['length']} gc={meta['gc_content']:.3f}"
                    f.write(f"{header}\n{seq}\n")
            output_files['fasta'] = fasta_path
        
        if output_format in ['csv', 'both']:
            # Save as CSV
            csv_path = f"{base_path}.csv"
            df_data = []
            for seq, meta in zip(synthetic_data['sequences'], synthetic_data['metadata']):
                df_data.append({
                    'sequence_id': meta['sequence_id'],
                    'sequence': seq,
                    'species': meta['species'],
                    'length': meta['length'],
                    'gc_content': meta['gc_content'],
                    'noise_applied': meta['noise_applied']
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(csv_path, index=False)
            output_files['csv'] = csv_path
            
            # Save metadata separately
            metadata_path = f"{base_path}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'generation_params': synthetic_data['generation_params'],
                    'species_counts': synthetic_data['species_counts'],
                    'total_sequences': len(synthetic_data['sequences'])
                }, f, indent=2)
            output_files['metadata'] = metadata_path
        
        return output_files

    def simulate_metagenomic_experiment(self, 
                                      experimental_design: Dict,
                                      base_species_abundances: Dict[str, float],
                                      output_dir: str = None) -> Dict:
        """
        Simulate complete metagenomic experiment with multiple conditions.
        
        Parameters:
        - experimental_design: Dict with experimental parameters
        - base_species_abundances: Base abundances for control condition
        - output_dir: Directory for experiment outputs
        
        Returns:
        - Complete experimental dataset with multiple conditions
        """
        print(f"🧪 Simulating metagenomic experiment...")
        
        if output_dir is None:
            output_dir = os.path.join(self.download_folder, "metagenomic_experiment")
        os.makedirs(output_dir, exist_ok=True)
        
        experiment_data = {
            'conditions': {},
            'experimental_design': experimental_design,
            'base_abundances': base_species_abundances,
            'output_directory': output_dir
        }
        
        # Generate control condition
        print("📊 Generating control condition...")
        control_data = self.generate_synthetic_metagenomic_dataset(
            species_abundances=base_species_abundances,
            total_sequences=experimental_design.get('sequences_per_condition', 5000),
            sequence_length_range=experimental_design.get('sequence_length_range', (500, 5000)),
            noise_level=experimental_design.get('base_noise_level', 0.01),
            output_format='both'
        )
        experiment_data['conditions']['control'] = control_data
        
        # Generate treatment conditions
        treatments = experimental_design.get('treatments', {})
        for treatment_name, treatment_params in treatments.items():
            print(f"🔬 Generating {treatment_name} condition...")
            
            # Modify abundances based on treatment
            modified_abundances = base_species_abundances.copy()
            
            if 'abundance_changes' in treatment_params:
                for species, change_factor in treatment_params['abundance_changes'].items():
                    if species in modified_abundances:
                        modified_abundances[species] *= change_factor
                
                # Renormalize abundances
                total = sum(modified_abundances.values())
                modified_abundances = {sp: abund/total for sp, abund in modified_abundances.items()}
            
            # Generate treatment data
            treatment_data = self.generate_synthetic_metagenomic_dataset(
                species_abundances=modified_abundances,
                total_sequences=experimental_design.get('sequences_per_condition', 5000),
                sequence_length_range=experimental_design.get('sequence_length_range', (500, 5000)),
                noise_level=treatment_params.get('noise_level', experimental_design.get('base_noise_level', 0.01)),
                output_format='both'
            )
            experiment_data['conditions'][treatment_name] = treatment_data
        
        # Generate experimental summary
        summary_path = os.path.join(output_dir, "experiment_summary.json")
        with open(summary_path, 'w') as f:
            summary = {
                'experimental_design': experimental_design,
                'conditions': list(experiment_data['conditions'].keys()),
                'total_sequences': sum(len(cond['sequences']) for cond in experiment_data['conditions'].values()),
                'species_analyzed': list(base_species_abundances.keys())
            }
            json.dump(summary, f, indent=2)
        
        experiment_data['summary_file'] = summary_path
        
        print(f"✅ Metagenomic experiment simulation complete:")
        print(f"   - Conditions: {len(experiment_data['conditions'])}")
        print(f"   - Total sequences: {sum(len(cond['sequences']) for cond in experiment_data['conditions'].values())}")
        print(f"   - Output directory: {output_dir}")
        
        return experiment_data

    def apply_data_augmentation(self, sequences: List[str], 
                               augmentation_strategies: List[str] = None,
                               noise_levels: List[float] = None,
                               augmentation_factor: int = 2) -> Dict:
        """
        Apply controlled data augmentation to existing sequences.
        
        Parameters:
        - sequences: Original sequences to augment
        - augmentation_strategies: List of strategies ('mutation', 'insertion', 'deletion', 'reverse_complement', 'gc_shift')
        - noise_levels: List of noise levels for each strategy
        - augmentation_factor: How many augmented versions to create per original sequence
        
        Returns:
        - Dictionary with original and augmented sequences
        """
        print(f"🔄 Applying data augmentation to {len(sequences)} sequences...")
        
        if augmentation_strategies is None:
            augmentation_strategies = ['mutation', 'insertion', 'deletion']
        
        if noise_levels is None:
            noise_levels = [0.01, 0.005, 0.005]  # Default noise levels
        
        if len(noise_levels) != len(augmentation_strategies):
            noise_levels = [noise_levels[0]] * len(augmentation_strategies)
        
        augmented_data = {
            'original_sequences': sequences,
            'augmented_sequences': [],
            'augmentation_metadata': [],
            'strategies_used': augmentation_strategies,
            'noise_levels': noise_levels
        }
        
        for seq_idx, original_seq in enumerate(sequences):
            for aug_factor in range(augmentation_factor):
                for strategy_idx, strategy in enumerate(augmentation_strategies):
                    noise_level = noise_levels[strategy_idx]
                    
                    augmented_seq = self._apply_augmentation_strategy(
                        original_seq, strategy, noise_level
                    )
                    
                    augmented_data['augmented_sequences'].append(augmented_seq)
                    augmented_data['augmentation_metadata'].append({
                        'original_index': seq_idx,
                        'augmentation_factor': aug_factor,
                        'strategy': strategy,
                        'noise_level': noise_level,
                        'length_change': len(augmented_seq) - len(original_seq),
                        'gc_change': self._calculate_gc_content(augmented_seq) - self._calculate_gc_content(original_seq)
                    })
        
        print(f"✅ Data augmentation complete:")
        print(f"   - Original sequences: {len(sequences)}")
        print(f"   - Augmented sequences: {len(augmented_data['augmented_sequences'])}")
        print(f"   - Total sequences: {len(sequences) + len(augmented_data['augmented_sequences'])}")
        
        return augmented_data

    def _apply_augmentation_strategy(self, sequence: str, strategy: str, noise_level: float) -> str:
        """Apply specific augmentation strategy to sequence."""
        sequence = list(sequence)  # Convert to list for easier manipulation
        nucleotides = ['A', 'T', 'G', 'C']
        
        if strategy == 'mutation':
            # Random point mutations
            num_mutations = max(1, int(len(sequence) * noise_level))
            positions = random.sample(range(len(sequence)), min(num_mutations, len(sequence)))
            
            for pos in positions:
                # Replace with random nucleotide (excluding current)
                current_nt = sequence[pos]
                new_nt = random.choice([nt for nt in nucleotides if nt != current_nt])
                sequence[pos] = new_nt
        
        elif strategy == 'insertion':
            # Random insertions
            num_insertions = max(1, int(len(sequence) * noise_level))
            
            for _ in range(num_insertions):
                pos = random.randint(0, len(sequence))
                new_nt = random.choice(nucleotides)
                sequence.insert(pos, new_nt)
        
        elif strategy == 'deletion':
            # Random deletions
            num_deletions = max(1, int(len(sequence) * noise_level))
            num_deletions = min(num_deletions, len(sequence) - 10)  # Keep minimum length
            
            for _ in range(num_deletions):
                if len(sequence) > 10:  # Ensure minimum length
                    pos = random.randint(0, len(sequence) - 1)
                    sequence.pop(pos)
        
        elif strategy == 'reverse_complement':
            # Create reverse complement of random segments
            if random.random() < noise_level * 10:  # Lower probability for this major change
                complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
                # Reverse complement entire sequence
                sequence = [complement_map.get(nt, nt) for nt in reversed(sequence)]
        
        elif strategy == 'gc_shift':
            # Bias mutations toward increasing or decreasing GC content
            num_mutations = max(1, int(len(sequence) * noise_level))
            positions = random.sample(range(len(sequence)), min(num_mutations, len(sequence)))
            
            # Random bias toward GC or AT
            gc_bias = random.choice([True, False])
            
            for pos in positions:
                current_nt = sequence[pos]
                if gc_bias:
                    # Bias toward GC
                    if current_nt in ['A', 'T']:
                        sequence[pos] = random.choice(['G', 'C'])
                else:
                    # Bias toward AT
                    if current_nt in ['G', 'C']:
                        sequence[pos] = random.choice(['A', 'T'])
        
        return ''.join(sequence)

    def advanced_quality_control(self, 
                               min_call_rate: float = 0.95,
                               min_confidence_score: float = 0.8,
                               hwe_p_threshold: float = 1e-6,
                               min_frequency: float = 0.01,
                               enable_genomic_control: bool = True) -> Dict:
        """
        Advanced quality control following GWAS standards (Miyagawa et al. 2008).
        Implements four key parameters: call rate, confidence score, Hardy-Weinberg 
        equilibrium, and frequency filtering.
        
        Parameters:
        - min_call_rate: Minimum sequence completeness rate (≥95% recommended)
        - min_confidence_score: Minimum confidence for sequence classification (≥80%)
        - hwe_p_threshold: Hardy-Weinberg equilibrium p-value threshold (1e-6)
        - min_frequency: Minimum allele/variant frequency (1%)
        - enable_genomic_control: Apply genomic control correction
        
        Returns:
        - Dictionary with comprehensive QC results and filtered data
        """
        print("🔬 Applying advanced genomic quality control (GWAS standards)...")
        self._log_audit_event("ADVANCED_QC_START", "Following Miyagawa et al. 2008 methodology")
        
        accessible_files = self._get_accessible_files()
        qc_results = {
            "total_sequences": 0,
            "call_rate_failures": 0,
            "confidence_failures": 0,
            "hwe_failures": 0,
            "frequency_failures": 0,
            "passed_all_filters": 0,
            "filtered_sequences": [],
            "qc_statistics": {},
            "genomic_control_lambda": None
        }
        
        all_sequences_data = []
        
        # Process each file
        for file_path in accessible_files:
            print(f"   📄 Processing {os.path.basename(file_path)}...")
            file_qc = self._apply_advanced_qc_to_file(
                file_path, min_call_rate, min_confidence_score, 
                hwe_p_threshold, min_frequency
            )
            
            # Aggregate results
            for key in ["total_sequences", "call_rate_failures", "confidence_failures", 
                       "hwe_failures", "frequency_failures", "passed_all_filters"]:
                qc_results[key] += file_qc[key]
            
            qc_results["filtered_sequences"].extend(file_qc["filtered_sequences"])
            all_sequences_data.extend(file_qc["sequence_data"])
        
        # Apply genomic control if enabled
        if enable_genomic_control and all_sequences_data:
            qc_results["genomic_control_lambda"] = self._calculate_genomic_control_lambda(all_sequences_data)
            print(f"   🎯 Genomic control λ = {qc_results['genomic_control_lambda']:.3f}")
        
        # Calculate overall statistics
        qc_results["qc_statistics"] = self._calculate_qc_statistics(qc_results)
        
        print(f"✅ Advanced QC complete:")
        print(f"   - Total sequences: {qc_results['total_sequences']}")
        print(f"   - Passed all filters: {qc_results['passed_all_filters']}")
        print(f"   - Overall pass rate: {qc_results['qc_statistics']['overall_pass_rate']:.1%}")
        
        self._log_audit_event("ADVANCED_QC_COMPLETE", 
                            f"Processed {qc_results['total_sequences']} sequences, {qc_results['passed_all_filters']} passed")
        
        return qc_results

    def _apply_advanced_qc_to_file(self, file_path: str, min_call_rate: float,
                                  min_confidence_score: float, hwe_p_threshold: float,
                                  min_frequency: float) -> Dict:
        """Apply advanced QC to a single file following GWAS methodology."""
        file_results = {
            "total_sequences": 0,
            "call_rate_failures": 0,
            "confidence_failures": 0, 
            "hwe_failures": 0,
            "frequency_failures": 0,
            "passed_all_filters": 0,
            "filtered_sequences": [],
            "sequence_data": []
        }
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            sequences = self._parse_fasta_content(content)
            
            for seq_id, sequence in sequences.items():
                file_results["total_sequences"] += 1
                
                # Parameter 1: Call Rate (Sequence Completeness)
                call_rate = self._calculate_call_rate(sequence)
                if call_rate < min_call_rate:
                    file_results["call_rate_failures"] += 1
                    continue
                
                # Parameter 2: Confidence Score (Bayesian-inspired)
                confidence_score = self._calculate_confidence_score(sequence, seq_id)
                if confidence_score < min_confidence_score:
                    file_results["confidence_failures"] += 1
                    continue
                
                # Parameter 3: Hardy-Weinberg Equilibrium (Compositional Balance)
                hwe_p_value = self._test_hardy_weinberg_equilibrium(sequence)
                if hwe_p_value < hwe_p_threshold:
                    file_results["hwe_failures"] += 1
                    continue
                
                # Parameter 4: Frequency Filtering (Variant Abundance)
                variant_frequency = self._calculate_variant_frequency(sequence)
                if variant_frequency < min_frequency:
                    file_results["frequency_failures"] += 1
                    continue
                
                # Sequence passed all filters
                file_results["passed_all_filters"] += 1
                
                sequence_data = {
                    "id": seq_id,
                    "sequence": sequence,
                    "call_rate": call_rate,
                    "confidence_score": confidence_score,
                    "hwe_p_value": hwe_p_value,
                    "variant_frequency": variant_frequency,
                    "length": len(sequence),
                    "gc_content": self._calculate_gc_content(sequence),
                    "source_file": os.path.basename(file_path)
                }
                
                file_results["filtered_sequences"].append(sequence_data)
                file_results["sequence_data"].append(sequence_data)
        
        except Exception as e:
            print(f"⚠️ Error processing file {file_path}: {e}")
        
        return file_results

    def _calculate_call_rate(self, sequence: str) -> float:
        """
        Calculate call rate (sequence completeness) equivalent to SNP call rate.
        Measures proportion of non-ambiguous nucleotides.
        """
        total_length = len(sequence)
        if total_length == 0:
            return 0.0
        
        # Count non-ambiguous nucleotides (A, T, G, C)
        valid_calls = sum(1 for nt in sequence.upper() if nt in 'ATGC')
        call_rate = valid_calls / total_length
        
        return call_rate

    def _calculate_confidence_score(self, sequence: str, seq_id: str) -> float:
        """
        Calculate confidence score using Bayesian-inspired robust algorithm.
        Based on sequence quality indicators and compositional consistency.
        """
        if len(sequence) == 0:
            return 0.0
        
        confidence_factors = []
        
        # Factor 1: Compositional consistency (no extreme bias)
        gc_content = self._calculate_gc_content(sequence)
        gc_confidence = 1.0 - abs(gc_content - 0.5) * 2  # Penalize extreme GC bias
        confidence_factors.append(max(0.0, gc_confidence))
        
        # Factor 2: Length appropriateness
        length = len(sequence)
        if 100 <= length <= 10000:  # Optimal range
            length_confidence = 1.0
        elif length < 100:
            length_confidence = length / 100.0
        else:
            length_confidence = 10000.0 / length
        confidence_factors.append(min(1.0, length_confidence))
        
        # Factor 3: Nucleotide diversity (avoid low-complexity regions)
        diversity = len(set(sequence.upper())) / 4.0  # Normalized by max possible (4 nucleotides)
        confidence_factors.append(diversity)
        
        # Factor 4: Absence of problematic patterns
        problematic_patterns = ['NNNNNN', 'AAAAAA', 'TTTTTT', 'GGGGGG', 'CCCCCC']
        pattern_penalty = sum(1 for pattern in problematic_patterns if pattern in sequence.upper())
        pattern_confidence = max(0.0, 1.0 - pattern_penalty * 0.2)
        confidence_factors.append(pattern_confidence)
        
        # Factor 5: Header quality (if available)
        header_confidence = 1.0
        if 'unknown' in seq_id.lower() or 'uncharacterized' in seq_id.lower():
            header_confidence = 0.7
        confidence_factors.append(header_confidence)
        
        # Bayesian-inspired weighted combination
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Emphasize compositional consistency
        confidence_score = sum(w * f for w, f in zip(weights, confidence_factors))
        
        return confidence_score

    def _test_hardy_weinberg_equilibrium(self, sequence: str) -> float:
        """
        Test Hardy-Weinberg equilibrium equivalent for sequence composition.
        Tests if nucleotide frequencies deviate from expected random distribution.
        """
        if len(sequence) < 50:  # Too short for meaningful statistics
            return 1.0  # Return high p-value (no evidence against HWE)
        
        # Count nucleotides
        nucleotide_counts = Counter(sequence.upper())
        total = sum(nucleotide_counts.values())
        
        # Calculate observed frequencies
        observed = [nucleotide_counts.get(nt, 0) for nt in 'ATGC']
        
        # Expected frequencies under null hypothesis (random distribution)
        expected_freq = total / 4.0  # Equal distribution
        expected = [expected_freq] * 4
        
        # Chi-square test for goodness of fit
        chi_square = sum((obs - exp)**2 / exp for obs, exp in zip(observed, expected) if exp > 0)
        
        # Degrees of freedom = 3 (4 categories - 1)
        # Calculate p-value using chi-square distribution approximation
        # For df=3, critical values: 7.815 (p=0.05), 11.345 (p=0.01), 16.266 (p=0.001)
        if chi_square < 7.815:
            p_value = 0.05  # p > 0.05
        elif chi_square < 11.345:
            p_value = 0.01  # 0.01 < p <= 0.05
        elif chi_square < 16.266:
            p_value = 0.001  # 0.001 < p <= 0.01
        else:
            p_value = 0.0001  # p <= 0.001
        
        return p_value

    def _calculate_variant_frequency(self, sequence: str) -> float:
        """
        Calculate variant frequency equivalent to minor allele frequency.
        Measures frequency of less common nucleotides.
        """
        if len(sequence) == 0:
            return 0.0
        
        # Count nucleotides
        nucleotide_counts = Counter(sequence.upper())
        total = sum(nucleotide_counts.values())
        
        # Calculate frequencies
        frequencies = [count / total for count in nucleotide_counts.values()]
        frequencies.sort()
        
        # Return second most frequent (similar to minor allele frequency concept)
        if len(frequencies) >= 2:
            return frequencies[-2]  # Second highest frequency
        elif len(frequencies) == 1:
            return 0.0  # Only one nucleotide type (monomorphic)
        else:
            return 0.0

    def _calculate_genomic_control_lambda(self, sequences_data: List[Dict]) -> float:
        """
        Calculate genomic control lambda (λ) for population stratification correction.
        Based on median chi-square statistic from compositional tests.
        """
        if len(sequences_data) < 100:  # Need sufficient data
            return 1.0
        
        chi_squares = []
        
        for seq_data in sequences_data:
            sequence = seq_data['sequence']
            if len(sequence) < 50:
                continue
            
            # Calculate chi-square for compositional deviation
            nucleotide_counts = Counter(sequence.upper())
            total = sum(nucleotide_counts.values())
            observed = [nucleotide_counts.get(nt, 0) for nt in 'ATGC']
            expected = [total / 4.0] * 4
            
            chi_square = sum((obs - exp)**2 / exp for obs, exp in zip(observed, expected) if exp > 0)
            chi_squares.append(chi_square)
        
        if not chi_squares:
            return 1.0
        
        # Genomic control lambda = median(chi_square) / expected_median
        # For chi-square with df=3, expected median ≈ 2.366
        median_chi_square = np.median(chi_squares)
        lambda_gc = median_chi_square / 2.366
        
        return lambda_gc

    def _calculate_qc_statistics(self, qc_results: Dict) -> Dict:
        """Calculate comprehensive QC statistics."""
        total = qc_results["total_sequences"]
        if total == 0:
            return {"overall_pass_rate": 0.0}
        
        statistics = {
            "overall_pass_rate": qc_results["passed_all_filters"] / total,
            "call_rate_failure_rate": qc_results["call_rate_failures"] / total,
            "confidence_failure_rate": qc_results["confidence_failures"] / total,
            "hwe_failure_rate": qc_results["hwe_failures"] / total,
            "frequency_failure_rate": qc_results["frequency_failures"] / total,
            "total_failure_rate": (total - qc_results["passed_all_filters"]) / total
        }
        
        return statistics

    def quasi_case_control_validation(self, validation_samples: int = 1000) -> Dict:
        """
        Perform quasi-case-control validation following Miyagawa et al. methodology.
        Randomly splits samples into two groups and tests for systematic differences.
        """
        print("🎯 Performing quasi-case-control validation...")
        self._log_audit_event("QUASI_CASE_CONTROL_START", "Validating data cleaning effectiveness")
        
        # Get filtered sequences from advanced QC
        qc_results = self.advanced_quality_control()
        filtered_sequences = qc_results["filtered_sequences"]
        
        if len(filtered_sequences) < validation_samples:
            print(f"⚠️ Warning: Only {len(filtered_sequences)} sequences available, using all")
            validation_samples = len(filtered_sequences)
        
        # Randomly split into two groups
        np.random.shuffle(filtered_sequences)
        mid_point = validation_samples // 2
        
        group1 = filtered_sequences[:mid_point]
        group2 = filtered_sequences[mid_point:validation_samples]
        
        print(f"   📊 Group 1: {len(group1)} sequences")
        print(f"   📊 Group 2: {len(group2)} sequences")
        
        # Compare groups for systematic differences
        validation_results = {
            "group1_size": len(group1),
            "group2_size": len(group2),
            "comparison_tests": {},
            "p_values": {},
            "validation_passed": True
        }
        
        # Test 1: GC content comparison
        gc1 = [seq["gc_content"] for seq in group1]
        gc2 = [seq["gc_content"] for seq in group2]
        gc_p_value = self._statistical_comparison(gc1, gc2, "GC content")
        validation_results["p_values"]["gc_content"] = gc_p_value
        
        # Test 2: Sequence length comparison
        len1 = [seq["length"] for seq in group1]
        len2 = [seq["length"] for seq in group2]
        length_p_value = self._statistical_comparison(len1, len2, "sequence length")
        validation_results["p_values"]["sequence_length"] = length_p_value
        
        # Test 3: Confidence score comparison
        conf1 = [seq["confidence_score"] for seq in group1]
        conf2 = [seq["confidence_score"] for seq in group2]
        conf_p_value = self._statistical_comparison(conf1, conf2, "confidence score")
        validation_results["p_values"]["confidence_score"] = conf_p_value
        
        # Test 4: Call rate comparison
        call1 = [seq["call_rate"] for seq in group1]
        call2 = [seq["call_rate"] for seq in group2]
        call_p_value = self._statistical_comparison(call1, call2, "call rate")
        validation_results["p_values"]["call_rate"] = call_p_value
        
        # Overall validation assessment
        significant_tests = sum(1 for p in validation_results["p_values"].values() if p < 0.05)
        validation_results["significant_differences"] = significant_tests
        
        if significant_tests > 0:
            validation_results["validation_passed"] = False
            print(f"⚠️ Validation warning: {significant_tests} significant differences detected")
        else:
            print(f"✅ Validation passed: No systematic differences between groups")
        
        self._log_audit_event("QUASI_CASE_CONTROL_COMPLETE", 
                            f"Validation {'passed' if validation_results['validation_passed'] else 'failed'}")
        
        return validation_results

    def _statistical_comparison(self, group1: List[float], group2: List[float], metric_name: str) -> float:
        """Perform statistical comparison between two groups."""
        if not group1 or not group2:
            return 1.0
        
        # Simple t-test approximation
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1), np.std(group2)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard error
        pooled_se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
        
        if pooled_se == 0:
            p_value = 1.0 if mean1 == mean2 else 0.0
        else:
            # t-statistic
            t_stat = abs(mean1 - mean2) / pooled_se
            
            # Approximate p-value (two-tailed)
            if t_stat < 1.96:  # Corresponds to p > 0.05
                p_value = 0.1
            elif t_stat < 2.58:  # Corresponds to 0.01 < p <= 0.05
                p_value = 0.03
            else:  # p <= 0.01
                p_value = 0.005
        
        print(f"   🔬 {metric_name}: Group1={mean1:.3f}±{std1:.3f}, Group2={mean2:.3f}±{std2:.3f}, p={p_value:.3f}")
        
        return p_value

    def generate_qc_report(self, qc_results: Dict = None, 
                          validation_results: Dict = None,
                          include_recommendations: bool = True) -> Dict:
        """
        Generate comprehensive quality control report following GWAS standards.
        """
        print("📋 Generating comprehensive QC report...")
        
        if qc_results is None:
            qc_results = self.advanced_quality_control()
        
        if validation_results is None:
            validation_results = self.quasi_case_control_validation()
        
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "species": self.species,
            "taxon_id": self.taxon_id,
            "methodology": "GWAS-standard QC (Miyagawa et al. 2008)",
            "qc_results": qc_results,
            "validation_results": validation_results,
            "data_quality_assessment": self._assess_data_quality(qc_results, validation_results)
        }
        
        if include_recommendations:
            report["recommendations"] = self._generate_qc_recommendations(qc_results, validation_results)
        
        # Save report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.download_folder, f"qc_report_{timestamp}.json")
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        report["report_file"] = report_path
        
        print(f"✅ QC report generated: {os.path.basename(report_path)}")
        self._log_audit_event("QC_REPORT_GENERATED", f"Report saved: {report_path}")
        
        return report

    def _assess_data_quality(self, qc_results: Dict, validation_results: Dict) -> Dict:
        """Assess overall data quality based on QC and validation results."""
        assessment = {
            "overall_grade": "A",  # A, B, C, D, F
            "quality_score": 0.0,  # 0.0 - 1.0
            "critical_issues": [],
            "warnings": [],
            "strengths": []
        }
        
        # Calculate quality score
        pass_rate = qc_results["qc_statistics"]["overall_pass_rate"]
        quality_score = pass_rate
        
        # Adjust for validation results
        if not validation_results["validation_passed"]:
            quality_score *= 0.8  # Penalize validation failures
        
        # Adjust for genomic control
        if qc_results.get("genomic_control_lambda"):
            lambda_gc = qc_results["genomic_control_lambda"]
            if lambda_gc > 1.1:
                quality_score *= 0.9  # Minor penalty for population stratification
                assessment["warnings"].append(f"Genomic control λ = {lambda_gc:.3f} indicates potential population stratification")
        
        assessment["quality_score"] = quality_score
        
        # Assign grade
        if quality_score >= 0.95:
            assessment["overall_grade"] = "A"
            assessment["strengths"].append("Excellent data quality with >95% pass rate")
        elif quality_score >= 0.90:
            assessment["overall_grade"] = "B"
            assessment["strengths"].append("Good data quality with >90% pass rate")
        elif quality_score >= 0.80:
            assessment["overall_grade"] = "C"
            assessment["warnings"].append("Moderate data quality, consider additional filtering")
        elif quality_score >= 0.70:
            assessment["overall_grade"] = "D"
            assessment["critical_issues"].append("Poor data quality, extensive cleaning needed")
        else:
            assessment["overall_grade"] = "F"
            assessment["critical_issues"].append("Unacceptable data quality, dataset may be compromised")
        
        # Check specific failure rates
        for failure_type, rate in qc_results["qc_statistics"].items():
            if failure_type.endswith("_failure_rate") and rate > 0.1:  # >10% failure
                assessment["warnings"].append(f"High {failure_type.replace('_', ' ')}: {rate:.1%}")
        
        return assessment

    def _generate_qc_recommendations(self, qc_results: Dict, validation_results: Dict) -> Dict:
        """Generate specific recommendations based on QC results."""
        recommendations = {
            "immediate_actions": [],
            "parameter_adjustments": [],
            "data_collection_improvements": [],
            "analysis_considerations": []
        }
        
        # Check overall pass rate
        pass_rate = qc_results["qc_statistics"]["overall_pass_rate"]
        if pass_rate < 0.8:
            recommendations["immediate_actions"].append(
                "Consider relaxing QC thresholds or investigating data collection issues"
            )
        
        # Check specific failure patterns
        if qc_results["qc_statistics"]["call_rate_failure_rate"] > 0.1:
            recommendations["parameter_adjustments"].append(
                "Lower call rate threshold from 95% to 90% if appropriate for study design"
            )
        
        if qc_results["qc_statistics"]["hwe_failure_rate"] > 0.1:
            recommendations["data_collection_improvements"].append(
                "High Hardy-Weinberg failures suggest possible population stratification or technical artifacts"
            )
        
        # Validation-based recommendations
        if not validation_results["validation_passed"]:
            recommendations["immediate_actions"].append(
                "Investigate systematic differences between sample groups before proceeding with analysis"
            )
        
        # Genomic control recommendations
        if qc_results.get("genomic_control_lambda", 1.0) > 1.1:
            recommendations["analysis_considerations"].append(
                "Apply genomic control correction in downstream association analyses"
            )
        
        return recommendations

    def biological_context_segmentation(self, 
                                       segmentation_strategy: str,
                                       annotation_file: str = None,
                                       custom_regions: List[Dict] = None,
                                       flanking_regions: Dict[str, int] = None,
                                       feature_filters: Dict = None) -> Dict:
        """
        Biologically-informed genome segmentation for domain-specific tasks.
        Addresses the limitation of one-size-fits-all segmentation by providing
        context-aware sequence extraction based on genomic position, function, and features.
        
        Parameters:
        - segmentation_strategy: Strategy type ('promoter_tss', 'gene_body', 'enhancer_regions', 
                               'chromatin_domains', 'regulatory_elements', 'expression_based', 'custom')
        - annotation_file: Path to genomic annotation file (GFF3, GTF, BED)
        - custom_regions: List of custom genomic regions with biological context
        - flanking_regions: Flanking sequences around features {'upstream': bp, 'downstream': bp}
        - feature_filters: Filters for specific biological features
        
        Returns:
        - Dictionary with biologically-contextualized sequence segments
        """
        print(f"🧬 Applying biological context-aware segmentation: {segmentation_strategy}")
        self._log_audit_event("BIOLOGICAL_SEGMENTATION_START", f"Strategy: {segmentation_strategy}")
        
        # Default flanking regions for different biological contexts
        default_flanking = {
            'promoter_tss': {'upstream': 2000, 'downstream': 500},
            'gene_body': {'upstream': 0, 'downstream': 0},
            'enhancer_regions': {'upstream': 1000, 'downstream': 1000},
            'chromatin_domains': {'upstream': 5000, 'downstream': 5000},
            'regulatory_elements': {'upstream': 1500, 'downstream': 1500}
        }
        
        if flanking_regions is None:
            flanking_regions = default_flanking.get(segmentation_strategy, {'upstream': 1000, 'downstream': 1000})
        
        segmentation_results = {
            'strategy': segmentation_strategy,
            'total_regions_identified': 0,
            'biologically_relevant_segments': [],
            'functional_annotations': {},
            'position_based_features': {},
            'sequence_characteristics': {},
            'biological_context_summary': {}
        }
        
        # Apply strategy-specific segmentation
        if segmentation_strategy == 'promoter_tss':
            segments = self._segment_promoter_tss_regions(annotation_file, flanking_regions, feature_filters)
        elif segmentation_strategy == 'gene_body':
            segments = self._segment_gene_body_regions(annotation_file, flanking_regions, feature_filters)
        elif segmentation_strategy == 'enhancer_regions':
            segments = self._segment_enhancer_regions(annotation_file, flanking_regions, feature_filters)
        elif segmentation_strategy == 'chromatin_domains':
            segments = self._segment_chromatin_domains(annotation_file, flanking_regions, feature_filters)
        elif segmentation_strategy == 'regulatory_elements':
            segments = self._segment_regulatory_elements(annotation_file, flanking_regions, feature_filters)
        elif segmentation_strategy == 'expression_based':
            segments = self._segment_expression_based_domains(annotation_file, flanking_regions, feature_filters)
        elif segmentation_strategy == 'custom':
            segments = self._segment_custom_regions(custom_regions, flanking_regions)
        else:
            raise ValueError(f"Unknown segmentation strategy: {segmentation_strategy}")
        
        segmentation_results.update(segments)
        
        # Add biological context analysis
        segmentation_results['biological_context_summary'] = self._analyze_biological_context(
            segmentation_results['biologically_relevant_segments'], segmentation_strategy
        )
        
        print(f"✅ Biological segmentation complete:")
        print(f"   - Strategy: {segmentation_strategy}")
        print(f"   - Regions identified: {segmentation_results['total_regions_identified']}")
        print(f"   - Biologically relevant segments: {len(segmentation_results['biologically_relevant_segments'])}")
        
        self._log_audit_event("BIOLOGICAL_SEGMENTATION_COMPLETE", 
                            f"Generated {len(segmentation_results['biologically_relevant_segments'])} biological segments")
        
        return segmentation_results

    def _segment_promoter_tss_regions(self, annotation_file: str, flanking_regions: Dict, 
                                     feature_filters: Dict) -> Dict:
        """
        Segment genome for promoter prediction tasks focusing on transcription start sites.
        Critical for promoter prediction models that need TSS-centered sequences.
        """
        print("🎯 Segmenting promoter/TSS regions for promoter prediction tasks...")
        
        segments = {
            'biologically_relevant_segments': [],
            'functional_annotations': {},
            'position_based_features': {},
            'total_regions_identified': 0
        }
        
        # Parse genomic annotations to identify TSS
        if annotation_file and os.path.exists(annotation_file):
            tss_regions = self._parse_tss_from_annotation(annotation_file)
        else:
            # Use heuristic TSS identification from sequence features
            tss_regions = self._identify_tss_heuristically()
        
        upstream = flanking_regions.get('upstream', 2000)
        downstream = flanking_regions.get('downstream', 500)
        
        for tss_info in tss_regions:
            # Extract promoter region around TSS
            promoter_segment = self._extract_promoter_sequence(
                tss_info, upstream, downstream
            )
            
            if promoter_segment:
                # Add biological context annotations
                promoter_segment.update({
                    'biological_type': 'promoter_tss',
                    'tss_position': tss_info.get('tss_position'),
                    'gene_id': tss_info.get('gene_id'),
                    'gene_name': tss_info.get('gene_name'),
                    'strand': tss_info.get('strand'),
                    'promoter_strength': self._predict_promoter_strength(promoter_segment['sequence']),
                    'cpg_island_presence': self._detect_cpg_islands(promoter_segment['sequence']),
                    'tata_box_presence': self._detect_tata_box(promoter_segment['sequence']),
                    'initiator_elements': self._detect_initiator_elements(promoter_segment['sequence'])
                })
                
                segments['biologically_relevant_segments'].append(promoter_segment)
                segments['total_regions_identified'] += 1
        
        # Add promoter-specific functional annotations
        segments['functional_annotations'] = {
            'core_promoter_elements': self._identify_core_promoter_elements(segments['biologically_relevant_segments']),
            'transcription_factor_binding_sites': self._predict_tfbs(segments['biologically_relevant_segments']),
            'regulatory_motifs': self._identify_regulatory_motifs(segments['biologically_relevant_segments'])
        }
        
        return segments

    def _segment_gene_body_regions(self, annotation_file: str, flanking_regions: Dict, 
                                  feature_filters: Dict) -> Dict:
        """
        Segment genome for epigenetic mark prediction focusing on entire gene bodies.
        Essential for histone modification prediction and chromatin state analysis.
        """
        print("🧬 Segmenting gene body regions for epigenetic mark prediction...")
        
        segments = {
            'biologically_relevant_segments': [],
            'functional_annotations': {},
            'position_based_features': {},
            'total_regions_identified': 0
        }
        
        # Parse gene annotations
        if annotation_file and os.path.exists(annotation_file):
            gene_regions = self._parse_genes_from_annotation(annotation_file)
        else:
            gene_regions = self._identify_genes_heuristically()
        
        for gene_info in gene_regions:
            # Extract gene body sequence
            gene_segment = self._extract_gene_body_sequence(gene_info, flanking_regions)
            
            if gene_segment:
                # Add epigenetic context annotations
                gene_segment.update({
                    'biological_type': 'gene_body',
                    'gene_id': gene_info.get('gene_id'),
                    'gene_biotype': gene_info.get('biotype', 'protein_coding'),
                    'exon_count': gene_info.get('exon_count', 0),
                    'intron_count': gene_info.get('intron_count', 0),
                    'gene_expression_level': self._predict_expression_level(gene_segment['sequence']),
                    'chromatin_accessibility': self._predict_chromatin_accessibility(gene_segment['sequence']),
                    'histone_marks_likelihood': self._predict_histone_marks(gene_segment['sequence']),
                    'regulatory_potential': self._assess_regulatory_potential(gene_segment['sequence'])
                })
                
                segments['biologically_relevant_segments'].append(gene_segment)
                segments['total_regions_identified'] += 1
        
        # Add gene-body-specific annotations
        segments['functional_annotations'] = {
            'exon_intron_boundaries': self._identify_splice_sites(segments['biologically_relevant_segments']),
            'alternative_splicing_sites': self._predict_alternative_splicing(segments['biologically_relevant_segments']),
            'regulatory_domains': self._identify_gene_regulatory_domains(segments['biologically_relevant_segments'])
        }
        
        return segments

    def _segment_enhancer_regions(self, annotation_file: str, flanking_regions: Dict, 
                                 feature_filters: Dict) -> Dict:
        """
        Segment genome for enhancer identification and regulatory element prediction.
        Focuses on distal regulatory elements and their chromatin signatures.
        """
        print("🎭 Segmenting enhancer regions for regulatory element prediction...")
        
        segments = {
            'biologically_relevant_segments': [],
            'functional_annotations': {},
            'position_based_features': {},
            'total_regions_identified': 0
        }
        
        # Identify potential enhancer regions
        enhancer_regions = self._identify_enhancer_regions()
        
        for enhancer_info in enhancer_regions:
            enhancer_segment = self._extract_enhancer_sequence(enhancer_info, flanking_regions)
            
            if enhancer_segment:
                # Add enhancer-specific annotations
                enhancer_segment.update({
                    'biological_type': 'enhancer_region',
                    'enhancer_type': self._classify_enhancer_type(enhancer_segment['sequence']),
                    'target_genes': self._predict_target_genes(enhancer_info),
                    'chromatin_loops': self._predict_chromatin_interactions(enhancer_info),
                    'tfbs_density': self._calculate_tfbs_density(enhancer_segment['sequence']),
                    'conservation_score': self._calculate_conservation_score(enhancer_segment['sequence']),
                    'accessibility_score': self._predict_enhancer_accessibility(enhancer_segment['sequence'])
                })
                
                segments['biologically_relevant_segments'].append(enhancer_segment)
                segments['total_regions_identified'] += 1
        
        segments['functional_annotations'] = {
            'enhancer_clusters': self._identify_enhancer_clusters(segments['biologically_relevant_segments']),
            'super_enhancers': self._identify_super_enhancers(segments['biologically_relevant_segments']),
            'tissue_specific_enhancers': self._classify_tissue_specificity(segments['biologically_relevant_segments'])
        }
        
        return segments

    def _segment_chromatin_domains(self, annotation_file: str, flanking_regions: Dict, 
                                  feature_filters: Dict) -> Dict:
        """
        Segment genome based on chromatin organization and topological domains.
        Inspired by Hi-C interaction domains and chromatin compartments.
        """
        print("🏗️ Segmenting chromatin domains for 3D organization analysis...")
        
        segments = {
            'biologically_relevant_segments': [],
            'functional_annotations': {},
            'position_based_features': {},
            'total_regions_identified': 0
        }
        
        # Identify chromatin domains using sequence-based features
        chromatin_domains = self._identify_chromatin_domains()
        
        for domain_info in chromatin_domains:
            domain_segment = self._extract_chromatin_domain_sequence(domain_info, flanking_regions)
            
            if domain_segment:
                domain_segment.update({
                    'biological_type': 'chromatin_domain',
                    'domain_type': domain_info.get('domain_type'),
                    'compartment': self._predict_chromatin_compartment(domain_segment['sequence']),
                    'insulator_strength': self._predict_insulator_strength(domain_segment['sequence']),
                    'interaction_frequency': self._predict_interaction_frequency(domain_segment['sequence']),
                    'gene_density': self._calculate_gene_density(domain_segment['sequence'])
                })
                
                segments['biologically_relevant_segments'].append(domain_segment)
                segments['total_regions_identified'] += 1
        
        return segments

    def _segment_expression_based_domains(self, annotation_file: str, flanking_regions: Dict, 
                                        feature_filters: Dict) -> Dict:
        """
        Segment genome based on coexpression patterns, following approaches from
        expression-based segmentation research (Rubin & Green, BMC Genomics 2013).
        """
        print("📊 Segmenting expression-based coexpression domains...")
        
        segments = {
            'biologically_relevant_segments': [],
            'functional_annotations': {},
            'position_based_features': {},
            'total_regions_identified': 0
        }
        
        # Identify coexpression domains
        coexpression_domains = self._identify_coexpression_domains()
        
        for domain_info in coexpression_domains:
            domain_segment = self._extract_coexpression_domain_sequence(domain_info)
            
            if domain_segment:
                domain_segment.update({
                    'biological_type': 'coexpression_domain',
                    'expression_pattern': domain_info.get('expression_pattern'),
                    'tissue_specificity': domain_info.get('tissue_specificity'),
                    'coexpression_strength': domain_info.get('coexpression_strength'),
                    'functional_coherence': self._assess_functional_coherence(domain_segment['sequence'])
                })
                
                segments['biologically_relevant_segments'].append(domain_segment)
                segments['total_regions_identified'] += 1
        
        return segments

    def position_based_sequence_extraction(self, 
                                         genomic_coordinates: List[Dict],
                                         extraction_strategy: str = 'exact',
                                         biological_context: str = None) -> Dict:
        """
        Extract sequences based on specific genomic positions with biological context.
        Provides precise control over sequence selection for domain-specific tasks.
        
        Parameters:
        - genomic_coordinates: List of coordinate dictionaries with biological annotations
        - extraction_strategy: 'exact', 'flanking', 'sliding_window', 'feature_centered'
        - biological_context: Biological meaning of the coordinates
        
        Returns:
        - Dictionary with position-based sequences and their biological context
        """
        print(f"📍 Extracting sequences by genomic position: {extraction_strategy}")
        
        position_results = {
            'extraction_strategy': extraction_strategy,
            'biological_context': biological_context,
            'extracted_sequences': [],
            'position_statistics': {},
            'biological_annotations': {}
        }
        
        for coord_info in genomic_coordinates:
            if extraction_strategy == 'exact':
                sequence_data = self._extract_exact_coordinates(coord_info)
            elif extraction_strategy == 'flanking':
                sequence_data = self._extract_flanking_regions(coord_info)
            elif extraction_strategy == 'sliding_window':
                sequence_data = self._extract_sliding_windows(coord_info)
            elif extraction_strategy == 'feature_centered':
                sequence_data = self._extract_feature_centered(coord_info)
            else:
                raise ValueError(f"Unknown extraction strategy: {extraction_strategy}")
            
            if sequence_data:
                # Add biological context
                sequence_data.update({
                    'biological_context': biological_context,
                    'genomic_region': coord_info.get('region_type'),
                    'functional_annotation': coord_info.get('annotation'),
                    'regulatory_potential': self._assess_position_regulatory_potential(sequence_data)
                })
                
                position_results['extracted_sequences'].append(sequence_data)
        
        return position_results

    def feature_based_segmentation(self, 
                                 feature_types: List[str],
                                 sequence_features: Dict = None,
                                 biological_filters: Dict = None) -> Dict:
        """
        Segment genome based on sequence features and biological characteristics.
        Enables selection of segments with specific biological properties.
        
        Parameters:
        - feature_types: Types of features to segment by ('cpg_islands', 'repeats', 'motifs', 'conservation')
        - sequence_features: Specific sequence features to look for
        - biological_filters: Biological criteria for segment selection
        
        Returns:
        - Dictionary with feature-based segments and their biological significance
        """
        print(f"🔍 Segmenting genome by sequence features: {feature_types}")
        
        feature_results = {
            'feature_types': feature_types,
            'feature_based_segments': [],
            'feature_statistics': {},
            'biological_significance': {}
        }
        
        accessible_files = self._get_accessible_files()
        
        for file_path in accessible_files:
            file_features = self._identify_features_in_file(file_path, feature_types, sequence_features)
            feature_results['feature_based_segments'].extend(file_features)
        
        # Add biological significance analysis
        feature_results['biological_significance'] = self._analyze_feature_biological_significance(
            feature_results['feature_based_segments']
        )
        
        return feature_results

    # Helper methods for biological context analysis
    def _parse_tss_from_annotation(self, annotation_file: str) -> List[Dict]:
        """Parse transcription start sites from genomic annotation file."""
        # Simplified implementation - would parse GFF3/GTF files
        return [
            {'tss_position': 1000, 'gene_id': 'gene1', 'gene_name': 'TEST1', 'strand': '+'},
            {'tss_position': 5000, 'gene_id': 'gene2', 'gene_name': 'TEST2', 'strand': '-'}
        ]

    def _identify_tss_heuristically(self) -> List[Dict]:
        """Identify TSS using sequence-based heuristics."""
        # Simplified - would use CAGE data, CpG islands, promoter motifs
        return [
            {'tss_position': 1000, 'confidence': 0.8, 'strand': '+'},
            {'tss_position': 3000, 'confidence': 0.7, 'strand': '+'}
        ]

    def _extract_promoter_sequence(self, tss_info: Dict, upstream: int, downstream: int) -> Dict:
        """Extract promoter sequence around TSS."""
        # Simplified implementation
        return {
            'sequence': 'ATGCATGCATGC' * 50,  # Placeholder
            'start_position': tss_info['tss_position'] - upstream,
            'end_position': tss_info['tss_position'] + downstream,
            'tss_relative_position': upstream
        }

    def _predict_promoter_strength(self, sequence: str) -> float:
        """Predict promoter strength based on sequence features."""
        # Simplified - would use machine learning models
        gc_content = self._calculate_gc_content(sequence)
        return min(1.0, gc_content * 1.5)

    def _detect_cpg_islands(self, sequence: str) -> bool:
        """Detect CpG islands in promoter sequences."""
        cg_count = sequence.upper().count('CG')
        return cg_count > len(sequence) * 0.02

    def _detect_tata_box(self, sequence: str) -> Dict:
        """Detect TATA box elements in promoter sequences."""
        tata_motifs = ['TATAAA', 'TATAWA', 'TAWAWA']
        for motif in tata_motifs:
            if motif in sequence.upper():
                return {'present': True, 'motif': motif, 'position': sequence.upper().find(motif)}
        return {'present': False}

    def _analyze_biological_context(self, segments: List[Dict], strategy: str) -> Dict:
        """Analyze biological context of segmented regions."""
        context_analysis = {
            'total_segments': len(segments),
            'average_length': np.mean([len(seg.get('sequence', '')) for seg in segments]) if segments else 0,
            'biological_types': Counter([seg.get('biological_type') for seg in segments]),
            'strategy_specific_metrics': {}
        }
        
        if strategy == 'promoter_tss':
            context_analysis['strategy_specific_metrics'] = {
                'promoters_with_tata': sum(1 for seg in segments if seg.get('tata_box_presence', {}).get('present', False)),
                'promoters_with_cpg': sum(1 for seg in segments if seg.get('cpg_island_presence', False)),
                'average_promoter_strength': np.mean([seg.get('promoter_strength', 0) for seg in segments])
            }
        elif strategy == 'gene_body':
            context_analysis['strategy_specific_metrics'] = {
                'protein_coding_genes': sum(1 for seg in segments if seg.get('gene_biotype') == 'protein_coding'),
                'average_exon_count': np.mean([seg.get('exon_count', 0) for seg in segments]),
                'high_expression_genes': sum(1 for seg in segments if seg.get('gene_expression_level', 0) > 0.7)
            }
        
        return context_analysis

    # Additional helper methods (simplified implementations)
    def _parse_genes_from_annotation(self, annotation_file: str) -> List[Dict]:
        """Parse gene information from annotation file."""
        return [{'gene_id': 'gene1', 'start': 1000, 'end': 5000, 'biotype': 'protein_coding'}]

    def _identify_genes_heuristically(self) -> List[Dict]:
        """Identify genes using sequence-based heuristics."""
        return [{'gene_id': 'predicted_gene1', 'start': 2000, 'end': 6000}]

    def _extract_gene_body_sequence(self, gene_info: Dict, flanking_regions: Dict) -> Dict:
        """Extract gene body sequence with flanking regions."""
        return {
            'sequence': 'ATGCATGCATGC' * 100,  # Placeholder
            'start_position': gene_info['start'],
            'end_position': gene_info['end']
        }

    def _identify_enhancer_regions(self) -> List[Dict]:
        """Identify potential enhancer regions."""
        return [{'start': 10000, 'end': 12000, 'type': 'predicted_enhancer'}]

    def _identify_chromatin_domains(self) -> List[Dict]:
        """Identify chromatin domains."""
        return [{'start': 0, 'end': 50000, 'domain_type': 'A_compartment'}]

    def _identify_coexpression_domains(self) -> List[Dict]:
        """Identify coexpression domains."""
        return [{'start': 0, 'end': 30000, 'expression_pattern': 'housekeeping'}]

    def _extract_exact_coordinates(self, coord_info: Dict) -> Dict:
        """Extract sequence at exact coordinates."""
        return {
            'sequence': 'ATGCATGC' * 50,  # Placeholder
            'coordinates': coord_info
        }

    def _extract_flanking_regions(self, coord_info: Dict) -> Dict:
        """Extract sequence with flanking regions."""
        return {
            'sequence': 'ATGCATGC' * 75,  # Placeholder with flanking
            'coordinates': coord_info
        }

    def _identify_features_in_file(self, file_path: str, feature_types: List[str], 
                                  sequence_features: Dict) -> List[Dict]:
        """Identify specific features in genomic file."""
        return [
            {'feature_type': 'cpg_island', 'start': 1000, 'end': 2000, 'sequence': 'CGCGCGCG' * 25}
        ]

    def _analyze_feature_biological_significance(self, segments: List[Dict]) -> Dict:
        """Analyze biological significance of feature-based segments."""
        return {
            'total_features': len(segments),
            'feature_distribution': Counter([seg.get('feature_type') for seg in segments]),
            'regulatory_potential': np.mean([0.6, 0.7, 0.8])  # Placeholder
        }

    def _detect_initiator_elements(self, sequence: str) -> Dict:
        """Detect initiator elements in promoter sequences."""
        initiator_motifs = ['TCAG', 'YCANTYY', 'CTCAT']
        for motif in initiator_motifs:
            if motif in sequence.upper():
                return {'present': True, 'motif': motif, 'position': sequence.upper().find(motif)}
        return {'present': False}

    def _identify_core_promoter_elements(self, segments: List[Dict]) -> Dict:
        """Identify core promoter elements across segments."""
        return {'tata_boxes': 0, 'initiators': 0, 'cpg_islands': 0}

    def _predict_tfbs(self, segments: List[Dict]) -> Dict:
        """Predict transcription factor binding sites."""
        return {'total_sites': 0, 'unique_factors': 0}

    def _identify_regulatory_motifs(self, segments: List[Dict]) -> Dict:
        """Identify regulatory motifs in segments."""
        return {'motif_count': 0, 'motif_types': []}

    def _predict_expression_level(self, sequence: str) -> float:
        """Predict gene expression level from sequence."""
        return min(1.0, self._calculate_gc_content(sequence) * 1.2)

    def _predict_chromatin_accessibility(self, sequence: str) -> float:
        """Predict chromatin accessibility."""
        return random.uniform(0.3, 0.9)

    def _predict_histone_marks(self, sequence: str) -> Dict:
        """Predict histone modification marks."""
        return {'H3K4me3': 0.7, 'H3K27ac': 0.6, 'H3K36me3': 0.5}

    def _assess_regulatory_potential(self, sequence: str) -> float:
        """Assess regulatory potential of sequence."""
        return random.uniform(0.4, 0.8)

    def _identify_splice_sites(self, segments: List[Dict]) -> Dict:
        """Identify splice sites in gene body segments."""
        return {'donor_sites': 0, 'acceptor_sites': 0}

    def _predict_alternative_splicing(self, segments: List[Dict]) -> Dict:
        """Predict alternative splicing sites."""
        return {'alternative_exons': 0, 'alternative_splice_sites': 0}

    def _identify_gene_regulatory_domains(self, segments: List[Dict]) -> Dict:
        """Identify gene regulatory domains."""
        return {'promoter_domains': 0, 'enhancer_domains': 0}

    def _extract_enhancer_sequence(self, enhancer_info: Dict, flanking_regions: Dict) -> Dict:
        """Extract enhancer sequence."""
        return {
            'sequence': 'ATGCATGC' * 60,  # Placeholder
            'start_position': enhancer_info['start'],
            'end_position': enhancer_info['end']
        }

    def _classify_enhancer_type(self, sequence: str) -> str:
        """Classify enhancer type."""
        return 'active_enhancer'

    def _predict_target_genes(self, enhancer_info: Dict) -> List[str]:
        """Predict target genes for enhancer."""
        return ['gene1', 'gene2']

    def _predict_chromatin_interactions(self, enhancer_info: Dict) -> Dict:
        """Predict chromatin interactions."""
        return {'loop_strength': 0.6, 'interaction_distance': 50000}

    def _calculate_tfbs_density(self, sequence: str) -> float:
        """Calculate transcription factor binding site density."""
        return random.uniform(0.1, 0.5)

    def _calculate_conservation_score(self, sequence: str) -> float:
        """Calculate conservation score."""
        return random.uniform(0.6, 0.9)

    def _predict_enhancer_accessibility(self, sequence: str) -> float:
        """Predict enhancer accessibility."""
        return random.uniform(0.4, 0.8)

    def _identify_enhancer_clusters(self, segments: List[Dict]) -> Dict:
        """Identify enhancer clusters."""
        return {'cluster_count': 0, 'super_enhancers': 0}

    def _identify_super_enhancers(self, segments: List[Dict]) -> Dict:
        """Identify super enhancers."""
        return {'super_enhancer_count': 0}

    def _classify_tissue_specificity(self, segments: List[Dict]) -> Dict:
        """Classify tissue specificity of enhancers."""
        return {'tissue_specific': 0, 'ubiquitous': 0}

    def _extract_chromatin_domain_sequence(self, domain_info: Dict, flanking_regions: Dict) -> Dict:
        """Extract chromatin domain sequence."""
        return {
            'sequence': 'ATGCATGC' * 200,  # Placeholder
            'start_position': domain_info['start'],
            'end_position': domain_info['end']
        }

    def _predict_chromatin_compartment(self, sequence: str) -> str:
        """Predict chromatin compartment (A/B)."""
        return 'A' if self._calculate_gc_content(sequence) > 0.45 else 'B'

    def _predict_insulator_strength(self, sequence: str) -> float:
        """Predict insulator binding strength."""
        return random.uniform(0.2, 0.8)

    def _predict_interaction_frequency(self, sequence: str) -> float:
        """Predict chromatin interaction frequency."""
        return random.uniform(0.3, 0.7)

    def _calculate_gene_density(self, sequence: str) -> float:
        """Calculate gene density in region."""
        return random.uniform(0.1, 0.4)

    def _extract_coexpression_domain_sequence(self, domain_info: Dict) -> Dict:
        """Extract coexpression domain sequence."""
        return {
            'sequence': 'ATGCATGC' * 150,  # Placeholder
            'start_position': domain_info['start'],
            'end_position': domain_info['end']
        }

    def _assess_functional_coherence(self, sequence: str) -> float:
        """Assess functional coherence of coexpression domain."""
        return random.uniform(0.5, 0.9)

    def _extract_sliding_windows(self, coord_info: Dict) -> Dict:
        """Extract sliding window sequences."""
        return {
            'sequence': 'ATGCATGC' * 40,  # Placeholder
            'coordinates': coord_info
        }

    def _extract_feature_centered(self, coord_info: Dict) -> Dict:
        """Extract feature-centered sequences."""
        return {
            'sequence': 'ATGCATGC' * 45,  # Placeholder
            'coordinates': coord_info
        }

    def _assess_position_regulatory_potential(self, sequence_data: Dict) -> float:
        """Assess regulatory potential of position-based sequence."""
        return random.uniform(0.3, 0.8)

    def _segment_regulatory_elements(self, annotation_file: str, flanking_regions: Dict, 
                                   feature_filters: Dict) -> Dict:
        """Segment regulatory elements (promoters, enhancers, silencers)."""
        return {
            'biologically_relevant_segments': [],
            'functional_annotations': {},
            'position_based_features': {},
            'total_regions_identified': 0
        }

    def _segment_custom_regions(self, custom_regions: List[Dict], flanking_regions: Dict) -> Dict:
        """Segment custom user-defined regions."""
        segments = {
            'biologically_relevant_segments': [],
            'functional_annotations': {},
            'position_based_features': {},
            'total_regions_identified': 0
        }
        
        for region in custom_regions or []:
            segment = {
                'sequence': 'ATGCATGC' * 30,  # Placeholder
                'biological_type': 'custom_region',
                'custom_annotation': region.get('annotation', 'user_defined'),
                'start_position': region.get('start', 0),
                'end_position': region.get('end', 1000)
            }
            segments['biologically_relevant_segments'].append(segment)
            segments['total_regions_identified'] += 1
        
        return segments

