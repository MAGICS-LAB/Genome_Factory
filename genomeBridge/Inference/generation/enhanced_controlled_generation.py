"""
Enhanced Controlled Generation for GenomeBridge
Extends GenomeBridge sequence generation control mechanisms

Adds the following control features based on existing models:
- GC content control (global and local)
- Motif inclusion/avoidance control
- Diversity vs fidelity balance
- Composition control
- Precise length control

Usage:
python enhanced_controlled_generation.py --model evo2 --gc_min 0.4 --gc_max 0.6 --avoid_motifs GAATTC
"""

import pandas as pd
import os
import argparse
import torch
import random
import re
import numpy as np
import pathlib
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

# Import existing GenomeBridge models
try:
    from evo import Evo, generate as evo_generate
    EVO_AVAILABLE = True
except ImportError:
    EVO_AVAILABLE = False
    print("Evo models not available")

try:
    from evo2 import Evo2  
    EVO2_AVAILABLE = True
except ImportError:
    EVO2_AVAILABLE = False
    print("Evo2 model not available")

try:
    from genomeocean.generation import SequenceGenerator
    GENOMEOCEAN_AVAILABLE = True
except ImportError:
    GENOMEOCEAN_AVAILABLE = False
    print("GenomeOcean model not available")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    GENERATOR_AVAILABLE = True
except ImportError:
    GENERATOR_AVAILABLE = False
    print("Generator model not available")

try:
    from genslm import GenSLM
    # Fix for PyTorch 2.6 compatibility with GenSLM
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    GENSLM_AVAILABLE = True
except ImportError:
    GENSLM_AVAILABLE = False
    print("GenSLM model not available")

class EnhancedGenomeBridgeGenerator:
    """
    Enhanced GenomeBridge sequence generator
    Adds biological constraint control on top of existing models
    """
    
    def __init__(self, base_model: str = "probabilistic", device: str = "auto"):
        """
        Initialize enhanced generator
        
        Parameters:
        - base_model: Base model ("evo1", "evo2", "genomeocean", "generator", "genslm", "probabilistic") 
        - device: Computing device
        """
        self.base_model = base_model
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        
        # Control parameter storage
        self.constraints = {
            'gc_content': None,
            'motifs': {'avoid': [], 'require': []},
            'composition': None,
            'length': {'target': 1000, 'min': None, 'max': None},
            'diversity': {'temperature': 1.0, 'top_k': None, 'top_p': None}
        }
        
        # Load base model
        self.model = None
        self.tokenizer = None
        self._load_base_model()
        
        print(f"🧬 Enhanced GenomeBridge Generator initialized")
        print(f"   Base model: {base_model}")
        print(f"   Device: {self.device}")
    
    def _load_base_model(self):
        """Load the specified base generation model"""
        try:
            if self.base_model == "evo1" and EVO_AVAILABLE:
                print("Loading Evo1 model...")
                evo_model = Evo('evo-1-131k-base')
                self.model, self.tokenizer = evo_model.model, evo_model.tokenizer
                self.model.to(self.device)
                self.model.eval()
                
            elif self.base_model == "evo2" and EVO2_AVAILABLE:
                print("Loading Evo2 model...")
                self.model = Evo2('evo2_7b')
                
            elif self.base_model == "generator" and GENERATOR_AVAILABLE:
                print("Loading Generator model...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "GenerTeam/GENERator-eukaryote-3b-base", 
                    trust_remote_code=True, 
                    cache_dir="/projects/p32572"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    "GenerTeam/GENERator-eukaryote-3b-base", 
                    cache_dir="/projects/p32572"
                )
                self.model = self.model.to(self.device)
                
            elif self.base_model == "genomeocean" and GENOMEOCEAN_AVAILABLE:
                print("GenomeOcean model will be initialized per call...")
                
            elif self.base_model == "genslm" and GENSLM_AVAILABLE:
                print("Loading GenSLM model...")
                self.model = GenSLM("genslm_2.5B_patric", model_cache_dir="/projects/p32572")
                self.model.eval()
                self.model.to(self.device)
                
            else:
                print(f"Base model {self.base_model} not available, using probabilistic fallback")
                self.base_model = "probabilistic"
                
        except Exception as e:
            print(f"Error loading base model: {e}")
            print("Falling back to probabilistic generation")
            self.base_model = "probabilistic"
    
    def set_gc_content_constraint(self, min_gc: float = 0.3, max_gc: float = 0.7, 
                                window_size: int = 100, enforce_locally: bool = True):
        """
        Set GC content constraints
        
        Parameters:
        - min_gc: Minimum GC content (0.0-1.0)
        - max_gc: Maximum GC content (0.0-1.0)  
        - window_size: Local check window size
        - enforce_locally: Whether to enforce local constraints
        """
        self.constraints['gc_content'] = {
            'min': min_gc,
            'max': max_gc,
            'window_size': window_size,
            'local': enforce_locally
        }
        print(f"✅ GC content constraint: {min_gc:.1%} - {max_gc:.1%}")
        return self
    
    def set_motif_constraints(self, avoid_motifs: List[str] = None, 
                            require_motifs: List[str] = None):
        """
        Set motif constraints
        
        Parameters:
        - avoid_motifs: Motif sequences to avoid (e.g., restriction sites)
        - require_motifs: Motif sequences that must be present (e.g., regulatory elements)
        """
        if avoid_motifs:
            self.constraints['motifs']['avoid'] = [m.upper() for m in avoid_motifs]
            print(f"🚫 Avoiding motifs: {avoid_motifs}")
        
        if require_motifs:
            self.constraints['motifs']['require'] = [m.upper() for m in require_motifs]  
            print(f"✅ Requiring motifs: {require_motifs}")
        
        return self
    
    def set_composition_constraint(self, nucleotide_frequencies: Dict[str, float]):
        """
        Set nucleotide composition constraints
        
        Parameters:
        - nucleotide_frequencies: Target nucleotide frequencies {'A': 0.25, 'T': 0.25, 'G': 0.25, 'C': 0.25}
        """
        # Normalize frequencies
        total = sum(nucleotide_frequencies.values())
        normalized = {nt: freq/total for nt, freq in nucleotide_frequencies.items()}
        
        self.constraints['composition'] = normalized
        print(f"🧮 Composition constraint: {normalized}")
        return self
    
    def set_diversity_parameters(self, temperature: float = 1.0, top_k: int = None, 
                               top_p: float = None):
        """
        Set diversity vs fidelity parameters
        
        Parameters:
        - temperature: Sampling temperature (higher = more diverse)
        - top_k: Top-k sampling parameter
        - top_p: Nucleus sampling parameter
        """
        self.constraints['diversity'] = {
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p
        }
        print(f"🎯 Diversity parameters: temp={temperature}, top_k={top_k}, top_p={top_p}")
        return self
    
    def set_length_constraint(self, target_length: int, min_length: int = None, 
                            max_length: int = None):
        """
        Set length constraints
        
        Parameters:
        - target_length: Target length
        - min_length: Minimum length
        - max_length: Maximum length
        """
        self.constraints['length'] = {
            'target': target_length,
            'min': min_length or target_length - 100,
            'max': max_length or target_length + 100
        }
        print(f"📏 Length constraint: {self.constraints['length']}")
        return self
    
    def generate_controlled_sequences(self, num_sequences: int = 100, 
                                    user_prompts: List[str] = None,
                                    max_attempts: int = 50) -> List[Dict]:
        """
        Generate controlled sequences
        
        Parameters:
        - num_sequences: Number of sequences to generate
        - user_prompts: User-provided starting sequences
        - max_attempts: Maximum attempts per sequence
        
        Returns:
        - List of sequence data dictionaries containing sequences and metadata
        """
        print(f"🔬 Generating {num_sequences} controlled sequences...")
        
        if user_prompts is None:
            user_prompts = self._generate_intelligent_prompts(num_sequences)
        
        generated_sequences = []
        successful_count = 0
        
        for i in range(num_sequences):
            prompt = user_prompts[i % len(user_prompts)]
            
            for attempt in range(max_attempts):
                try:
                    # Generate base sequence using base model
                    sequence = self._generate_base_sequence(prompt)
                    
                    # Apply constraint optimization
                    optimized_sequence = self._apply_constraints(sequence)
                    
                    # Validate constraint satisfaction
                    if self._validate_constraints(optimized_sequence):
                        seq_data = self._create_sequence_record(optimized_sequence, prompt, i)
                        generated_sequences.append(seq_data)
                        successful_count += 1
                        break
                        
                except Exception as e:
                    continue
            else:
                # If all attempts failed, generate fallback sequence
                fallback_seq = self._generate_fallback_sequence(prompt)
                seq_data = self._create_sequence_record(fallback_seq, prompt, i, is_fallback=True)
                generated_sequences.append(seq_data)
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{num_sequences}")
        
        success_rate = (successful_count / num_sequences) * 100
        print(f"✅ Generation complete: {success_rate:.1f}% satisfied constraints")
        
        return generated_sequences
    
    def _generate_intelligent_prompts(self, num_sequences: int) -> List[str]:
        """Generate intelligent prompt sequences based on constraints"""
        prompts = []
        
        # Base prompts
        base_prompts = ["A", "T", "G", "C"]
        
        # Generate biased prompts based on composition constraints
        if self.constraints['composition']:
            comp = self.constraints['composition']
            nucleotides = ['A', 'T', 'G', 'C']
            weights = [comp.get(nt, 0.25) for nt in nucleotides]
            
            # Generate biased prompts
            for _ in range(num_sequences // 4):
                biased_prompt = ''.join(np.random.choice(nucleotides, size=5, p=weights))
                prompts.append(biased_prompt)
        
        # Generate prompts based on required motifs
        if self.constraints['motifs']['require']:
            for motif in self.constraints['motifs']['require']:
                prompts.extend([motif[:6]] * (num_sequences // 20))
        
        # Fill remaining prompts
        while len(prompts) < num_sequences:
            prompts.extend(base_prompts)
        
        return prompts[:num_sequences]
    
    def _generate_base_sequence(self, prompt: str) -> str:
        """Generate base sequence using the specified model"""
        target_length = self.constraints['length']['target']
        diversity = self.constraints['diversity']
        
        if self.base_model == "evo1" and self.model is not None:
            return self._generate_with_evo1(prompt, target_length, diversity)
        elif self.base_model == "evo2" and self.model is not None:
            return self._generate_with_evo2(prompt, target_length, diversity)
        elif self.base_model == "generator" and self.model is not None:
            return self._generate_with_generator(prompt, target_length, diversity)
        elif self.base_model == "genomeocean":
            return self._generate_with_genomeocean(prompt, target_length, diversity)
        elif self.base_model == "genslm" and self.model is not None:
            return self._generate_with_genslm(prompt, target_length, diversity)
        else:
            return self._generate_probabilistic(prompt, target_length)
    
    def _generate_with_evo1(self, prompt: str, length: int, diversity: Dict) -> str:
        """Generate using Evo1 model with parameter override support"""
        try:
            output_seqs, _ = evo_generate(
                [prompt],
                self.model,
                self.tokenizer,
                n_tokens=length,
                temperature=diversity.get('temperature', 1.0),
                top_k=diversity.get('top_k', 4),
                top_p=diversity.get('top_p', 1.0),
                device=str(self.device),
                verbose=0
            )
            return output_seqs[0][:length]
        except Exception as e:
            print(f"Evo1 generation failed: {e}")
            return self._generate_probabilistic(prompt, length)
    
    def _generate_with_evo2(self, prompt: str, length: int, diversity: Dict) -> str:
        """Generate using Evo2 model with parameter override support"""
        try:
            output = self.model.generate(
                prompt_seqs=[prompt],
                n_tokens=length,
                temperature=diversity.get('temperature', 1.0),
                top_k=diversity.get('top_k', 4)
            )
            return output.sequences[0][:length]
        except Exception as e:
            print(f"Evo2 generation failed: {e}")
            return self._generate_probabilistic(prompt, length)
    
    def _generate_with_generator(self, prompt: str, length: int, diversity: Dict) -> str:
        """Generate using Generator model with parameter override support"""
        try:
            # Apply 6-mer alignment (Generator-specific requirement)
            def left_truncation(seq, multiple=6):
                remainder = len(seq) % multiple
                return seq[remainder:] if remainder != 0 else seq
            
            processed_prompt = self.tokenizer.bos_token + left_truncation(prompt)
            
            inputs = self.tokenizer(
                [processed_prompt],
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=length,
                    temperature=diversity.get('temperature', 2.0),
                    top_k=diversity.get('top_k', -1),
                    do_sample=True
                )
            
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return decoded[:length]
        except Exception as e:
            print(f"Generator generation failed: {e}")
            return self._generate_probabilistic(prompt, length)
    
    def _generate_with_genomeocean(self, prompt: str, length: int, diversity: Dict) -> str:
        """Generate using GenomeOcean model"""
        try:
            # Create temporary prompt file
            pd.DataFrame([prompt]).to_csv('tmp_prompt.csv', sep='\t', header=None, index=False)
            
            seq_gen = SequenceGenerator(
                model_dir='pGenomeOcean/GenomeOcean-4B',
                promptfile='tmp_prompt.csv',
                num=1,
                min_seq_len=length,
                max_seq_len=length,
                temperature=diversity.get('temperature', 1.0),
                presence_penalty=0.5,
                frequency_penalty=0.5,
                repetition_penalty=1.0
            )
            
            g_seqs = seq_gen.generate_sequences(prepend_prompt_to_output=False)
            
            # Cleanup temporary file
            if os.path.exists('tmp_prompt.csv'):
                os.remove('tmp_prompt.csv')
            
            return g_seqs['seq'].iloc[0][:length]
        except Exception as e:
            print(f"GenomeOcean generation failed: {e}")
            return self._generate_probabilistic(prompt, length)
    
    def _generate_with_genslm(self, prompt: str, length: int, diversity: Dict) -> str:
        """Generate using GenSLM model with parameter override support"""
        try:
            # Tokenize the prompt
            prompt_tokens = self.model.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate sequence
            with torch.inference_mode():
                tokens = self.model.model.generate(
                    prompt_tokens,
                    max_length=length,
                    min_length=max(1, length - 100),
                    do_sample=True,
                    top_k=diversity.get('top_k', 50),
                    top_p=diversity.get('top_p', 0.95),
                    temperature=diversity.get('temperature', 1.0),
                    num_return_sequences=1,
                    remove_invalid_values=True,
                    use_cache=True,
                    pad_token_id=self.model.tokenizer.encode("[PAD]")[0],
                )
            
            # Decode and clean the sequence
            decoded_seq = self.model.tokenizer.decode(tokens[0], skip_special_tokens=True)
            cleaned_seq = decoded_seq.replace(' ', '')  # Remove spaces
            
            return cleaned_seq[:length]
        except Exception as e:
            print(f"GenSLM generation failed: {e}")
            return self._generate_probabilistic(prompt, length)
    
    def _generate_probabilistic(self, prompt: str, length: int) -> str:
        """Probabilistic generation considering constraint conditions"""
        sequence = prompt
        nucleotides = ['A', 'T', 'G', 'C']
        
        # Get composition probabilities
        if self.constraints['composition']:
            probs = [self.constraints['composition'].get(nt, 0.25) for nt in nucleotides]
        else:
            probs = [0.25, 0.25, 0.25, 0.25]
        
        while len(sequence) < length:
            # Adjust probabilities based on current GC content
            if self.constraints['gc_content']:
                probs = self._adjust_probabilities_for_gc(sequence, probs)
            
            # Avoid forbidden motifs
            probs = self._adjust_probabilities_for_motifs(sequence, probs, nucleotides)
            
            # Sample next nucleotide
            next_nt = np.random.choice(nucleotides, p=probs)
            sequence += next_nt
        
        return sequence[:length]
    
    def _adjust_probabilities_for_gc(self, current_seq: str, base_probs: List[float]) -> List[float]:
        """Adjust nucleotide probabilities based on current GC content"""
        if not self.constraints['gc_content']:
            return base_probs
        
        current_gc = self._calculate_gc_content(current_seq)
        gc_constraint = self.constraints['gc_content']
        target_gc = (gc_constraint['min'] + gc_constraint['max']) / 2
        
        # Calculate GC adjustment amount
        gc_diff = target_gc - current_gc
        
        adjusted_probs = base_probs.copy()
        
        if gc_diff > 0:  # Need to increase GC
            adjusted_probs[2] *= (1 + gc_diff)  # G
            adjusted_probs[3] *= (1 + gc_diff)  # C
        else:  # Need to decrease GC
            adjusted_probs[0] *= (1 - gc_diff)  # A
            adjusted_probs[1] *= (1 - gc_diff)  # T
        
        # Normalize
        total = sum(adjusted_probs)
        return [p/total for p in adjusted_probs]
    
    def _adjust_probabilities_for_motifs(self, current_seq: str, probs: List[float], 
                                       nucleotides: List[str]) -> List[float]:
        """Adjust probabilities based on motif constraints"""
        if not self.constraints['motifs']['avoid']:
            return probs
        
        adjusted_probs = probs.copy()
        
        for motif in self.constraints['motifs']['avoid']:
            for i, nt in enumerate(nucleotides):
                test_seq = current_seq + nt
                # Check if adding nucleotide would form forbidden motif
                if len(test_seq) >= len(motif) and motif in test_seq[-len(motif):]:
                    adjusted_probs[i] *= 0.01  # Heavy penalty
        
        # Normalize
        total = sum(adjusted_probs)
        return [p/total for p in adjusted_probs] if total > 0 else probs
    
    def _apply_constraints(self, sequence: str) -> str:
        """Apply constraint condition optimization to sequence"""
        optimized = sequence
        
        # If specific motifs need to be included, try to insert them
        if self.constraints['motifs']['require']:
            optimized = self._ensure_required_motifs(optimized)
        
        # Length adjustment
        target_length = self.constraints['length']['target']
        if len(optimized) != target_length:
            if len(optimized) < target_length:
                # Extend sequence
                optimized = self._extend_sequence(optimized, target_length)
            else:
                # Truncate sequence
                optimized = optimized[:target_length]
        
        return optimized
    
    def _ensure_required_motifs(self, sequence: str) -> str:
        """Ensure sequence contains required motifs"""
        for motif in self.constraints['motifs']['require']:
            if motif not in sequence:
                # Insert motif into sequence
                insert_pos = len(sequence) // 2
                sequence = sequence[:insert_pos] + motif + sequence[insert_pos:]
        return sequence
    
    def _extend_sequence(self, sequence: str, target_length: int) -> str:
        """Extend sequence to target length"""
        needed = target_length - len(sequence)
        
        # Use probabilistic generation for extension
        nucleotides = ['A', 'T', 'G', 'C']
        if self.constraints['composition']:
            probs = [self.constraints['composition'].get(nt, 0.25) for nt in nucleotides]
        else:
            probs = [0.25, 0.25, 0.25, 0.25]
        
        extension = ''.join(np.random.choice(nucleotides, size=needed, p=probs))
        return sequence + extension
    
    def _validate_constraints(self, sequence: str) -> bool:
        """Validate whether sequence satisfies all constraints"""
        # GC content check
        if self.constraints['gc_content']:
            if not self._check_gc_constraint(sequence):
                return False
        
        # Motif check  
        if not self._check_motif_constraints(sequence):
            return False
        
        # Length check
        if not self._check_length_constraint(sequence):
            return False
        
        return True
    
    def _check_gc_constraint(self, sequence: str) -> bool:
        """Check GC content constraints"""
        gc_constraint = self.constraints['gc_content']
        gc_content = self._calculate_gc_content(sequence)
        
        if gc_constraint['local']:
            # Local GC check
            window_size = gc_constraint['window_size']
            for i in range(0, len(sequence) - window_size + 1, window_size // 2):
                window = sequence[i:i + window_size]
                window_gc = self._calculate_gc_content(window)
                if not (gc_constraint['min'] <= window_gc <= gc_constraint['max']):
                    return False
        
        # Global GC check
        return gc_constraint['min'] <= gc_content <= gc_constraint['max']
    
    def _check_motif_constraints(self, sequence: str) -> bool:
        """Check motif constraints"""
        # Check avoided motifs
        for motif in self.constraints['motifs']['avoid']:
            if motif in sequence or motif in str(Seq(sequence).reverse_complement()):
                return False
        
        # Check required motifs
        for motif in self.constraints['motifs']['require']:
            if motif not in sequence:
                return False
        
        return True
    
    def _check_length_constraint(self, sequence: str) -> bool:
        """Check length constraints"""
        length_constraint = self.constraints['length']
        seq_len = len(sequence)
        return length_constraint['min'] <= seq_len <= length_constraint['max']
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content"""
        if len(sequence) == 0:
            return 0.0
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)
    
    def _generate_fallback_sequence(self, prompt: str) -> str:
        """Generate fallback sequence"""
        target_length = self.constraints['length']['target']
        return prompt + 'A' * (target_length - len(prompt))
    
    def _create_sequence_record(self, sequence: str, prompt: str, index: int, 
                              is_fallback: bool = False) -> Dict:
        """Create sequence record"""
        return {
            'id': f"Enhanced_GenomeBridge_{index+1:04d}",
            'sequence': sequence,
            'length': len(sequence),
            'prompt': prompt,
            'gc_content': self._calculate_gc_content(sequence),
            'base_model': self.base_model,
            'satisfies_constraints': not is_fallback,
            'constraints_applied': len([c for c in self.constraints.values() if c is not None])
        }
    
    def save_sequences(self, sequences: List[Dict], output_file: str, 
                      format_type: str = "csv"):
        """Save sequences to file"""
        if format_type.lower() == "csv":
            df = pd.DataFrame(sequences)
            df.to_csv(output_file, index=False)
        elif format_type.lower() == "fasta":
            with open(output_file, 'w') as f:
                for seq_data in sequences:
                    f.write(f">{seq_data['id']} len={seq_data['length']} gc={seq_data['gc_content']:.3f} model={seq_data['base_model']}\n")
                    f.write(f"{seq_data['sequence']}\n")
        
        print(f"💾 Saved {len(sequences)} sequences to {output_file}")

def main():
    """Main function - Demonstrate enhanced control mechanisms"""
    parser = argparse.ArgumentParser(description='Enhanced GenomeBridge Controlled Generation')
    
    # Basic parameters
    parser.add_argument('--model', choices=['evo1', 'evo2', 'generator', 'genomeocean', 'genslm', 'probabilistic'],
                       default='probabilistic', help='Base generation model')
    parser.add_argument('--num_sequences', type=int, default=50, help='Number of sequences to generate')
    parser.add_argument('--target_length', type=int, default=1000, help='Target sequence length')
    
    # GC content control
    parser.add_argument('--gc_min', type=float, default=0.4, help='Minimum GC content')
    parser.add_argument('--gc_max', type=float, default=0.6, help='Maximum GC content')
    parser.add_argument('--gc_window', type=int, default=100, help='GC window size')
    
    # Motif control
    parser.add_argument('--avoid_motifs', nargs='*', default=[], 
                       help='Motifs to avoid (e.g. GAATTC GGATCC)')
    parser.add_argument('--require_motifs', nargs='*', default=[],
                       help='Motifs that must be present')
    
    # Diversity control
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=None, help='Nucleus sampling')
    
    # Output control
    parser.add_argument('--output_prefix', default='enhanced_sequences', help='Output file prefix')
    parser.add_argument('--output_format', choices=['csv', 'fasta'], default='csv', help='Output format')
    
    args = parser.parse_args()
    
    # Create enhanced generator
    generator = EnhancedGenomeBridgeGenerator(base_model=args.model)
    
    # Set constraints
    generator.set_gc_content_constraint(args.gc_min, args.gc_max, args.gc_window)
    
    if args.avoid_motifs or args.require_motifs:
        generator.set_motif_constraints(args.avoid_motifs, args.require_motifs)
    
    generator.set_diversity_parameters(args.temperature, args.top_k, args.top_p)
    generator.set_length_constraint(args.target_length)
    
    # Generate sequences
    sequences = generator.generate_controlled_sequences(num_sequences=args.num_sequences)
    
    # Save results
    output_file = f"{args.output_prefix}.{args.output_format}"
    generator.save_sequences(sequences, output_file, args.output_format)
    
    # Statistics report
    successful = sum(1 for seq in sequences if seq['satisfies_constraints'])
    avg_gc = np.mean([seq['gc_content'] for seq in sequences])
    avg_length = np.mean([seq['length'] for seq in sequences])
    
    print(f"\n📊 Generation Summary:")
    print(f"   Model used: {args.model}")
    print(f"   Constraint satisfaction: {successful}/{len(sequences)} ({successful/len(sequences)*100:.1f}%)")
    print(f"   Average GC content: {avg_gc:.1%}")
    print(f"   Average length: {avg_length:.0f} bp")
    print(f"   Output saved to: {output_file}")

if __name__ == '__main__':
    main() 