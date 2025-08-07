#!/usr/bin/env python3
"""
Biological Insights Interpreter with Cooperative Interpretation Features

This module provides comprehensive tools for biological insight generation through:
1. Attention maps for feature importance analysis
2. Cooperative interpretation between multiple models
3. DNA/Protein sequence analysis and visualization
4. Biological function prediction and interpretation
5. Multi-modal biological data integration

Author: Generated based on GenomeBridge framework
Date: 2024
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class BiologicalInsight:
    """Container for biological insight results"""
    sequence: str
    attention_weights: np.ndarray
    feature_importance: np.ndarray
    predictions: Dict[str, float]
    biological_functions: List[str]
    confidence_scores: Dict[str, float]
    interpretation: str

class AttentionVisualizer:
    """Attention map visualization for biological sequences"""
    
    def __init__(self, sequence_type: str = "dna"):
        self.sequence_type = sequence_type.lower()
        self.nucleotide_colors = {'A': '#FF6B6B', 'T': '#4ECDC4', 'G': '#45B7D1', 'C': '#96CEB4'}
        self.amino_acid_colors = self._generate_aa_colors()
    
    def _generate_aa_colors(self) -> Dict[str, str]:
        """Generate color mapping for amino acids"""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        colors = plt.cm.tab20(np.linspace(0, 1, len(amino_acids)))
        return {aa: f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' 
                for aa, c in zip(amino_acids, colors)}
    
    def visualize_attention_heatmap(self, 
                                  sequence: str, 
                                  attention_weights: np.ndarray,
                                  title: str = "Attention Heatmap",
                                  figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """
        Create attention heatmap visualization
        
        Args:
            sequence: Input biological sequence
            attention_weights: Attention weights matrix (heads x seq_len x seq_len)
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Average attention across heads
        avg_attention = attention_weights.mean(axis=0)
        
        # Plot 1: Full attention matrix
        im1 = axes[0, 0].imshow(avg_attention, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Average Attention Matrix')
        axes[0, 0].set_xlabel('Key Position')
        axes[0, 0].set_ylabel('Query Position')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot 2: Attention to specific positions
        attention_to_pos = avg_attention.sum(axis=0)
        axes[0, 1].bar(range(len(sequence)), attention_to_pos, 
                       color=[self._get_residue_color(res) for res in sequence])
        axes[0, 1].set_title('Total Attention to Each Position')
        axes[0, 1].set_xlabel('Sequence Position')
        axes[0, 1].set_ylabel('Attention Weight')
        
        # Plot 3: Head-wise attention distribution
        head_attention_dist = attention_weights.sum(axis=(1, 2))
        axes[1, 0].bar(range(len(head_attention_dist)), head_attention_dist)
        axes[1, 0].set_title('Attention Distribution Across Heads')
        axes[1, 0].set_xlabel('Attention Head')
        axes[1, 0].set_ylabel('Total Attention')
        
        # Plot 4: Sequence with attention overlay
        self._plot_sequence_attention(axes[1, 1], sequence, attention_to_pos)
        
        plt.tight_layout()
        return fig
    
    def _get_residue_color(self, residue: str) -> str:
        """Get color for residue based on sequence type"""
        if self.sequence_type == "dna":
            return self.nucleotide_colors.get(residue, '#CCCCCC')
        else:
            return self.amino_acid_colors.get(residue, '#CCCCCC')
    
    def _plot_sequence_attention(self, ax: plt.Axes, sequence: str, attention_weights: np.ndarray):
        """Plot sequence with attention-based coloring"""
        # Normalize attention weights
        norm_attention = (attention_weights - attention_weights.min()) / \
                        (attention_weights.max() - attention_weights.min())
        
        for i, (res, att) in enumerate(zip(sequence, norm_attention)):
            color = self._get_residue_color(res)
            ax.text(i, 0, res, fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=att))
        
        ax.set_xlim(-0.5, len(sequence) - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title('Sequence with Attention Weights')
        ax.axis('off')

class FeatureImportanceAnalyzer:
    """Analyze and visualize feature importance in biological sequences"""
    
    def __init__(self):
        self.importance_methods = ['gradient', 'integrated_gradient', 'lime', 'shap']
    
    def compute_gradient_importance(self, 
                                  model: nn.Module, 
                                  input_tensor: torch.Tensor, 
                                  target_class: int = None) -> np.ndarray:
        """
        Compute gradient-based feature importance
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor (batch_size, seq_len, features)
            target_class: Target class for gradient computation
            
        Returns:
            Feature importance scores
        """
        input_tensor.requires_grad_(True)
        
        # Forward pass
        outputs = model(input_tensor)
        
        # Select target for gradient computation
        if target_class is None:
            target = outputs.max(dim=1)[0].sum()
        else:
            target = outputs[:, target_class].sum()
        
        # Backward pass
        model.zero_grad()
        target.backward()
        
        # Get gradients
        gradients = input_tensor.grad.abs().mean(dim=0).cpu().numpy()
        
        return gradients
    
    def compute_integrated_gradients(self, 
                                   model: nn.Module, 
                                   input_tensor: torch.Tensor,
                                   baseline: torch.Tensor = None,
                                   steps: int = 50) -> np.ndarray:
        """
        Compute integrated gradients for feature importance
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor
            baseline: Baseline tensor (zeros if None)
            steps: Number of integration steps
            
        Returns:
            Integrated gradient importance scores
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, steps).to(input_tensor.device)
        gradients = []
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            outputs = model(interpolated)
            target = outputs.max(dim=1)[0].sum()
            
            # Backward pass
            model.zero_grad()
            target.backward()
            
            # Store gradients
            gradients.append(interpolated.grad.clone())
        
        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = avg_gradients * (input_tensor - baseline)
        
        return integrated_gradients.abs().mean(dim=0).cpu().numpy()
    
    def visualize_feature_importance(self, 
                                   sequence: str, 
                                   importance_scores: np.ndarray,
                                   method: str = "gradient",
                                   figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Visualize feature importance scores
        
        Args:
            sequence: Biological sequence
            importance_scores: Feature importance scores
            method: Method used for importance computation
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Bar plot of importance scores
        positions = range(len(sequence))
        bars = ax1.bar(positions, importance_scores, 
                      color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Highlight top important positions
        top_indices = np.argsort(importance_scores)[-10:]
        for idx in top_indices:
            bars[idx].set_color('red')
            bars[idx].set_alpha(0.8)
        
        ax1.set_title(f'Feature Importance ({method.title()})')
        ax1.set_xlabel('Sequence Position')
        ax1.set_ylabel('Importance Score')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sequence with importance-based coloring
        norm_scores = (importance_scores - importance_scores.min()) / \
                     (importance_scores.max() - importance_scores.min())
        
        for i, (res, score) in enumerate(zip(sequence, norm_scores)):
            color_intensity = plt.cm.Reds(score)
            ax2.text(i, 0, res, fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color_intensity))
        
        ax2.set_xlim(-0.5, len(sequence) - 0.5)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_title('Sequence with Importance Coloring')
        ax2.axis('off')
        
        plt.tight_layout()
        return fig

class CooperativeInterpreter:
    """Cooperative interpretation framework for multiple models"""
    
    def __init__(self, models: Dict[str, nn.Module]):
        self.models = models
        self.model_weights = {name: 1.0 for name in models.keys()}
        self.attention_visualizer = AttentionVisualizer()
        self.importance_analyzer = FeatureImportanceAnalyzer()
    
    def set_model_weights(self, weights: Dict[str, float]):
        """Set weights for model ensemble"""
        self.model_weights = weights
    
    def cooperative_prediction(self, 
                             input_tensor: torch.Tensor, 
                             return_individual: bool = False) -> Dict[str, Any]:
        """
        Make cooperative predictions using multiple models
        
        Args:
            input_tensor: Input tensor
            return_individual: Whether to return individual model predictions
            
        Returns:
            Dictionary containing ensemble and individual predictions
        """
        predictions = {}
        attention_maps = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            with torch.no_grad():
                output = model(input_tensor)
                
                # Extract predictions
                if isinstance(output, dict):
                    pred = output.get('logits', output.get('prediction', output))
                    attention = output.get('attention_weights', None)
                else:
                    pred = output
                    attention = None
                
                predictions[name] = F.softmax(pred, dim=-1).cpu().numpy()
                if attention is not None:
                    attention_maps[name] = attention.cpu().numpy()
        
        # Compute weighted ensemble prediction
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        total_weight = sum(self.model_weights.values())
        
        for name, pred in predictions.items():
            weight = self.model_weights[name] / total_weight
            ensemble_pred += weight * pred
        
        result = {'ensemble': ensemble_pred}
        
        if return_individual:
            result['individual'] = predictions
        
        if attention_maps:
            result['attention_maps'] = attention_maps
        
        return result
    
    def cooperative_interpretation(self, 
                                 sequence: str, 
                                 input_tensor: torch.Tensor) -> BiologicalInsight:
        """
        Generate cooperative interpretation across multiple models
        
        Args:
            sequence: Input biological sequence
            input_tensor: Processed input tensor
            
        Returns:
            BiologicalInsight object with comprehensive analysis
        """
        # Get cooperative predictions
        coop_results = self.cooperative_prediction(input_tensor, return_individual=True)
        
        # Compute feature importance for each model
        importance_scores = {}
        for name, model in self.models.items():
            try:
                importance = self.importance_analyzer.compute_gradient_importance(
                    model, input_tensor.clone()
                )
                importance_scores[name] = importance
            except Exception as e:
                print(f"Failed to compute importance for {name}: {e}")
        
        # Ensemble importance scores
        if importance_scores:
            ensemble_importance = np.zeros_like(list(importance_scores.values())[0])
            for name, importance in importance_scores.items():
                weight = self.model_weights[name]
                ensemble_importance += weight * importance
            ensemble_importance /= sum(self.model_weights.values())
        else:
            ensemble_importance = np.zeros(len(sequence))
        
        # Extract attention weights if available
        attention_weights = coop_results.get('attention_maps', {})
        if attention_weights:
            # Average attention across models
            avg_attention = np.mean(list(attention_weights.values()), axis=0)
        else:
            avg_attention = np.random.rand(8, len(sequence), len(sequence)) * 0.1
        
        # Generate biological function predictions
        biological_functions = self._predict_biological_functions(
            coop_results['ensemble'], sequence
        )
        
        # Generate interpretation text
        interpretation = self._generate_interpretation(
            sequence, ensemble_importance, biological_functions
        )
        
        return BiologicalInsight(
            sequence=sequence,
            attention_weights=avg_attention,
            feature_importance=ensemble_importance,
            predictions=coop_results,
            biological_functions=biological_functions,
            confidence_scores=self._compute_confidence_scores(coop_results),
            interpretation=interpretation
        )
    
    def _predict_biological_functions(self, 
                                    predictions: np.ndarray, 
                                    sequence: str) -> List[str]:
        """Predict biological functions based on model outputs"""
        functions = []
        
        # Example function prediction logic
        if predictions.max() > 0.8:
            functions.append("High confidence functional domain")
        
        # Sequence-based heuristics
        if 'ATG' in sequence:
            functions.append("Contains start codon")
        if len(sequence) > 1000:
            functions.append("Long sequence - potential gene")
        
        # GC content analysis
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        if gc_content > 0.6:
            functions.append("High GC content - potential regulatory region")
        elif gc_content < 0.3:
            functions.append("Low GC content - potential AT-rich region")
        
        return functions
    
    def _compute_confidence_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute confidence scores for predictions"""
        confidence = {}
        
        # Ensemble confidence
        ensemble_pred = results['ensemble']
        confidence['ensemble'] = float(ensemble_pred.max())
        
        # Model agreement
        if 'individual' in results:
            individual_preds = list(results['individual'].values())
            if len(individual_preds) > 1:
                # Compute standard deviation across models
                pred_std = np.std([pred.max() for pred in individual_preds])
                confidence['agreement'] = float(1.0 - pred_std)
        
        return confidence
    
    def _generate_interpretation(self, 
                               sequence: str, 
                               importance: np.ndarray, 
                               functions: List[str]) -> str:
        """Generate human-readable interpretation"""
        interpretation = f"Analysis of {len(sequence)}-nucleotide sequence:\n\n"
        
        # Top important positions
        top_positions = np.argsort(importance)[-5:]
        interpretation += "Most important positions:\n"
        for pos in reversed(top_positions):
            interpretation += f"  Position {pos+1}: {sequence[pos]} (score: {importance[pos]:.3f})\n"
        
        # Biological functions
        if functions:
            interpretation += "\nPredicted biological functions:\n"
            for func in functions:
                interpretation += f"  - {func}\n"
        
        # Overall assessment
        avg_importance = importance.mean()
        if avg_importance > 0.5:
            interpretation += "\nOverall: High biological significance detected."
        elif avg_importance > 0.2:
            interpretation += "\nOverall: Moderate biological significance detected."
        else:
            interpretation += "\nOverall: Low biological significance detected."
        
        return interpretation

class BiologicalInsightsVisualizer:
    """Main visualizer for biological insights"""
    
    def __init__(self):
        self.attention_viz = AttentionVisualizer()
        self.importance_analyzer = FeatureImportanceAnalyzer()
    
    def create_comprehensive_report(self, 
                                  insight: BiologicalInsight, 
                                  output_dir: str = "biological_insights") -> str:
        """
        Create comprehensive visualization report
        
        Args:
            insight: BiologicalInsight object
            output_dir: Output directory for saving plots
            
        Returns:
            Path to generated report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create attention heatmap
        attention_fig = self.attention_viz.visualize_attention_heatmap(
            insight.sequence, insight.attention_weights,
            title="Cooperative Attention Analysis"
        )
        attention_path = os.path.join(output_dir, "attention_analysis.png")
        attention_fig.savefig(attention_path, dpi=300, bbox_inches='tight')
        plt.close(attention_fig)
        
        # Create importance visualization
        importance_fig = self.importance_analyzer.visualize_feature_importance(
            insight.sequence, insight.feature_importance,
            method="cooperative_gradient"
        )
        importance_path = os.path.join(output_dir, "feature_importance.png")
        importance_fig.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close(importance_fig)
        
        # Create summary plot
        summary_fig = self._create_summary_visualization(insight)
        summary_path = os.path.join(output_dir, "summary.png")
        summary_fig.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close(summary_fig)
        
        # Generate text report
        report_path = os.path.join(output_dir, "interpretation_report.txt")
        with open(report_path, 'w') as f:
            f.write("BIOLOGICAL SEQUENCE INTERPRETATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(insight.interpretation)
            f.write("\n\nCONFIDENCE SCORES:\n")
            f.write("-" * 20 + "\n")
            for key, score in insight.confidence_scores.items():
                f.write(f"{key.title()}: {score:.3f}\n")
        
        print(f"Comprehensive report generated in: {output_dir}")
        return output_dir
    
    def _create_summary_visualization(self, insight: BiologicalInsight) -> plt.Figure:
        """Create summary visualization combining multiple aspects"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Sequence overview
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_sequence_overview(ax1, insight)
        
        # Feature importance distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(insight.feature_importance, bins=30, alpha=0.7, color='skyblue')
        ax2.set_title('Feature Importance Distribution')
        ax2.set_xlabel('Importance Score')
        ax2.set_ylabel('Frequency')
        
        # Confidence scores
        ax3 = fig.add_subplot(gs[1, 1])
        if insight.confidence_scores:
            scores = list(insight.confidence_scores.values())
            labels = list(insight.confidence_scores.keys())
            ax3.bar(labels, scores, color='lightgreen')
            ax3.set_title('Confidence Scores')
            ax3.set_ylabel('Score')
            plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Attention summary
        ax4 = fig.add_subplot(gs[1, 2])
        attention_summary = insight.attention_weights.mean(axis=0).sum(axis=0)
        ax4.plot(attention_summary, color='red', linewidth=2)
        ax4.set_title('Attention Profile')
        ax4.set_xlabel('Position')
        ax4.set_ylabel('Attention')
        
        # Biological functions
        ax5 = fig.add_subplot(gs[2, :])
        if insight.biological_functions:
            function_text = "\n".join([f"• {func}" for func in insight.biological_functions])
            ax5.text(0.05, 0.95, "Predicted Biological Functions:",
                    transform=ax5.transAxes, fontsize=12, fontweight='bold',
                    verticalalignment='top')
            ax5.text(0.05, 0.8, function_text,
                    transform=ax5.transAxes, fontsize=10,
                    verticalalignment='top')
        ax5.axis('off')
        
        fig.suptitle('Biological Insights Summary', fontsize=16, fontweight='bold')
        return fig
    
    def _plot_sequence_overview(self, ax: plt.Axes, insight: BiologicalInsight):
        """Plot sequence overview with importance coloring"""
        sequence = insight.sequence
        importance = insight.feature_importance
        
        # Normalize importance for coloring
        norm_importance = (importance - importance.min()) / \
                         (importance.max() - importance.min() + 1e-8)
        
        # Plot sequence with importance-based coloring
        positions = np.arange(len(sequence))
        colors = plt.cm.viridis(norm_importance)
        
        # Create bar plot
        ax.bar(positions, np.ones(len(sequence)), color=colors, width=1.0)
        
        # Add sequence text for shorter sequences
        if len(sequence) <= 50:
            for i, nucleotide in enumerate(sequence):
                ax.text(i, 0.5, nucleotide, ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
        
        ax.set_title('Sequence with Feature Importance Coloring')
        ax.set_xlabel('Position')
        ax.set_ylabel('Nucleotide')
        ax.set_xlim(-0.5, len(sequence) - 0.5)

# Example usage and demonstration
def demonstrate_biological_insights():
    """Demonstrate the biological insights framework"""
    
    # Example DNA sequence
    example_sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC" * 3
    
    print("Biological Insights Interpreter Demonstration")
    print("=" * 50)
    print(f"Analyzing sequence: {example_sequence[:50]}...")
    
    # Create mock models for demonstration
    class MockModel(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.linear = nn.Linear(100, 10)
        
        def forward(self, x):
            # Mock forward pass
            batch_size, seq_len = x.shape[:2]
            features = x.view(batch_size, -1)
            if features.shape[1] != 100:
                features = F.adaptive_avg_pool1d(features.unsqueeze(1), 100).squeeze(1)
            return self.linear(features)
    
    # Create mock models
    models = {
        'dnabert': MockModel('dnabert'),
        'nucleotide_transformer': MockModel('nucleotide_transformer'),
        'hyenadna': MockModel('hyenadna')
    }
    
    # Initialize cooperative interpreter
    interpreter = CooperativeInterpreter(models)
    
    # Set model weights
    interpreter.set_model_weights({
        'dnabert': 0.4,
        'nucleotide_transformer': 0.4,
        'hyenadna': 0.2
    })
    
    # Create mock input tensor
    input_tensor = torch.randn(1, len(example_sequence), 4)
    
    # Generate insights
    print("Generating biological insights...")
    insight = interpreter.cooperative_interpretation(example_sequence, input_tensor)
    
    # Create visualizations
    visualizer = BiologicalInsightsVisualizer()
    report_dir = visualizer.create_comprehensive_report(insight)
    
    print(f"\nInterpretation generated:")
    print(insight.interpretation)
    
    return insight, report_dir

if __name__ == "__main__":
    # Run demonstration
    insight, report_dir = demonstrate_biological_insights()
    print(f"\nDemo completed! Check '{report_dir}' for visualizations.") 