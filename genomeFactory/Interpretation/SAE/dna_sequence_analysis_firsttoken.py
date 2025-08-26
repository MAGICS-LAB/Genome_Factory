import os
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import gc
import argparse

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import custom modules
from genomeFactory.Interpretation.SAE.model_wrapper import Model
from genomeFactory.Interpretation.SAE.dna_sae_module import DNASAELightningModule

def load_dna_data(csv_path):
    """Load DNA sequence data"""
    df = pd.read_csv(csv_path)
    sequences = df['DNA_sequence'].tolist()
    sequence_lengths = df['sequence_length'].values
    return sequences, sequence_lengths

def process_sequences_streaming(sequences, model, sae_model, k=64, batch_size=16):
    """
    Stream processing sequences: DNA sequences -> DNABERT -> SAE -> first token latent
    Avoid storing large intermediate tensors, directly output first token latent results
    """
    pooled_results = []
    total_sequences = len(sequences)
    
    for i in range(0, total_sequences, batch_size):
        batch_sequences = sequences[i:i+batch_size]
        
        try:
            with torch.no_grad():
                # 1. Get DNABERT embeddings
                tokens, hidden_states, attention_mask = model.get_layer_activations(batch_sequences)
                
                # 2. Process each sequence's SAE latents and take the first token
                for seq_idx in range(len(batch_sequences)):
                    seq_embedding = hidden_states[seq_idx:seq_idx+1].to(device)
                    seq_attention = attention_mask[seq_idx:seq_idx+1].to(device)
                    
                    # SAE encoding
                    x, mu, std = sae_model.sae_model.LN(seq_embedding)
                    x = x - sae_model.sae_model.b_pre
                    pre_acts = x @ sae_model.sae_model.w_enc + sae_model.sae_model.b_enc
                    latents = sae_model.sae_model.topK_activation(pre_acts, k)
                    
                    # Use first token's latent
                    first_token_latent = latents[:, 0, :]  # Take the first token
                    
                    pooled_results.append(first_token_latent.cpu())
                
                # Clean up GPU memory
                del hidden_states, attention_mask, tokens
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            continue
    
    if pooled_results:
        return torch.cat(pooled_results, dim=0)
    else:
        raise RuntimeError("No results generated")

def perform_ridge_regression(X, y, alpha=1.0, test_size=0.2, random_state=42):
    """Perform Ridge regression analysis"""
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    ridge_model = Ridge(alpha=alpha, random_state=random_state)
    ridge_model.fit(X_train, y_train)
    
    y_pred_train = ridge_model.predict(X_train)
    y_pred_test = ridge_model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    return ridge_model, scaler_X, {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse
    }

def save_feature_weights(ridge_model, output_path, feature_names=None):
    """Save feature weights to CSV file"""
    weights = ridge_model.coef_
    
    if feature_names is None:
        feature_names = [f'latent_{i}' for i in range(len(weights))]
    
    weights_df = pd.DataFrame({
        'feature_name': feature_names,
        'weight': weights,
        'abs_weight': np.abs(weights)
    })
    
    weights_df = weights_df.sort_values('abs_weight', ascending=False)
    weights_df['rank'] = range(1, len(weights_df) + 1)
    weights_df.to_csv(output_path, index=False)
    
    return weights_df

def main():
    """Main function - use streaming processing to avoid memory issues"""
    parser = argparse.ArgumentParser(description='DNA sequence analysis with Ridge regression using first token')
    parser.add_argument('--csv_path', type=str, default='feature_seqlength.csv',
                        help='Path to the CSV file containing DNA sequences')
    parser.add_argument('--sae_checkpoint_path', type=str, 
                        default='/projects/p32572/interprot/interprot/dna_results_dim4096_k64_auxk256_/checkpoints/dnabert2_768_sae4096_k64_auxk256_-step=54000-avg_mse_loss=0.04.ckpt',
                        help='Path to the SAE checkpoint file')
    parser.add_argument('--output_path', type=str, default='ridge_regression_weights_first_token.csv',
                        help='Output path for the ridge regression weights CSV file')
    
    args = parser.parse_args()
    
    # Configure paths from arguments
    csv_path = args.csv_path
    sae_checkpoint_path = args.sae_checkpoint_path
    output_path = args.output_path
    
    # Parameter settings
    TOP_K = 64
    RIDGE_ALPHA = 1.0
    BATCH_SIZE = 16  # Small batch processing to avoid memory issues
    
    try:
        # 1. Load data
        print("Loading DNA sequence data...")
        sequences, sequence_lengths = load_dna_data(csv_path)
        print(f"Loaded {len(sequences)} sequences")
        
        # 2. Load models
        print("Loading models...")
        model = Model()
        model.to(device)
        model.eval()
        
        sae_model = DNASAELightningModule.load_from_checkpoint(sae_checkpoint_path)
        sae_model.to(device)
        sae_model.eval()
        
        # 3. Stream process all sequences
        print("Processing sequences with streaming approach...")
        pooled_latents = process_sequences_streaming(
            sequences, model, sae_model, k=TOP_K, batch_size=BATCH_SIZE
        )
        
        # Clean up models
        del model, sae_model
        torch.cuda.empty_cache()
        gc.collect()
        
        # 4. Ridge regression
        print("Performing Ridge regression...")
        ridge_model, scaler, metrics = perform_ridge_regression(
            pooled_latents.numpy(), sequence_lengths, alpha=RIDGE_ALPHA
        )
        
        # 5. Save results
        print("Saving results...")
        weights_df = save_feature_weights(ridge_model, output_path)
        
        # 6. Output results
        print(f"Analysis completed successfully!")
        print(f"Sequences processed: {len(sequences)}")
        print(f"Ridge regression R²: {metrics['test_r2']:.4f}")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 