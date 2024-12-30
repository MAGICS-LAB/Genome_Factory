import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
import os
from dataloader.dataset import GTDBDataset
from models import CNN_S, MLP_S, LSTM_S, Tran_S
from torch.utils.data import Dataset, DataLoader
import json
import argparse
import torch
import tqdm

def normalize_dm(df):

    df_min = df.min().min()
    df_max = df.max().max()
    normalized_df = (df - df_min) / (df_max - df_min)
    return normalized_df

def test_embedding(test_dataset, labels, backbone_type, model_type, model_path, data_type_train, data_type_test, batch_size, seq_length, feat_dim, output_dim, resolution, max_length, output_path):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    if model_type != 'None':
        if model_type == "CNN":
            model = CNN_S(embedding_dim=feat_dim, output_dim=output_dim, input_len=seq_length)
        elif model_type == "MLP":
            model = MLP_S(embedding_dim=feat_dim, output_dim=output_dim, input_len=seq_length)
        elif model_type == "LSTM":
            if seq_length == 1:
                raise ValueError("Sequence length cannot be 1 for LSTM. Please set a different sequence length.")
            model = LSTM_S(embedding_dim=feat_dim, output_dim=output_dim, input_len=seq_length)
        elif model_type == "Tran":
            if seq_length == 1:
                raise ValueError("Sequence length cannot be 1 for Transformer. Please set a different sequence length.")
            model = Tran_S(embedding_dim=feat_dim, output_dim=output_dim, input_len=seq_length)
      
        model.load_state_dict(torch.load(model_path+"/pytorch_model.bin"))
        model.eval()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(tqdm.tqdm(test_loader)):
                feature = model(inputs['embeddings'].float())
                if batch_idx == 0:
                    features = feature
                else:
                    features = torch.cat((features, feature), dim=0)
    else: 
        for batch_idx, inputs in enumerate(test_loader):
            feature = inputs['embeddings']
            if batch_idx == 0:
                features = feature
            else:
                features = torch.cat((features, feature), dim=0)
        
    features = np.array(features)
    differences = features[:, None, :] - features[None, :, :]
    distance_matrix_square = np.sum(differences**2, axis=2)
    distance_df = pd.DataFrame(distance_matrix_square, index=None, columns=None)
    
    emb_dm = normalize_dm(distance_df)
    dm_gt = normalize_dm(labels)
    
    mask = np.triu(np.ones(emb_dm.shape), k=0).astype(bool)
    
    emb_dm = emb_dm.values[mask]
    dm_gt = dm_gt.values[mask]
    
    formatted_path_segment = f"{data_type_train}_{data_type_test}_{max_length}_{resolution}"
    save_path = os.path.join(output_path, 'correlation', backbone_type, model_type, formatted_path_segment)
    
    r2 = r2_score(emb_dm.flatten(), dm_gt.flatten())
    print(f'R2 = {r2}')
    pearson_corr, _ = pearsonr(emb_dm.flatten(), dm_gt.flatten())
    print(f"Pearson r: {pearson_corr}")
    spearman_corr, _ = spearmanr(emb_dm.flatten(), dm_gt.flatten())
    print(f"Spearman r: {spearman_corr}")
    
    results = {'R2':r2, 'Pearson':pearson_corr, 'Spearman': spearman_corr}
    os.makedirs(save_path, exist_ok=True)
    with open(save_path+'/results.json', 'w') as file:
        json.dump(results, file, indent=4)
        
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_type", type=str, default=None, help="backbone type to extract the embedding")
    parser.add_argument("--model_type", type=str, default='None', help="model type")
    parser.add_argument("--model_path", type=str, default=None, help="model path")
    parser.add_argument("--data_path", type=str, default=None, help="path to the testing DNA embeddings")
    parser.add_argument("--label_path", type=str, default=None, help="path to the testing labels")
    parser.add_argument("--data_type_train", type=str, default='None', help="training data type")
    parser.add_argument("--data_type_test", type=str, default='None', help="testing data type")
    parser.add_argument("--batch_size", type=int, default=4)  
    parser.add_argument("--max_length", type=int, default=10240)  
    parser.add_argument("--resolution", type=int, default=10240) 
    parser.add_argument("--feat_dim", type=int, default=3072)
    parser.add_argument("--output_dim", type=int, default=768)
    parser.add_argument("--output_path", type=str, default=None, help="path to save the results")
    args = parser.parse_args()
    
    test_dataset = GTDBDataset(os.path.join(args.data_path, 'test')) 
    labels = pd.read_csv(os.path.join(args.label_path, 'test/label.csv'), header=None, index_col=None)
    seq_length = int(args.max_length/args.resolution)
    
    test_embedding(test_dataset=test_dataset, 
                  labels=labels,
                  backbone_type=args.backbone_type,
                  model_type=args.model_type,
                  model_path=args.model_path, 
                  data_type_train=args.data_type_train,
                  data_type_test=args.data_type_test,
                  batch_size=args.batch_size, 
                  seq_length=seq_length,
                  feat_dim=args.feat_dim,
                  output_dim=args.output_dim,
                  resolution=args.resolution,
                  max_length=args.max_length,
                  output_path=args.output_path)

if __name__ == "__main__":
    main()