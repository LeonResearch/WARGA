import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim

from model import GCNModelAE, Regularizer
from optimizer import loss_function1
from utils import mask_test_edges, preprocess_graph, get_roc_score,load_data_with_labels


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in the first encoding layer.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in the second embedding layer.')
parser.add_argument('--hidden3', type=int, default=16, help='Number of units in the first hidden layer of Regularizer.')
parser.add_argument('--hidden4', type=int, default=64, help='Number of units in the second hidden layer of Regularizer.')
parser.add_argument('--clamp', type=float, default=0.01, help='Weight clamp for Regularizer Parameters.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for Generator.')
parser.add_argument('--dislr', type=float, default=0.001, help='Initial learning rate for Regularizer.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features, true_labels= load_data_with_labels(args.dataset_str)
    n_nodes, feat_dim = features.shape
    features = features.to(device)
    
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_norm = adj_norm.to(device)
    
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())
    adj_label = adj_label.to(device)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    
    # Models
    model = GCNModelAE(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    regularizer = Regularizer(args.hidden3, args.hidden2, args.hidden4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    regularizer_optimizer = optim.Adam(regularizer.parameters(), lr=args.lr)

    # Learning
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        regularizer.train()
        
        # Generate embeddings
        predicted_labels_prob, emb = model(features, adj_norm)
        
        # Wasserstein Regularizer
        for i in range(1):
            f_z = regularizer(emb).to(device)
            r = torch.normal(0.0, 1.0, [n_nodes, args.hidden2]).to(device)
            f_r = regularizer(r)          
            
            reg_loss = - f_r.mean() + f_z.mean() 
            
            regularizer_optimizer.zero_grad()
            reg_loss.backward(retain_graph=True)
            regularizer_optimizer.step()
            
            # weight clamp
            for p in regularizer.parameters():
                p.data.clamp_(-args.clamp, args.clamp)

        # Update Generator
        f_z = regularizer(emb)  
        generator_loss = -f_z.mean()        
        loss = loss_function1(preds=predicted_labels_prob, labels=adj_label,
                             norm=norm, pos_weight=torch.tensor(pos_weight))
        loss = loss + generator_loss
        optimizer.zero_grad()
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()
        
        # Show validation results
        hidden_emb = emb.cpu().data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t))  
    
    # Testing results
    print("Optimization Finished!")
    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':
    gae_for(args)
