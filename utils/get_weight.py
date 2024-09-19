
import torch
from scipy.optimize import minimize
from utils import globalvar as gl
from torchmin import minimize

def get_weight_Chi2(Xs,ys,Xt,yt_hat):
    DEVICE = gl.get_value('DEVICE')
    ms, mt = len(Xs), len(Xt)
    X = torch.cat((Xs, Xt), dim=0)
    y = torch.cat((ys, yt_hat), dim=0)

    epsilon = 0.01

    # compute the Gaussian kernel width
    pairwise_dist = torch.cdist(X, X, p=2)**2 
    sigma = torch.median(pairwise_dist[pairwise_dist!=0]) 

    X_norm = torch.sum(X ** 2, axis = -1)
    K = torch.exp(-(X_norm[:,None] + X_norm[None,:] - 2 * torch.matmul(X, X.t())) / sigma) * torch.as_tensor(y[:,None]==y, dtype=torch.float64, device=DEVICE) # kernel matrix  
    Ks, Kt = K[:ms], K[ms:]
    H = 1.0 / mt * torch.matmul(Kt.t(), Kt) 
    invM = torch.inverse(H + epsilon * torch.eye(ms+mt, device=DEVICE))
    A = 2 * (1.0 / ms)**2 * torch.matmul(torch.matmul(Ks, invM), Ks.t())
    B = (1.0 / ms)**2 * torch.matmul(torch.matmul(torch.matmul(torch.matmul(Ks, invM), H), invM), Ks.t())
    C = A - B

    def obj(w):
        w = torch.softmax(w, dim=0) * ms
        div = torch.matmul(torch.matmul(w, C), w)  - 1.0 
        return div

    w0 = torch.zeros(ms, dtype=torch.double, device=DEVICE)
    result = minimize(obj, w0, method='l-bfgs') #
    w = torch.softmax(result.x, dim=0) * ms
    return w

def get_weight_L2(Xs,ys,Xt,yt_hat):
    DEVICE = gl.get_value('DEVICE')
    Xs = Xs / torch.linalg.norm(Xs, axis=1, keepdims=True)
    Xt = Xt / torch.linalg.norm(Xt, axis=1, keepdims=True)
    ms, mt = len(Xs), len(Xt)
    X = torch.cat((Xs, Xt), dim=0)
    y = torch.cat((ys, yt_hat), dim=0)

    epsilon = 0.01
    sigma2 = 1.0 / torch.pi

    X_norm = torch.sum(X ** 2, axis = -1)
    Xt_norm = torch.sum(Xt ** 2, axis = -1)
    K = torch.exp(-(X_norm[:,None] + Xt_norm[None,:] - 2 * torch.matmul(X, Xt.t())) / (2 * sigma2)) * torch.as_tensor(y[:,None]==yt_hat, dtype=torch.float32, device=DEVICE) # kernel matrix  

    Ks, Kt = K[:ms], K[ms:]
    Kt_mean = torch.mean(Kt, dim=0)
    H = torch.exp(-(Xt_norm[:,None] + Xt_norm[None,:] - 2 * torch.matmul(Xt, Xt.t())) / (4 * sigma2)) * torch.as_tensor(yt_hat[:,None]==yt_hat, dtype=torch.float32, device=DEVICE)
    invM = torch.inverse(H + epsilon * torch.eye(mt, device=DEVICE))

    A = (1.0 / ms)**2 * torch.matmul(torch.matmul(Ks, invM), Ks.t()) - 0.5 * (1.0 / ms)**2 * torch.matmul(torch.matmul(torch.matmul(torch.matmul(Ks, invM), H), invM), Ks.t())
    B1 = 2 * (1.0 / ms) * torch.matmul(torch.matmul(Kt_mean, invM), Ks.t()) 
    B2 = (1.0 / ms) * torch.matmul(torch.matmul(torch.matmul(torch.matmul(Kt_mean, invM), H), invM), Ks.t())
    B = B1 - B2
    C = torch.matmul(torch.matmul(Kt_mean, invM), Kt_mean) - 0.5 * torch.matmul(torch.matmul(torch.matmul(torch.matmul(Kt_mean, invM), H), invM), Kt_mean)

    def obj(w):
        w = torch.exp(w) / torch.mean(torch.exp(w))
        div = torch.matmul(torch.matmul(w, A), w)  - torch.matmul(B, w) + C
        return div

    w0 = torch.zeros(ms, dtype=torch.float32, device=DEVICE)
    result = minimize(obj, w0, method='l-bfgs') #
    w = torch.exp(result.x) / torch.mean(torch.exp(result.x))
    return w