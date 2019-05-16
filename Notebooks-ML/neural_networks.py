import numpy as np
import linear_regression_models as lin

def train_perceptron(x, y, num_epochs=100, alpha=1, w_initial=None, build_regressors=True):
    
    if build_regressors:
        x_matrix = lin.build_poly_regressors(x)    
    else:
        x_matrix = x        
        
    y[y == 0] = -1    
        
    if len(y.shape) == 1:   
        y = y[:,None]
    
    if w_initial is None:
        w = np.zeros((x_matrix.shape[1], y.shape[1]))
    else:
        w = w_initial.copy()
    
    loss_history = []
    for epoch in range(num_epochs):
        random_permutation = np.random.permutation(y.shape[0])
        for xi, yi in zip(x_matrix[random_permutation], y[random_permutation]):
            error = yi - np.sign(xi @ w)      
            w += alpha * xi[:,None] @ error[:,None].T                
        loss_history.append(np.sum(np.maximum(0, -(y * np.sign(x_matrix @ w)))))
        
    return {'w': w, 'loss_history': loss_history}

def train_adaline(x, y, num_epochs=100, alpha=0.001, w_initial=None, build_regressors=True):
    
    if build_regressors:
        x_matrix = lin.build_poly_regressors(x)    
    else:
        x_matrix = x   
        
    if len(y.shape) == 1:   
        y = y[:,None]
    
    if w_initial is None:
        w = np.zeros((x_matrix.shape[1], y.shape[1]))
    else:
        w = w_initial.copy()
    
    loss_history = []
    for epoch in range(num_epochs):
        random_permutation = np.random.permutation(y.shape[0])
        for xi, yi in zip(x_matrix[random_permutation], y[random_permutation]):
            error = yi - xi @ w      
            w += alpha * xi[:,None] @ error[:,None].T   
        loss_history.append(np.mean((y - x_matrix @ w)**2))
        
    return {'w': w, 'loss_history': loss_history}

def linear_class_predic(model, x, build_regressors=True):
        
    if build_regressors:
        x_matrix = lin.build_poly_regressors(x)    
    else:
        x_matrix = x
    
    if model['w'].shape[1] > 1:    
        return np.argmax(x_matrix @ model['w'], axis=1)
    else:
        return np.maximum(0, np.sign(x_matrix @ model['w']))
    