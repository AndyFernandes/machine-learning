import numpy as np
import linear_regression_models as lin

def create_GDA(x, y, class_prior=None):
    
    classes = np.unique(y)
    K = classes.shape[0]
    D = x.shape[1]
    
    if class_prior is None or class_prior.lower() == 'proportional':
        class_prior = np.array([ np.sum(y == k) for k in classes])/y.shape[0]
    else:
        class_prior = (1/K) * np.ones(K)
    
    mu = np.zeros((D, K))
    var = np.zeros((D, D, K))
    for k in classes:
        x_k = x[y == k]
        mu[:,k] = np.mean(x_k, axis=0)
        var[:,:,k] = (x_k - mu[:,k]).T @ (x_k - mu[:,k]) / x_k.shape[0]
        var[:,:,k] += np.eye(D) * np.mean(np.diag(var[:,:,k])) * 10 ** -6
    
    model = {'classes': classes, 'K': K, 'D': D, 'class_prior': class_prior,
             'mu': mu, 'var': var}
    
    return model

def create_GNB(x, y, class_prior=None):
    
    classes = np.unique(y)
    K = classes.shape[0]
    D = x.shape[1]
    
    if class_prior is None or class_prior.lower() == 'proportional':
        class_prior = np.array([ np.sum(y == k) for k in classes])/y.shape[0]
    else:
        class_prior = (1/K) * np.ones(K)
    
    mu = np.zeros((D, K))
    var = np.zeros((D, K))
    for k in classes:
        x_k = x[y == k]
        mu[:,k] = np.mean(x_k, axis=0)
        var[:,k] = np.sum((x_k - mu[:,k])**2, axis=0) / x_k.shape[0]
        var[:,k] += np.mean(var[:,k]) * 10 ** -6
    
    model = {'classes': classes, 'K': K, 'D': D, 'class_prior': class_prior,
             'mu': mu, 'var': var}
    
    return model

def predict_GDA(model, x):
    
    pred = np.zeros((x.shape[0], model['K']))
    
    for k in range(model['K']):
        inv_cov_matrix = np.linalg.inv(model['var'][:,:,k])
        
        pred[:,k] = (- 0.5 * np.log(np.linalg.det(model['var'][:,:,k])) \
            - 0.5 * (np.sum((x @ inv_cov_matrix) * x, axis=1)[:,None] \
            + model['mu'][:,k] @ inv_cov_matrix @ model['mu'][:,k] - 2 * (x @ inv_cov_matrix @ model['mu'][:,[k]])) \
            + model['class_prior'][k])[:,0]

    return model['classes'][np.argmax(pred, axis=1)]

def predict_GNB(model, x):
    
    pred = np.zeros((x.shape[0], model['K']))
    
    for k in range(model['K']):
        pred[:,k] = - 0.5 * np.sum(np.log(2*np.pi*model['var'][:,k])) \
                    - 0.5 * np.sum((x - model['mu'][:,k])**2 / model['var'][:,k], axis=1) \
                    + model['class_prior'][k]

    return model['classes'][np.argmax(pred, axis=1)]
    