import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def compute_laplacian_eigenmodes(points, elements, k=10):
    """
    Computes the first k non-trivial eigenmodes of the graph Laplacian for a given mesh.
    
    Args:
        points: (N, 3) numpy array of vertex coordinates.
        elements: (M, 3) or (M, 4) numpy array of faces or tetrahedra.
        k: number of eigenmodes to return.
        
    Returns:
        vals: (k,) eigenvalues.
        vecs: (N, k) eigenvectors.
    """
    num_points = len(points)
    
    # Extract edges based on element type (triangles or tetrahedra)
    if elements.shape[1] == 3: # Triangles
        edges = np.vstack((elements[:, [0, 1]], 
                           elements[:, [1, 2]], 
                           elements[:, [2, 0]]))
    elif elements.shape[1] == 4: # Tetrahedra
        edges = np.vstack((elements[:, [0, 1]], elements[:, [0, 2]], elements[:, [0, 3]],
                           elements[:, [1, 2]], elements[:, [1, 3]], elements[:, [2, 3]]))
    else:
        raise ValueError("Elements must be triangles (3 vertices) or tetrahedra (4 vertices).")
        
    # Ensure unique undirected edges
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    
    # Build sparse adjacency matrix
    data = np.ones(len(edges))
    A = sp.coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(num_points, num_points))
    A = A + A.T # Make it symmetric
    
    # Build degree matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    
    # Compute graph Laplacian
    L = D - A
    
    # Compute eigenvalues and eigenvectors
    # We compute k+1 modes because the first one is the trivial constant mode (eigenvalue ~ 0)
    # Using shift-invert mode (sigma=-1e-5) is much faster and more stable for finding eigenvalues near 0
    vals, vecs = eigsh(L.astype(float), k=k+1, sigma=-1e-5, which='LM')
    
    # Sort the results by eigenvalue
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # Skip the first mode (constant mode with eigenvalue 0)
    return vals[1:], vecs[:, 1:]
