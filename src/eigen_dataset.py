import os
import numpy as np
import trimesh
from torch.utils.data import Dataset, DataLoader
from eigen_decomp import compute_laplacian_eigenmodes

class EigenMeshDataset(Dataset):
    """
    A Dataset for loading 3D meshes and their precomputed Laplacian eigenmodes.
    """
    def __init__(self, data_dir="data/eigen_mesh", cache_dir="data/cache", k=64):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.k = k
        self.mesh_files = []
        
        # Ensure directories exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"Created directory {self.data_dir}. Please place .obj files here.")
            
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Gather all .obj files
        for f in os.listdir(self.data_dir):
            if f.endswith('.obj'):
                self.mesh_files.append(os.path.join(self.data_dir, f))
                
        self.mesh_files.sort()
        
    def prepare_cache(self):
        """
        Precomputes and caches the Laplacian eigenmodes for all meshes in the dataset.
        """
        if not self.mesh_files:
            print(f"No .obj files found in {self.data_dir}")
            return
            
        print(f"Checking and preparing cache for {len(self.mesh_files)} meshes...")
        for mesh_path in self.mesh_files:
            mesh_name = os.path.basename(mesh_path)
            cache_name = mesh_name.replace('.obj', f'_eigen_{self.k}.npz')
            cache_path = os.path.join(self.cache_dir, cache_name)
            
            if os.path.exists(cache_path):
                print(f"Cache already exists for {mesh_name}, skipping.")
                continue
                
            print(f"Computing {self.k} eigenmodes for {mesh_name}...")
            try:
                # Load mesh using trimesh
                mesh = trimesh.load(mesh_path, force='mesh')
                vertices = mesh.vertices
                faces = mesh.faces
                
                # Compute eigenmodes
                vals, vecs = compute_laplacian_eigenmodes(vertices, faces, k=self.k)
                
                # Save to cache
                np.savez(cache_path, eigenvals=vals, eigenvecs=vecs)
                print(f"Saved cache to {cache_path}")
            except Exception as e:
                print(f"Error processing {mesh_name}: {e}")

    def __len__(self):
        return len(self.mesh_files)
        
    def __getitem__(self, idx):
        mesh_path = self.mesh_files[idx]
        mesh_name = os.path.basename(mesh_path)
        cache_name = mesh_name.replace('.obj', f'_eigen_{self.k}.npz')
        cache_path = os.path.join(self.cache_dir, cache_name)
        
        # If cache is missing, compute it on the fly
        if not os.path.exists(cache_path):
            print(f"Cache missing for {mesh_name}, computing on the fly...")
            mesh = trimesh.load(mesh_path, force='mesh')
            vals, vecs = compute_laplacian_eigenmodes(mesh.vertices, mesh.faces, k=self.k)
            np.savez(cache_path, eigenvals=vals, eigenvecs=vecs)
        else:
            # Load from cache
            data = np.load(cache_path)
            vals = data['eigenvals']
            vecs = data['eigenvecs']
            
        return {
            'mesh_path': mesh_path,
            'eigenvals': vals,
            'eigenvecs': vecs
        }

if __name__ == "__main__":
    # Setup paths relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "eigen_mesh")
    cache_dir = os.path.join(project_root, "data", "cache")
    
    # Initialize dataset
    dataset = EigenMeshDataset(data_dir=data_dir, cache_dir=cache_dir, k=64)
    
    # Run the cache preparation step
    dataset.prepare_cache()
    
    # Test the DataLoader
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        print(f"\nSuccessfully loaded {len(dataset)} items into DataLoader.")
        
        for batch in dataloader:
            print(f"Batch item loaded: {batch['mesh_path'][0]}")
            print(f"Eigenvalues shape: {batch['eigenvals'].shape}")
            print(f"Eigenvectors shape: {batch['eigenvecs'].shape}")
            break
