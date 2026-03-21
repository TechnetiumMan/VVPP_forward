import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torchaudio
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import trimesh
import meshio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.spatial import KDTree
import polyscope as ps
import polyscope.imgui as psim
from eigen_decomp import compute_laplacian_eigenmodes
import sounddevice as sd
from collections import defaultdict
import multiprocessing as mp

class VVImpactDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, n_mels=64, transform_image=None):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Audio transform to Mel Spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        
        self.transform_image = transform_image or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        
        self.samples = []
        self.obj_to_samples = defaultdict(list)
        
        # Traverse audios directory
        audios_dir = os.path.join(data_dir, 'audios')
        if not os.path.exists(audios_dir):
            return
            
        for group in os.listdir(audios_dir):
            group_path = os.path.join(audios_dir, group)
            if not os.path.isdir(group_path):
                continue
                
            for obj_id in os.listdir(group_path):
                obj_dir = os.path.join(group_path, obj_id)
                if not os.path.isdir(obj_dir):
                    continue
                    
                mesh_path = os.path.join(data_dir, 'mesh', group, f"{obj_id}.obj")
                img_path = os.path.join(data_dir, 'images', group, obj_id, "0.png")
                
                # Check if corresponding mesh and image exist
                if not os.path.exists(mesh_path) or not os.path.exists(img_path):
                    continue
                
                for wav_file in os.listdir(obj_dir):
                    if wav_file.endswith('.wav'):
                        wav_path = os.path.join(obj_dir, wav_file)
                        
                        # Extract vertex id from 'audio_<vid>.wav'
                        vertex_id = 'unknown'
                        if '_' in wav_file:
                            vertex_id = wav_file.split('_')[1].split('.')[0]
                            
                        sample_info = {
                            'wav_path': wav_path,
                            'mesh_path': mesh_path,
                            'img_path': img_path,
                            'obj_id': obj_id,
                            'vertex_id': vertex_id
                        }
                        self.samples.append(sample_info)
                        self.obj_to_samples[obj_id].append(sample_info)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        return self.load_sample(sample_info)
        
    def load_sample(self, sample_info, load_img=True):
        # Load Audio using scipy
        sr, waveform_np = wavfile.read(sample_info['wav_path'])
        
        # Convert to float32 if not already
        if np.issubdtype(waveform_np.dtype, np.integer):
            waveform_np = waveform_np.astype(np.float32) / np.iinfo(waveform_np.dtype).max
        elif waveform_np.dtype != np.float32:
            waveform_np = waveform_np.astype(np.float32)
            
        waveform = torch.from_numpy(waveform_np).unsqueeze(0) # Shape: (1, time)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to Mel Spectrogram and then to dB
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.db_transform(mel_spec)
        
        result = {
            'mel_spectrogram': mel_spec_db.squeeze(0), # Shape: (n_mels, time)
            'mesh_path': sample_info['mesh_path'],
            'obj_id': sample_info['obj_id'],
            'vertex_id': sample_info['vertex_id'],
            'waveform': waveform.squeeze(0).numpy(),
            'sample_rate': self.sample_rate
        }
        
        if load_img:
            # Load Image
            image = Image.open(sample_info['img_path']).convert('RGB')
            image_tensor = self.transform_image(image)
            result['image'] = image_tensor
            
        return result

def play_audio_process(waveform, sample_rate):
    """Function to play audio in a separate process"""
    sd.play(waveform, sample_rate)
    sd.wait()

class PolyscopeViewer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.obj_ids = list(dataset.obj_to_samples.keys())
        self.current_obj_idx = 0
        self.current_mesh = None
        self.current_ps_mesh = None
        self.current_ps_cloud = None
        self.current_samples = []
        self.vertex_to_sample = {}
        self.audio_process = None
        
        self.num_eigenmodes = 0
        self.current_eigenmode_idx = 0
        self.eigenvals = None
        self.eigenvecs = None
        
        # Initialize polyscope
        ps.init()
        ps.set_program_name("VV-Impact Viewer")
        ps.set_up_dir("y_up")
        
        # Register UI callback
        ps.set_user_callback(self.ui_callback)
        
        # Load initial object
        if self.obj_ids:
            self.load_object(self.obj_ids[0])
            
    def load_object(self, obj_id):
        ps.remove_all_structures()
        
        self.current_samples = self.dataset.obj_to_samples[obj_id]
        if not self.current_samples:
            return
            
        mesh_path = self.current_samples[0]['mesh_path']
        msh_path = mesh_path + "_.msh"
        
        # Load tetrahedral mesh using meshio
        self.current_mesh = meshio.read(msh_path)
        
        # Extract tetrahedra
        tets = self.current_mesh.cells_dict.get('tetra', None)
        if tets is None:
            for cell_block in self.current_mesh.cells:
                if cell_block.type == "tetra":
                    tets = cell_block.data
                    break
                    
        if tets is None:
            print(f"Warning: No tetrahedra found in {msh_path}")
            return
            
        # Register volume mesh to polyscope
        self.current_ps_mesh = ps.register_volume_mesh(
            f"Mesh_{obj_id}", 
            self.current_mesh.points, 
            tets=tets,
            color=[0.8, 0.8, 0.8],
            edge_width=1.0
        )
        
        # Map vertices to samples
        self.vertex_to_sample = {}
        valid_vertices = []
        features_64d_list = []
        
        print(f"Loading and processing audio for PCA... (Object: {obj_id})")
        for sample in self.current_samples:
            try:
                vid = int(sample['vertex_id'])
                self.vertex_to_sample[vid] = sample
                valid_vertices.append(vid)
                
                # Load sample without image to save time
                data = self.dataset.load_sample(sample, load_img=False)
                # Compute 64D feature: mean along time dimension
                mel_spec = data['mel_spectrogram']
                mel_mean = mel_spec.mean(dim=1)
                features_64d_list.append(mel_mean)
            except ValueError:
                pass
                
        # Visualize audio vertices as a point cloud
        if valid_vertices:
            audio_points = self.current_mesh.points[valid_vertices]
            self.current_ps_cloud = ps.register_point_cloud(
                f"Audio_Vertices_{obj_id}",
                audio_points,
                radius=0.01
            )
            
            if len(features_64d_list) > 0:
                features_tensor = torch.stack(features_64d_list) # (N, 64)
                n_samples = features_tensor.shape[0]
                n_components = min(3, n_samples)
                
                # Apply PCA using torch.pca_lowrank
                mean_feat = features_tensor.mean(dim=0, keepdim=True)
                centered_features = features_tensor - mean_feat
                
                # Note: pca_lowrank returns U, S, V
                U, S, V = torch.pca_lowrank(features_tensor, q=n_components, center=True)
                features_3d = torch.matmul(centered_features, V[:, :n_components]).numpy()
                
                # Normalize to [0, 1] for RGB mapping
                min_vals = features_3d.min(axis=0)
                max_vals = features_3d.max(axis=0)
                ranges = max_vals - min_vals
                # Prevent division by zero
                ranges[ranges == 0] = 1.0
                
                colors_rgb = (features_3d - min_vals) / ranges
                
                # Pad to 3 channels if fewer components
                if colors_rgb.shape[1] < 3:
                    padded = np.zeros((n_samples, 3))
                    padded[:, :colors_rgb.shape[1]] = colors_rgb
                    colors_rgb = padded
                    
                # Add PCA colors to the point cloud
                self.current_ps_cloud.add_color_quantity("PCA_RGB", colors_rgb, enabled=True)
                
                # Interpolate PCA RGB to the entire mesh using Inverse Distance Weighting (IDW)
                # This provides much smoother transitions than Nearest Neighbor
                all_mesh_points = self.current_mesh.points
                tree = KDTree(audio_points)
                k_neighbors = min(10, len(audio_points))
                distances, indices = tree.query(all_mesh_points, k=k_neighbors)
                
                # Calculate weights (inverse distance squared)
                eps = 1e-8
                weights = 1.0 / (distances + eps)**2
                weights /= weights.sum(axis=1, keepdims=True)
                
                # Apply weights to colors
                neighbor_colors = colors_rgb[indices] # Shape: (N, k, 3)
                weights_expanded = np.expand_dims(weights, axis=2) # Shape: (N, k, 1)
                mesh_colors_rgb = np.sum(neighbor_colors * weights_expanded, axis=1) # Shape: (N, 3)
                
                # Add the interpolated colors to the mesh
                self.current_ps_mesh.add_color_quantity("Interpolated_PCA_RGB", mesh_colors_rgb, defined_on='vertices', enabled=True)
            
            # Initialize vertex selection
            valid_vids = sorted(list(self.vertex_to_sample.keys()))
            self._v_idx = 0
            if valid_vids:
                self.highlight_selected_vertex(valid_vids[0])
                
        # Compute Laplacian eigenmodes
        try:
            print(f"Computing Laplacian eigenmodes for {obj_id} (this may take a few seconds)...")
            self.eigenvals, self.eigenvecs = compute_laplacian_eigenmodes(self.current_mesh.points, tets, k=64)
            self.num_eigenmodes = self.eigenvecs.shape[1]
            self.current_eigenmode_idx = 0
            self.update_eigenmode_visualization()
            print("Eigenmodes computed successfully.")
        except Exception as e:
            print(f"Failed to compute eigenmodes: {e}")
            self.num_eigenmodes = 0

    def update_eigenmode_visualization(self):
        if self.num_eigenmodes > 0 and self.current_ps_mesh is not None:
            mode_data = self.eigenvecs[:, self.current_eigenmode_idx]
            self.current_ps_mesh.add_scalar_quantity(
                "Eigenmode", 
                mode_data, 
                defined_on='vertices', 
                cmap='coolwarm', 
                enabled=True
            )

    def highlight_selected_vertex(self, vertex_id):
        if self.current_mesh is None:
            return
        # Get the 3D coordinate of the selected vertex
        pos = self.current_mesh.points[vertex_id]
        # Register a single point cloud for the highlighted vertex
        ps.register_point_cloud(
            "Selected_Vertex",
            np.array([pos]),
            radius=0.02, # slightly larger
            color=[0.0, 1.0, 0.0] # Green
        )
            
    def play_audio_and_show_mel(self, vertex_id):
        if vertex_id not in self.vertex_to_sample:
            print(f"No audio found for vertex {vertex_id}")
            return
            
        sample_info = self.vertex_to_sample[vertex_id]
        data = self.dataset.load_sample(sample_info)
        
        # Play audio in a separate process
        print(f"Playing audio for vertex {vertex_id}...")
        if self.audio_process and self.audio_process.is_alive():
            self.audio_process.terminate()
            
        self.audio_process = mp.Process(target=play_audio_process, args=(data['waveform'], data['sample_rate']))
        self.audio_process.start()
        
        # Show Mel Spectrogram
        mel_spec = data['mel_spectrogram'].numpy()
        
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"Mel Spectrogram - Object: {data['obj_id']} | Vertex: {vertex_id}")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Bins")
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show(block=False)
        
    def ui_callback(self):
        psim.PushItemWidth(150)
        
        # Object selection
        changed, self.current_obj_idx = psim.Combo(
            "Select Object", 
            self.current_obj_idx, 
            self.obj_ids
        )
        if changed:
            self.load_object(self.obj_ids[self.current_obj_idx])
            
        psim.Separator()
        
        # Vertex selection and playback
        psim.TextUnformatted("Vertex Selection:")
        
        # List of available vertices
        valid_vids = sorted(list(self.vertex_to_sample.keys()))
        if valid_vids:
            # Ensure _v_idx is within bounds
            if not hasattr(self, '_v_idx') or self._v_idx >= len(valid_vids):
                self._v_idx = 0
                
            # Slider for vertex index
            changed, new_v_idx = psim.SliderInt("Vertex Index", self._v_idx, 0, len(valid_vids) - 1)
            if changed:
                self._v_idx = new_v_idx
                self.highlight_selected_vertex(valid_vids[self._v_idx])
                
            # Next / Prev buttons
            if psim.Button("< Prev") and self._v_idx > 0:
                self._v_idx -= 1
                self.highlight_selected_vertex(valid_vids[self._v_idx])
            psim.SameLine()
            if psim.Button("Next >") and self._v_idx < len(valid_vids) - 1:
                self._v_idx += 1
                self.highlight_selected_vertex(valid_vids[self._v_idx])
                
            current_vid = valid_vids[self._v_idx]
            psim.TextUnformatted(f"Current Vertex ID: {current_vid}")
            
            if psim.Button("Play Audio & Show Mel Spectrogram"):
                self.play_audio_and_show_mel(current_vid)
        else:
            psim.TextUnformatted("No audio vertices found for this object.")
            
        psim.Separator()
        psim.TextUnformatted("Laplacian Eigenmodes:")
        if hasattr(self, 'num_eigenmodes') and self.num_eigenmodes > 0:
            changed, new_idx = psim.SliderInt("Mode Index", self.current_eigenmode_idx, 0, self.num_eigenmodes - 1)
            if changed:
                self.current_eigenmode_idx = new_idx
                self.update_eigenmode_visualization()
            psim.TextUnformatted(f"Eigenvalue: {self.eigenvals[self.current_eigenmode_idx]:.6f}")
        else:
            psim.TextUnformatted("Eigenmodes not available.")
            
        psim.PopItemWidth()

    def run(self):
        ps.show()
        
        # Cleanup process on exit
        if self.audio_process and self.audio_process.is_alive():
            self.audio_process.terminate()

if __name__ == "__main__":
    # Get the directory of the current script, then go up one level to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    dataset = VVImpactDataset(data_dir=data_dir)
    
    if len(dataset) > 0:
        print(f"Loaded {len(dataset)} samples across {len(dataset.obj_to_samples)} objects.")
        viewer = PolyscopeViewer(dataset)
        viewer.run()
    else:
        print("No valid samples found. Please check your dataset path and structure.")