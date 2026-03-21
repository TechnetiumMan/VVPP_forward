import os
import torch
import torchaudio
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import trimesh
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile

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
                            
                        self.samples.append({
                            'wav_path': wav_path,
                            'mesh_path': mesh_path,
                            'img_path': img_path,
                            'obj_id': obj_id,
                            'vertex_id': vertex_id
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
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
        
        # Load Image
        image = Image.open(sample_info['img_path']).convert('RGB')
        image_tensor = self.transform_image(image)
        
        # Return paths for mesh instead of loading object for memory efficiency
        return {
            'mel_spectrogram': mel_spec_db.squeeze(0), # Shape: (n_mels, time)
            'image': image_tensor,
            'mesh_path': sample_info['mesh_path'],
            'obj_id': sample_info['obj_id'],
            'vertex_id': sample_info['vertex_id']
        }

def visualize_sample(batch, save_path="sample_visualization.png"):
    '''
    Visualize a single sample from the DataLoader.
    Since DataLoader returns batched elements, we visualize the first item.
    '''
    mel_spec = batch['mel_spectrogram'][0].numpy()
    image = batch['image'][0].permute(1, 2, 0).numpy()
    mesh_path = batch['mesh_path'][0]
    obj_id = batch['obj_id'][0]
    vertex_id = batch['vertex_id'][0]
    
    # Load Mesh
    # force='mesh' forces trimesh to return a single mesh object (useful if scene)
    mesh = trimesh.load(mesh_path, force='mesh')
    vertices = mesh.vertices
    faces = mesh.faces
    
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"Object: {obj_id} | Vertex: {vertex_id}", fontsize=16)
    
    # 1. Plot Image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image)
    ax1.set_title("Rendered Image (0.png)")
    ax1.axis('off')
    
    # 2. Plot Mel Spectrogram
    ax2 = fig.add_subplot(1, 3, 2)
    cax = ax2.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_title("Mel Spectrogram")
    ax2.set_xlabel("Time Frames")
    ax2.set_ylabel("Mel Frequency Bins")
    fig.colorbar(cax, ax=ax2, format='%+2.0f dB')
    
    # 3. Plot 3D Mesh
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    # Subsample faces if mesh is too large to plot efficiently
    if len(faces) > 10000:
        step = len(faces) // 10000
        faces_to_plot = faces[::step]
    else:
        faces_to_plot = faces
        
    ax3.plot_trisurf(vertices[:, 0], vertices[:, 1], faces_to_plot, vertices[:, 2], 
                     cmap='viridis', edgecolor='none', alpha=0.5)
    ax3.set_title("3D Mesh")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    # Highlight the specific vertex if it corresponds to an index
    try:
        v_idx = int(vertex_id)
        if v_idx < len(vertices):
            v_point = vertices[v_idx]
            ax3.scatter(v_point[0], v_point[1], v_point[2], color='red', s=100, label=f'Vertex {v_idx}')
            ax3.legend()
    except ValueError:
        pass
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization successfully saved to {save_path}")

if __name__ == "__main__":
    # Test the dataset and visualization
    # Get the directory of the current script, then go up one level to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    
    dataset = VVImpactDataset(data_dir=data_dir)
    print(f"Total valid samples found: {len(dataset)}")
    
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        batch = next(iter(dataloader))
        
        print("Batch keys:", batch.keys())
        print("Mel spectrogram shape:", batch['mel_spectrogram'].shape)
        print("Image shape:", batch['image'].shape)
        
        visualize_sample(batch)
    else:
        print("No valid samples found. Please check your dataset path and structure.")