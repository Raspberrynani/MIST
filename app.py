import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# Model definition (same as training)
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
        super(ConditionalVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        self.num_classes = num_classes
        
    def decode(self, z, y):
        y_onehot = F.one_hot(y, self.num_classes).float()
        zy = torch.cat([z, y_onehot], dim=1)
        return self.decoder(zy)

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = ConditionalVAE(784, 400, 20, 10)
    
    try:
        model.load_state_dict(torch.load('cvae_mnist_model.pth', map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'cvae_mnist_model.pth' is uploaded.")
        return None, None

def generate_digits(model, digit, num_samples=5, latent_dim=20, device='cpu'):
    model.eval()
    with torch.no_grad():
        # Create diverse latent vectors
        z = torch.randn(num_samples, latent_dim).to(device)
        z = z + torch.randn_like(z) * 0.3  # Add diversity
        
        labels = torch.tensor([digit] * num_samples).to(device)
        generated = model.decode(z, labels)
        generated = generated.view(num_samples, 28, 28)
        return generated.cpu().numpy()

def main():
    st.title("🔢 MNIST Handwritten Digit Generator")
    st.markdown("### Generate realistic handwritten digits using a Conditional Variational Autoencoder")
    
    # Load model
    model, device = load_model()
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("🎮 Controls")
    st.sidebar.markdown("Select a digit and generate 5 unique samples!")
    
    selected_digit = st.sidebar.selectbox(
        "Choose digit to generate:",
        options=list(range(10)),
        index=0,
        format_func=lambda x: f"Digit {x}"
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎲 Generate 5 Images", type="primary", use_container_width=True):
            with st.spinner(f"Generating digit {selected_digit}..."):
                generated_images = generate_digits(model, selected_digit, 5, 20, device)
                
                st.success(f"✅ Generated 5 samples of digit **{selected_digit}**!")
                
                # Display in grid
                fig, axes = plt.subplots(1, 5, figsize=(12, 2.5))
                for i, ax in enumerate(axes):
                    ax.imshow(generated_images[i], cmap='gray', interpolation='nearest')
                    ax.set_title(f'Sample {i+1}', fontsize=12, fontweight='bold')
                    ax.axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Individual images
                st.subheader("Individual Samples (28×28 pixels)")
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    with col:
                        fig_ind, ax = plt.subplots(figsize=(3, 3))
                        ax.imshow(generated_images[i], cmap='gray', interpolation='nearest')
                        ax.set_title(f'Sample {i+1}', fontweight='bold')
                        ax.axis('off')
                        
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        buf.seek(0)
                        st.image(buf, width=120)
                        plt.close()
    
    with col2:
        st.markdown("### 📊 Model Info")
        st.info("""
        **Architecture:** Conditional VAE
        
        **Dataset:** MNIST (60,000 samples)
        
        **Input:** 28×28 grayscale
        
        **Latent Dim:** 20
        
        **Training:** From scratch on T4 GPU
        """)
        
        st.markdown("### 🎯 How to Use")
        st.markdown("""
        1. **Select** a digit (0-9)
        2. **Click** Generate button
        3. **View** 5 unique variations
        4. Each image is 28×28 pixels
        5. Generated in MNIST style
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🧠 About This Model")
    
    tab1, tab2, tab3 = st.tabs(["Model Details", "Architecture", "Performance"])
    
    with tab1:
        st.markdown("""
        This **Conditional Variational Autoencoder (CVAE)** was trained from scratch on the MNIST dataset.
        The model learns to generate handwritten digits by encoding them into a latent space and then
        decoding them back to images, conditioned on the desired digit class.
        
        **Key Features:**
        - ✅ Generates diverse, realistic digits
        - ✅ Fast inference (< 1 second)
        - ✅ Consistent digit recognition
        - ✅ MNIST-compatible format
        """)
    
    with tab2:
        st.markdown("""
        **Encoder:**
        - Input: 784 (flattened 28×28) + 10 (one-hot digit)
        - Hidden layers: 400 → 400 neurons
        - Output: μ and σ for 20D latent space
        
        **Decoder:**
        - Input: 20D latent vector + 10 (one-hot digit)
        - Hidden layers: 400 → 400 neurons  
        - Output: 784 (reconstructed image)
        """)
    
    with tab3:
        st.markdown("""
        **Training Details:**
        - **Epochs:** 30
        - **Batch Size:** 128
        - **Learning Rate:** 0.001
        - **Loss:** Reconstruction + KL Divergence
        - **Hardware:** Google Colab T4 GPU
        - **Training Time:** ~20 minutes
        """)

if __name__ == "__main__":
    main()
