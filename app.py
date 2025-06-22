import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import io
import time

st.set_page_config(page_title="🎯 Enhanced Digit Generator", page_icon="✨", layout="centered")

# Your EXISTING model (no changes needed)
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
        model.load_state_dict(torch.load('cvae_mnist_model.pth', map_location=device, weights_only=True))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_clean_digits(model, digit, device):
    """OPTIMIZED: Much cleaner output with less noise"""
    model.eval()
    with torch.no_grad():
        # REDUCED NOISE PATTERNS for cleaner output
        clean_patterns = [
            torch.randn(1, 20, device=device) * 0.5,      # Very clean (50% less noise)
            torch.randn(1, 20, device=device) * 0.6,      # Clean 
            torch.randn(1, 20, device=device) * 0.7,      # Slight variation
            torch.randn(1, 20, device=device) * 0.6 + 0.1, # Clean with shift
            torch.randn(1, 20, device=device) * 0.5 - 0.1  # Very clean with shift
        ]
        
        images = []
        for i in range(5):
            z = clean_patterns[i]
            label = torch.tensor([digit], device=device)
            
            # Generate and clean up the image
            generated = model.decode(z, label).view(28, 28).cpu().numpy()
            
            # POST-PROCESS for cleaner appearance
            generated = np.clip(generated, 0, 1)  # Ensure valid range
            generated = (generated > 0.3).astype(float)  # Threshold for cleaner lines
            
            images.append(generated)
        
        return np.array(images)

def create_beautiful_plot(images, digit):
    """Enhanced visualization with cleaner appearance"""
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.8))
    fig.patch.set_facecolor('#f8f9fa')
    
    for i, ax in enumerate(axes):
        # Clean display with high contrast
        ax.imshow(images[i], cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'Sample {i+1}', fontsize=13, fontweight='bold', color='#333', pad=10)
        ax.axis('off')
        
        # Add clean border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#ddd')
            spine.set_linewidth(1)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    fig.suptitle(f'✨ Clean Generated Digit: {digit}', fontsize=17, fontweight='bold', color='#333')
    
    return fig

# CLEAN UI
st.title("✨ Clean Digit Generator")
st.markdown("**Optimized for clean, readable digits**")

# Load model
model, device = load_model()
if model is None:
    st.stop()

st.success("✅ Model loaded - Ready to generate clean digits!")

# Simple selection
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    digit = st.selectbox(
        "🎯 Select digit to generate:",
        options=list(range(10)),
        index=0,
        format_func=lambda x: f"Digit {x} 🔢"
    )
    
    if st.button(f"✨ Generate Clean {digit}s", type="primary", use_container_width=True):
        with st.spinner("Creating clean digits..."):
            time.sleep(0.5)  # Brief pause for effect
            
            # Generate with reduced noise
            images = generate_clean_digits(model, digit, device)
        
        st.success(f"🎉 Generated 5 clean samples of digit {digit}!")
        
        # Display with enhanced visualization
        fig = create_beautiful_plot(images, digit)
        st.pyplot(fig)
        plt.close()
        
        # Clean metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("🎯 Digit", digit)
        with col_b:
            st.metric("📊 Samples", 5)
        with col_c:
            st.metric("✨ Quality", "Clean")

# Footer
st.markdown("---")
st.markdown("**🧠 Same model • ✨ Cleaner output • 🎯 Better readability**")
