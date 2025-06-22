import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import io

# Minimal page config
st.set_page_config(
    page_title="Digit Generator",
    page_icon="🔢",
    layout="centered"
)

# Lean model definition
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

def generate_digit_images(model, digit, device):
    """Generate exactly 5 images of the specified digit"""
    model.eval()
    with torch.no_grad():
        # Generate 5 diverse samples
        z = torch.randn(5, 20, device=device)
        z = z + torch.randn_like(z) * 0.3  # Add diversity
        
        labels = torch.tensor([digit] * 5, device=device)
        generated = model.decode(z, labels)
        generated = generated.view(5, 28, 28)
        return generated.cpu().numpy()

def display_images(images, digit):
    """Display 5 images in a clean grid"""
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    fig.suptitle(f'Generated Digit: {digit}', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray', interpolation='nearest')
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Clean title
    st.title("Handwritten Digit Generator")
    st.markdown("Select a digit and generate 5 handwritten samples")
    
    # Load model
    model, device = load_model()
    if model is None:
        st.stop()
    
    # Digit selection - centered and prominent
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_digit = st.selectbox(
            "Choose digit to generate:",
            options=list(range(10)),
            index=0,
            key="digit_selector"
        )
        
        # Generate button - full width in center column
        generate_button = st.button(
            f"Generate 5 images of digit {selected_digit}",
            type="primary",
            use_container_width=True
        )
    
    # Generation and display
    if generate_button:
        with st.spinner("Generating..."):
            # Generate 5 images
            generated_images = generate_digit_images(model, selected_digit, device)
            
        # Display results
        st.success(f"Generated 5 samples of digit {selected_digit}")
        
        # Show images
        fig = display_images(generated_images, selected_digit)
        st.pyplot(fig)
        plt.close()
        
        # Optional: Individual images in smaller format
        with st.expander("View individual images"):
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    fig_small, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(generated_images[i], cmap='gray')
                    ax.axis('off')
                    ax.set_title(f'#{i+1}')
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    st.image(buf)
                    plt.close()

if __name__ == "__main__":
    main()
