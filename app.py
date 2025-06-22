import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import io
import time

# Stylish page config
st.set_page_config(
    page_title="✨ AI Digit Generator",
    page_icon="🎯",
    layout="centered"
)

# Custom CSS for style
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .digit-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .generated-grid {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    .stats-box {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 0.5rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model definition
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
        st.error(f"🚫 Error loading model: {str(e)}")
        return None, None

def generate_digit_images(model, digit, device):
    """Generate exactly 5 images of the specified digit with enhanced diversity"""
    model.eval()
    with torch.no_grad():
        # Generate more diverse samples
        z = torch.randn(5, 20, device=device)
        # Add different levels of noise for variety
        noise_scales = [0.2, 0.3, 0.4, 0.3, 0.25]
        for i in range(5):
            z[i] = z[i] + torch.randn(20, device=device) * noise_scales[i]
        
        labels = torch.tensor([digit] * 5, device=device)
        generated = model.decode(z, labels)
        generated = generated.view(5, 28, 28)
        return generated.cpu().numpy()

def create_stylish_plot(images, digit):
    """Create a stylish plot with gradient background"""
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.5))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Add gradient background
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='viridis', interpolation='bilinear')
        ax.set_title(f'Sample {i+1}', fontsize=12, fontweight='bold', color='#333')
        ax.axis('off')
        
        # Add subtle border
        for spine in ax.spines.values():
            spine.set_edgecolor('#ddd')
            spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig.suptitle(f'🎯 Generated Digit: {digit}', fontsize=16, fontweight='bold', color='#333')
    
    return fig

def main():
    # Stylish header
    st.markdown('<h1 class="main-header">✨ AI Digit Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">🧠 Powered by Conditional VAE • 🎨 Create Handwritten Digits</p>', unsafe_allow_html=True)
    
    # Load model with progress
    with st.spinner('🚀 Loading AI model...'):
        model, device = load_model()
    
    if model is None:
        st.stop()
    
    st.success("✅ AI Model loaded successfully!")
    
    # Main interface
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Digit selection with style
        st.markdown("### 🎯 Select Your Digit")
        
        # Create digit buttons in a grid
        digit_cols = st.columns(5)
        selected_digit = None
        
        for i in range(10):
            col_idx = i % 5
            if i == 5:
                digit_cols = st.columns(5)
                col_idx = 0
            
            with digit_cols[col_idx]:
                if st.button(f"{i}", key=f"digit_{i}", use_container_width=True):
                    selected_digit = i
        
        # Fallback dropdown
        if selected_digit is None:
            selected_digit = st.selectbox(
                "Or choose from dropdown:",
                options=list(range(10)),
                index=0,
                format_func=lambda x: f"Digit {x} 🔢"
            )
        
        # Display selected digit with style
        if selected_digit is not None:
            st.markdown(f"""
            <div class="digit-card">
                <h2>Selected Digit: {selected_digit}</h2>
                <p>Ready to generate 5 unique samples! 🎨</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate button with animation
            if st.button(f"🎲 Generate 5 Handwritten {selected_digit}s", type="primary", use_container_width=True):
                # Progress animation
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text('🧠 AI thinking...')
                    elif i < 60:
                        status_text.text('🎨 Creating pixels...')
                    elif i < 90:
                        status_text.text('✨ Adding magic...')
                    else:
                        status_text.text('🎯 Almost ready!')
                    time.sleep(0.01)
                
                # Generate images
                generated_images = generate_digit_images(model, selected_digit, device)
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                # Success message with style
                st.balloons()
                st.success(f"🎉 Generated 5 unique samples of digit {selected_digit}!")
                
                # Display results
                fig = create_stylish_plot(generated_images, selected_digit)
                st.pyplot(fig)
                plt.close()
                
                # Stats
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown('<div class="stats-box">🎯 Digit: ' + str(selected_digit) + '</div>', unsafe_allow_html=True)
                with col_b:
                    st.markdown('<div class="stats-box">📊 Samples: 5</div>', unsafe_allow_html=True)
                with col_c:
                    st.markdown('<div class="stats-box">🖼️ Size: 28×28</div>', unsafe_allow_html=True)
                
                # Individual images with hover effect
                with st.expander("🔍 View Individual Samples", expanded=False):
                    img_cols = st.columns(5)
                    for i, col in enumerate(img_cols):
                        with col:
                            fig_small, ax = plt.subplots(figsize=(2.5, 2.5))
                            ax.imshow(generated_images[i], cmap='plasma', interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title(f'Sample {i+1}', fontweight='bold')
                            
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
                            buf.seek(0)
                            st.image(buf)
                            plt.close()
    
    # Footer with style
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🤖 Built with Conditional VAE • 🎨 Trained on MNIST • ⚡ Powered by PyTorch</p>
        <p><small>Created by Raspberrynani • 2025</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
