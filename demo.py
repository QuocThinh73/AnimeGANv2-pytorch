import streamlit as st
import io
import os

# Try to import PyTorch and related libraries
try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from models import AnimeGANGenerator
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    TORCH_ERROR = str(e)
except OSError as e:
    TORCH_AVAILABLE = False
    TORCH_ERROR = str(e)

# Page config
st.set_page_config(
    page_title="AnimeGANv2 Demo",
    page_icon="🎨",
    layout="wide"
)

# Title
st.title("🎨 AnimeGANv2 - Chuyển ảnh thành Anime")

st.markdown("Upload checkpoint G.pth và ảnh để tạo ảnh anime style!")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sidebar for model loading
with st.sidebar:
    st.header("⚙️ Cài đặt Model")
    
    # Option 1: Upload checkpoint file
    st.subheader("1. Upload Checkpoint")
    uploaded_checkpoint = st.file_uploader(
        "Chọn file G.pth",
        type=['pth'],
        help="Upload file checkpoint của Generator (G.pth)"
    )
    
    # Option 2: Use default checkpoint
    st.subheader("2. Hoặc sử dụng checkpoint mặc định")
    default_checkpoint_path = "output/G.pth"
    use_default = st.checkbox("Sử dụng checkpoint mặc định (output/G.pth)", value=False)
    
    # Load model button
    load_model = st.button("🔄 Load Model", type="primary")
    
    if load_model:
        checkpoint_path = None
        
        if use_default and os.path.exists(default_checkpoint_path):
            checkpoint_path = default_checkpoint_path
            st.success(f"Đang sử dụng checkpoint mặc định: {default_checkpoint_path}")
        elif uploaded_checkpoint is not None:
            # Save uploaded file temporarily
            with open("temp_G.pth", "wb") as f:
                f.write(uploaded_checkpoint.getbuffer())
            checkpoint_path = "temp_G.pth"
            st.success("Đã upload checkpoint!")
        else:
            st.error("Vui lòng upload checkpoint hoặc chọn sử dụng checkpoint mặc định!")
            checkpoint_path = None
        
        if checkpoint_path:
            try:
                with st.spinner("Đang load model..."):
                    # Initialize model
                    model = AnimeGANGenerator().to(st.session_state.device)
                    
                    # Load checkpoint
                    state_dict = torch.load(checkpoint_path, map_location=st.session_state.device)
                    model.load_state_dict(state_dict)
                    model.eval()
                    
                    # Save to session state
                    st.session_state.model = model
                    
                    st.success("✅ Model đã được load thành công!")
                    
                    # Show device info
                    device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
                    st.info(f"Đang sử dụng: {device_name}")
                    
            except Exception as e:
                st.error(f"Lỗi khi load model: {str(e)}")
    
    # Clean up temp file
    if uploaded_checkpoint and os.path.exists("temp_G.pth"):
        pass  # Keep it for now, will be cleaned up later

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("📸 Ảnh gốc")
    
    # Image upload
    uploaded_image = st.file_uploader(
        "Chọn ảnh để chuyển đổi",
        type=['png', 'jpg', 'jpeg'],
        help="Upload ảnh bạn muốn chuyển thành anime style"
    )
    
    if uploaded_image is not None:
        # Display original image
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption="Ảnh gốc", use_container_width=True)
        
        # Image info
        st.info(f"Kích thước: {image.size[0]} x {image.size[1]} pixels")

with col2:
    st.header("🎨 Ảnh Anime")
    
    if st.session_state.model is None:
        st.warning("⚠️ Vui lòng load model trước (bên sidebar)")
    else:
        if uploaded_image is not None:
            # Inference button
            if st.button("✨ Tạo ảnh Anime", type="primary"):
                try:
                    with st.spinner("Đang xử lý..."):
                        # Preprocess image
                        transform = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                        ])
                        
                        # Convert PIL to tensor
                        image_tensor = transform(image).unsqueeze(0).to(st.session_state.device)
                        
                        # Inference
                        with torch.no_grad():
                            output = st.session_state.model(image_tensor)
                            
                            # Denormalize: from [-1, 1] to [0, 1]
                            output = output * 0.5 + 0.5
                            output = torch.clamp(output, 0, 1)
                            
                            # Convert to PIL Image
                            output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
                            
                            # Display result
                            st.image(output_image, caption="Ảnh Anime", use_container_width=True)
                            
                            # Download button
                            buf = io.BytesIO()
                            output_image.save(buf, format='PNG')
                            st.download_button(
                                label="💾 Tải ảnh về",
                                data=buf.getvalue(),
                                file_name="anime_result.png",
                                mime="image/png"
                            )
                            
                            st.success("✅ Hoàn thành!")
                            
                except Exception as e:
                    st.error(f"Lỗi khi xử lý ảnh: {str(e)}")
                    st.exception(e)
        else:
            st.info("👆 Upload ảnh ở cột bên trái để bắt đầu")

# Footer
st.markdown("---")
st.markdown("### 📝 Hướng dẫn sử dụng:")
st.markdown("""
1. **Load Model**: 
   - Upload file G.pth ở sidebar hoặc chọn sử dụng checkpoint mặc định
   - Click nút "Load Model" để load model vào memory

2. **Upload Ảnh**: 
   - Chọn ảnh bạn muốn chuyển đổi (PNG, JPG, JPEG)

3. **Tạo Ảnh Anime**: 
   - Click nút "Tạo ảnh Anime" để thực hiện inference
   - Kết quả sẽ hiển thị ở cột bên phải
   - Bạn có thể tải ảnh kết quả về máy

**Lưu ý**: 
- Model sẽ được resize ảnh về 256x256 pixels
- Sử dụng GPU sẽ nhanh hơn CPU
- Model chỉ cần load 1 lần, có thể dùng cho nhiều ảnh
""")

# Cleanup temp file on app restart
if os.path.exists("temp_G.pth"):
    try:
        os.remove("temp_G.pth")
    except:
        pass

