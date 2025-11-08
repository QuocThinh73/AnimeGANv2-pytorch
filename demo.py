import streamlit as st
import io
import os
import pickle

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

# Helper function to load checkpoint with compatibility fixes
def load_checkpoint_compatible(checkpoint_path, device='cpu'):
    """
    Load checkpoint with multiple compatibility methods to handle
    PyTorch version mismatches, especially the _rebuild_device_tensor_from_cpu_tensor error.
    """
    # Fix: Monkey patch _rebuild_device_tensor_from_cpu_tensor if it doesn't exist
    # This handles the case where checkpoint was saved with newer PyTorch but loaded with older
    if not hasattr(torch._utils, '_rebuild_device_tensor_from_cpu_tensor'):
        def _rebuild_device_tensor_from_cpu_tensor(storage, device_str):
            """Fallback for missing _rebuild_device_tensor_from_cpu_tensor"""
            # Try to use _rebuild_tensor as fallback
            if hasattr(torch._utils, '_rebuild_tensor'):
                # Convert device string to device object
                device_obj = torch.device(device_str) if isinstance(device_str, str) else device_str
                return torch._utils._rebuild_tensor(storage, device_obj)
            else:
                # Last resort: return storage as tensor
                return storage
        torch._utils._rebuild_device_tensor_from_cpu_tensor = _rebuild_device_tensor_from_cpu_tensor
    
    # Method 1: Standard load with weights_only=False (PyTorch 2.0+)
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except (AttributeError, RuntimeError, pickle.UnpicklingError, TypeError) as e:
        pass
    
    # Method 2: Load without weights_only (older PyTorch or compatibility)
    try:
        return torch.load(checkpoint_path, map_location=device)
    except (AttributeError, RuntimeError, pickle.UnpicklingError, TypeError) as e:
        pass
    
    # Method 3: Try loading with pickle_module explicitly
    try:
        return torch.load(checkpoint_path, map_location=device, pickle_module=pickle)
    except Exception as e:
        pass
    
    # If all methods fail, raise error with helpful message
    raise RuntimeError(
        f"Không thể load checkpoint từ {checkpoint_path}. "
        "Lỗi có thể do không tương thích phiên bản PyTorch. "
        "Thử cài đặt lại PyTorch với phiên bản tương thích hoặc train lại model."
    )

# Page config
st.set_page_config(
    page_title="AnimeGANv2 Demo",
    page_icon="🎨",
    layout="wide"
)

# Title
st.title("🎨 AnimeGANv2 - Chuyển ảnh thành Anime")
st.markdown("Chọn checkpoint và upload ảnh để tạo ảnh anime style!")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if 'selected_epoch' not in st.session_state:
    st.session_state.selected_epoch = None

# Function to scan available checkpoints
def scan_checkpoints():
    """Scan for available checkpoints in output/animegan/checkpoints/"""
    checkpoints_dir = "output/animegan/checkpoints"
    available_epochs = []
    
    if os.path.exists(checkpoints_dir):
        # Get all epoch directories
        for item in os.listdir(checkpoints_dir):
            epoch_dir = os.path.join(checkpoints_dir, item)
            if os.path.isdir(epoch_dir) and item.startswith("epoch_"):
                g_path = os.path.join(epoch_dir, "G.pth")
                if os.path.exists(g_path):
                    # Extract epoch number
                    try:
                        epoch_num = int(item.replace("epoch_", ""))
                        available_epochs.append({
                            'epoch': epoch_num,
                            'name': item,
                            'path': g_path
                        })
                    except ValueError:
                        continue
    
    # Sort by epoch number
    available_epochs.sort(key=lambda x: x['epoch'], reverse=True)
    return available_epochs

# Sidebar for model loading
with st.sidebar:
    st.header("⚙️ Cài đặt Model")
    
    # Scan for available checkpoints
    available_checkpoints = scan_checkpoints()
    
    if not available_checkpoints:
        st.error("❌ Không tìm thấy checkpoint nào!")
        st.info("Vui lòng đảm bảo có checkpoint trong `output/animegan/checkpoints/epoch_xxx/`")
    else:
        st.success(f"✅ Tìm thấy {len(available_checkpoints)} checkpoint(s)")
        
        # Create list of epoch names for selectbox
        epoch_options = [f"Epoch {cp['epoch']:03d}" for cp in available_checkpoints]
        
        # Selectbox for choosing epoch
        selected_index = st.selectbox(
            "Chọn checkpoint:",
            options=range(len(epoch_options)),
            format_func=lambda x: epoch_options[x],
            help="Chọn epoch checkpoint bạn muốn sử dụng"
        )
        
        selected_checkpoint = available_checkpoints[selected_index]
        st.info(f"📁 Đường dẫn: `{selected_checkpoint['path']}`")
        
        # Load model button
        load_model = st.button("🔄 Load Model", type="primary")
        
        if load_model:
            checkpoint_path = selected_checkpoint['path']
            try:
                with st.spinner("Đang load model..."):
                    # Initialize model
                    model = AnimeGANGenerator().to(st.session_state.device)
                    
                    # Load checkpoint with compatibility handling
                    # Use helper function that tries multiple methods
                    state_dict = load_checkpoint_compatible(checkpoint_path, device='cpu')
                    
                    # Load state dict to model
                    model.load_state_dict(state_dict)
                    
                    # Move model to target device after loading
                    model = model.to(st.session_state.device)
                    model.eval()
                    
                    # Save to session state
                    st.session_state.model = model
                    st.session_state.selected_epoch = selected_checkpoint['epoch']
                    
                    st.success(f"✅ Model đã được load thành công! (Epoch {selected_checkpoint['epoch']:03d})")
                    
                    # Show device info
                    device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
                    st.info(f"Đang sử dụng: {device_name}")
                    
            except Exception as e:
                st.error(f"Lỗi khi load model: {str(e)}")
                st.exception(e)
                st.markdown("""
                **Gợi ý khắc phục:**
                - Lỗi này thường do không tương thích phiên bản PyTorch
                - Thử cài đặt lại PyTorch với phiên bản tương thích
                - Hoặc train lại model với phiên bản PyTorch hiện tại
                """)
        
        # Show current loaded model info
        if st.session_state.model is not None and st.session_state.selected_epoch is not None:
            st.markdown("---")
            st.success(f"✅ Model hiện tại: Epoch {st.session_state.selected_epoch:03d}")

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
1. **Chọn và Load Model**: 
   - Ở sidebar, chọn checkpoint từ danh sách các epoch có sẵn
   - Click nút "Load Model" để load model vào memory
   - Checkpoint được tự động quét từ `output/animegan/checkpoints/epoch_xxx/`

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
- Checkpoint được sắp xếp theo epoch (mới nhất ở trên)
""")

