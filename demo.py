import streamlit as st
import os
import tempfile
import torch
from torchvision import transforms
from PIL import Image
import io

# Import logic từ infer.py
try:
    from infer import infer, detect_model_type
except ImportError:
    st.error(
        "❌ Không tìm thấy file 'infer.py'. Vui lòng đảm bảo file này nằm cùng thư mục với demo.py")
    st.stop()

# Page config
st.set_page_config(
    page_title="AnimeGANv2/CycleGAN Demo",
    page_icon="🎨",
    layout="wide"
)

# Title
st.title("🎨 AI Art Style Transfer")
st.markdown("Chuyển đổi ảnh thật sang Anime/Art style sử dụng model của bạn.")

# Initialize session state
if 'selected_model_path' not in st.session_state:
    st.session_state.selected_model_path = None

# --- Function 1: Recursive Scan ---


def scan_models_recursive():
    """
    Quét toàn bộ file .pth trong folder 'output' và các thư mục con.
    """
    root_dir = "generators"
    available_models = []

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        return []

    # Sử dụng os.walk để duyệt cây thư mục
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pth'):
                full_path = os.path.join(root, file)
                # Tạo tên hiển thị (vd: folder_con/model.pth)
                rel_path = os.path.relpath(full_path, root_dir)

                # Xác định loại model để hiển thị icon cho đẹp
                try:
                    model_type = detect_model_type(full_path)
                    icon = "🎌" if model_type == "animegan" else "🔄"
                except:
                    icon = "❓"
                    model_type = "unknown"

                available_models.append({
                    'display_name': f"{icon} {rel_path}",
                    'full_path': full_path,
                    'filename': file,
                    'type': model_type
                })

    # Sắp xếp theo tên
    available_models.sort(key=lambda x: x['display_name'])
    return available_models


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Chọn Model")

    models = scan_models_recursive()

    if not models:
        st.warning("⚠️ Không tìm thấy file .pth nào trong folder `output/`")
        st.info("Hãy copy file checkpoint vào `output/` hoặc các sub-folder của nó.")
    else:
        # Selectbox
        selected_index = st.selectbox(
            "Danh sách Model có sẵn:",
            options=range(len(models)),
            format_func=lambda x: models[x]['display_name']
        )

        selected_item = models[selected_index]
        st.session_state.selected_model_path = selected_item['full_path']

        st.success(f"✅ Đã chọn: **{selected_item['filename']}**")
        st.caption(f"Đường dẫn: `{selected_item['full_path']}`")
        st.caption(f"Loại model: `{selected_item['type'].upper()}`")

        st.markdown("---")
        device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.info(f"💻 Chế độ chạy: {device_name}")

# --- Main Content ---
col1, col2 = st.columns(2)

with col1:
    st.header("📸 Ảnh gốc")
    uploaded_file = st.file_uploader(
        "Upload ảnh (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Original Image", use_container_width=True)

with col2:
    st.header("✨ Kết quả")

    if uploaded_file and st.session_state.selected_model_path:
        if st.button("🚀 Chuyển đổi (Infer)", type="primary"):
            try:
                with st.spinner("Đang xử lý... (Gọi infer.py)"):
                    # 1. Lưu ảnh upload ra file tạm thời (vì infer.py yêu cầu đường dẫn file)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                        image.save(tmp_file.name)
                        tmp_path = tmp_file.name

                    # 2. Gọi hàm infer từ file infer.py
                    # Hàm này trả về Tensor (đã denormalize)
                    output_tensor = infer(
                        image_file=tmp_path,
                        ckpt_file=st.session_state.selected_model_path,
                        image_size=256,  # Bạn có thể đổi thành 512 hoặc thêm slider chỉnh size
                        device=None     # Để None nó sẽ tự detect CUDA/CPU
                    )

                    # 3. Xóa file tạm
                    os.remove(tmp_path)

                    # 4. Chuyển Tensor thành Ảnh để hiển thị
                    # output_tensor có shape [1, 3, H, W], cần squeeze bỏ dimension đầu
                    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))

                    # Hiển thị
                    st.image(output_image, caption="Result Image",
                             use_container_width=True)

                    # Nút download
                    buf = io.BytesIO()
                    output_image.save(buf, format='PNG')
                    st.download_button(
                        label="💾 Tải ảnh về",
                        data=buf.getvalue(),
                        file_name=f"result_{selected_item['filename']}.png",
                        mime="image/png"
                    )

                    st.success("✅ Xử lý hoàn tất!")

            except ValueError as ve:
                st.error(f"Lỗi Model: {str(ve)}")
                st.warning(
                    "Tên file model phải chứa chữ 'animegan' hoặc 'cyclegan' để code nhận diện được loại model.")
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {str(e)}")
                st.exception(e)

    elif not uploaded_file:
        st.info("👈 Vui lòng upload ảnh bên trái.")
    elif not st.session_state.selected_model_path:
        st.warning("👈 Vui lòng chọn model ở sidebar.")
