import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


class ImageProcessor:
    def __init__(self, image):
        self.image = image
        self.filtered_image = None
        self.binary_image = None
        self.contours = None

    def apply_filters(self, blur, contrast, median_filter):
        img = self.image.copy()
        if blur > 0:
            img = cv2.GaussianBlur(img, (blur * 2 + 1, blur * 2 + 1), 0)
        if contrast > 1.0:
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
        if median_filter > 0:
            img = cv2.medianBlur(img, median_filter * 2 + 1)
        self.filtered_image = img
        return img

    def process_image(self, scaling_factor, tolerance, binary_thresh, adaptive_thresh):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        inverted_image = cv2.bitwise_not(gray)
        if adaptive_thresh:
            binary = cv2.adaptiveThreshold(
                inverted_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            _, binary = cv2.threshold(inverted_image, binary_thresh, 255, cv2.THRESH_BINARY)
        self.binary_image = binary

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        simplified_contours = [
            (cv2.approxPolyDP(contour, epsilon=tolerance, closed=True) * scaling_factor).astype(np.int32)
            for contour in contours if len(contour) >= 3
        ]
        self.contours = simplified_contours
        return simplified_contours

    def clean_contours(self, area_thresh, perimeter_thresh):
        self.contours = [
            contour for contour in self.contours
            if cv2.contourArea(contour) >= area_thresh and cv2.arcLength(contour, closed=True) >= perimeter_thresh
        ]
        return self.contours

    def draw_contours(self, highlight_index=None):
        result_image = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
        for idx, contour in enumerate(self.contours):
            color = (0, 255, 0) if idx == highlight_index else (255, 0, 255)
            cv2.polylines(result_image, [contour.astype(np.int32)], isClosed=True, color=color, thickness=2)
        return result_image


# Утилита для изменения размера изображений
def resize_image(image, max_size=1024):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scaling_factor = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scaling_factor), int(h * scaling_factor)))
    return image


# Настройка страницы Streamlit
st.set_page_config(page_title="Обработка изображений", layout="wide")
st.title("Интерактивная обработка изображений")

# Загрузка изображения
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Открытие изображения
    op_image = Image.open(uploaded_file)
    image = np.array(op_image)
    image = resize_image(image)
    processor = ImageProcessor(image)

    # Отображение исходного изображения
    st.image(image, caption="Исходное изображение", use_container_width=True)

    # Настройки в боковой панели
    with st.sidebar:
        st.header("Настройки")
        blur = st.slider("Размытие", 0, 10, 0)
        contrast = st.slider("Контрастность", 1.0, 3.0, 1.0)
        median_filter = st.slider("Медианный фильтр", 0, 10, 0)
        scaling_factor = st.slider("Масштаб контуров", 0.1, 1.0, 1.0)
        tolerance = st.slider("Упрощение контуров", 0.1, 10.0, 1.0)
        binary_thresh = st.slider("Порог бинаризации", 0, 255, 127)
        adaptive_thresh = st.checkbox("Адаптивная бинаризация")
        area_thresh = st.slider("Минимальная площадь", 1, 1000, 10)
        perimeter_thresh = st.slider("Минимальная длина периметра", 1, 500, 10)

    # Применение фильтров
    filtered_image = processor.apply_filters(blur, contrast, median_filter)
    st.image(filtered_image, caption="Изображение после фильтрации", use_container_width=True, key="filtered")

    # Обработка изображения
    processor.process_image(scaling_factor, tolerance, binary_thresh, adaptive_thresh)
    processor.clean_contours(area_thresh, perimeter_thresh)

    # Интерактивный выбор контура
    contour_indices = list(range(len(processor.contours)))
    selected_contour = st.selectbox("Выберите контур для подсветки:", options=contour_indices, index=0)

    # Подсветка выбранного контура
    result_image = processor.draw_contours(highlight_index=selected_contour)
    st.image(result_image, caption="Обработанные контуры", use_container_width=True, key="result")

    # Сохранение результата
    buffer = BytesIO()
    is_success, encoded_image = cv2.imencode('.jpg', result_image)
    if is_success:
        buffer.write(encoded_image.tobytes())
        st.download_button(
            "Скачать обработанное изображение",
            data=buffer,
            file_name="processed_image.jpg",
            mime="image/jpeg"
        )
