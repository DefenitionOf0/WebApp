import streamlit as st
import cv2
import numpy as np
from PIL import Image


class ImageProcessor:
    def __init__(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("Ошибка: входное изображение должно быть массивом NumPy.")
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
        gray = cv2.cvtColor(self.filtered_image, cv2.COLOR_BGR2GRAY) if len(self.filtered_image.shape) == 3 else self.filtered_image
        inverted_image = cv2.bitwise_not(gray)
        if adaptive_thresh:
            binary = cv2.adaptiveThreshold(
                inverted_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            _, binary = cv2.threshold(inverted_image, binary_thresh, 255, cv2.THRESH_BINARY)
        self.binary_image = binary
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = [
            (cv2.approxPolyDP(contour, epsilon=tolerance, closed=True) * scaling_factor).astype(np.int32)
            for contour in contours if len(contour) >= 3
        ]
        return self.contours

    def clean_contours(self, area_thresh, perimeter_thresh):
        self.contours = [
            contour for contour in self.contours
            if cv2.contourArea(contour) >= area_thresh and cv2.arcLength(contour, closed=True) >= perimeter_thresh
        ]
        return self.contours

    def draw_contours(self, contours, highlight_index=None):
        result_image = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
        for idx, contour in enumerate(contours):
            color = (0, 255, 0) if idx == highlight_index else (255, 0, 255)
            cv2.polylines(result_image, [contour.astype(np.int32)], isClosed=True, color=color, thickness=2)
        return result_image


# Настройка Streamlit
st.set_page_config(page_title="Интерактивная обработка изображений", layout="wide")
st.title("Интерактивная обработка изображений")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file:
    op_image = Image.open(uploaded_file)
    image = np.array(op_image)
    processor = ImageProcessor(image)

    # Сохраняем текущие данные в session_state
    if "contours" not in st.session_state:
        st.session_state.contours = None

    if "filtered_image" not in st.session_state:
        st.session_state.filtered_image = None

    if "binary_image" not in st.session_state:
        st.session_state.binary_image = None

    if "updated" not in st.session_state:
        st.session_state.updated = True

    # Боковая панель с фильтрами
    blur = st.sidebar.slider("Размытие", 0, 10, 0)
    contrast = st.sidebar.slider("Контрастность", 1.0, 3.0, 1.0)
    median_filter = st.sidebar.slider("Медианный фильтр", 0, 10, 0)
    scaling_factor = st.sidebar.slider("Масштаб контуров", 0.1, 1.0, 1.0)
    tolerance = st.sidebar.slider("Упрощение контуров", 0.1, 10.0, 1.0)
    binary_thresh = st.sidebar.slider("Порог бинаризации", 0, 255, 127)
    adaptive_thresh = st.sidebar.checkbox("Адаптивная бинаризация")
    area_thresh = st.sidebar.slider("Минимальная площадь", 1, 1000, 10)
    perimeter_thresh = st.sidebar.slider("Минимальная длина периметра", 1, 500, 10)

    # Обновляем данные при изменении параметров
    if st.session_state.updated:
        st.session_state.filtered_image = processor.apply_filters(blur, contrast, median_filter)
        processor.filtered_image = st.session_state.filtered_image
        st.session_state.contours = processor.process_image(scaling_factor, tolerance, binary_thresh, adaptive_thresh)
        processor.clean_contours(area_thresh, perimeter_thresh)
        st.session_state.updated = False

    # Отображение фильтрованного изображения
    st.image(st.session_state.filtered_image, caption="Фильтрованное изображение", use_container_width=True)

    # Отображение контуров
    selected_contour = st.slider("Выберите контур", 0, len(st.session_state.contours) - 1, 0)
    result_image = processor.draw_contours(st.session_state.contours, highlight_index=selected_contour)
    st.image(result_image, caption="Контуры", use_container_width=True)

    # Сброс состояния
    if st.button("Обновить данные"):
        st.session_state.updated = True
