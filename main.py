import streamlit as st
import cv2
import numpy as np
from PIL import Image

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
            binary = cv2.adaptiveThreshold(inverted_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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

    def draw_contours(self):
        result_image = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
        for contour in self.contours:
            cv2.polylines(result_image, [contour.astype(np.int32)], isClosed=True, color=(255, 0, 255), thickness=1)
        return result_image


# Интерфейс Streamlit
st.set_page_config(page_title="Обработка изображений", layout="centered")
st.title("Интерактивная обработка изображений")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file:
    op_image = Image.open(uploaded_file)
    image = np.array(op_image)
    processor = ImageProcessor(image)

    st.image(image, caption="Исходное изображение", use_container_width=True)

    # Фильтры
    with st.sidebar:
        st.header("Настройки обработки")
        blur = st.slider("Размытие", 0, 10, 0)
        contrast = st.slider("Контрастность", 1.0, 3.0, 1.0)
        median_filter = st.slider("Медианный фильтр", 0, 10, 0)
        scaling_factor = st.slider("Масштаб контуров", 0.1, 1.0, 1.0)
        tolerance = st.slider("Упрощение контуров", 0.1, 10.0, 1.0)
        binary_thresh = st.slider("Порог бинаризации", 0, 255, 127)
        adaptive_thresh = st.checkbox("Адаптивная бинаризация")
        area_thresh = st.slider("Минимальная площадь", 1, 1000, 10)
        perimeter_thresh = st.slider("Минимальная длина периметра", 1, 500, 10)

    if st.button("Применить фильтры"):
        filtered_image = processor.apply_filters(blur, contrast, median_filter)
        st.image(filtered_image, caption="Изображение после фильтрации", use_container_width=True)

    if st.button("Обработать изображение"):
        processor.apply_filters(blur, contrast, median_filter)
        processor.process_image(scaling_factor, tolerance, binary_thresh, adaptive_thresh)
        processor.clean_contours(area_thresh, perimeter_thresh)

        if processor.contours:
            result_image = processor.draw_contours()
            st.image(processor.binary_image, caption="Бинаризованное изображение", use_container_width=True)
            st.image(result_image, caption="Обработанные контуры", use_container_width=True)
            st.download_button("Скачать обработанное изображение",
                               data=cv2.imencode('.jpg', result_image)[1].tobytes(),
                               file_name="processed_image.jpg",
                               mime="image/jpeg")
        else:
            st.warning("Контуры отсутствуют. Попробуйте изменить параметры.")
