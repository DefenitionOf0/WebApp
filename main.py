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

    def filter_contours(self, contours, area_thresh, perimeter_thresh):
        """Фильтрует контуры по площади и длине периметра."""
        return [
            contour for contour in contours
            if cv2.contourArea(contour) >= area_thresh and cv2.arcLength(contour, closed=True) >= perimeter_thresh
        ]

    def draw_contours(self, contours, highlight_index=None):
        result_image = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
        for idx, contour in enumerate(contours):
            color = (0, 255, 0) if idx == highlight_index else (255, 0, 255)
            cv2.polylines(result_image, [contour.astype(np.int32)], isClosed=True, color=color, thickness=2)
        return result_image

    def export_to_mpf(self, contours):
        gcode = []
        gcode.append("BEGIN PGM G-CODE EXPORT\n")
        for contour in contours:
            gcode.append("G0 Z10.0\n")  # Подъем инструмента
            start_point = contour[0][0]
            gcode.append(f"G0 X{start_point[0]} Y{start_point[1]}\n")  # Переход к стартовой точке
            gcode.append("G1 Z-1.0\n")  # Опускание инструмента
            for point in contour:
                x, y = point[0]
                gcode.append(f"G1 X{x} Y{y}\n")
            gcode.append("G0 Z10.0\n")  # Подъем инструмента после контура
        gcode.append("END PGM\n")
        return "\n".join(gcode)


# Настройка Streamlit
st.set_page_config(page_title="Интерактивная обработка изображений", layout="wide")
st.title("Интерактивная обработка изображений")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file:
    op_image = Image.open(uploaded_file)
    image = np.array(op_image)
    processor = ImageProcessor(image)

    # Сохранение состояния
    if "all_original_contours" not in st.session_state:
        st.session_state.all_original_contours = None  # Список всех контуров
    if "filtered_contours" not in st.session_state:
        st.session_state.filtered_contours = None  # Отфильтрованные контуры
    if "filtered_image" not in st.session_state:
        st.session_state.filtered_image = None  # Обработанное изображение
    if "current_contour" not in st.session_state:
        st.session_state.current_contour = 0  # Индекс выбранного контура

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

    # Применение фильтров и пересчёт контуров
    st.session_state.filtered_image = processor.apply_filters(blur, contrast, median_filter)
    processor.filtered_image = st.session_state.filtered_image

    if st.session_state.all_original_contours is None:
        st.session_state.all_original_contours = processor.process_image(
            scaling_factor, tolerance, binary_thresh, adaptive_thresh
        )

    # Применение фильтров минимальной площади и периметра
    st.session_state.filtered_contours = processor.filter_contours(
        st.session_state.all_original_contours, area_thresh, perimeter_thresh
    )

    # Выбор текущего контура
    selected_contour_idx = st.sidebar.selectbox(
        "Выберите контур:",
        options=range(len(st.session_state.filtered_contours)) if st.session_state.filtered_contours else [],
        format_func=lambda idx: f"Контур {idx + 1}",
        index=st.session_state.current_contour if st.session_state.filtered_contours else 0
    )
    st.session_state.current_contour = selected_contour_idx

    # Удаление текущего контура
    if st.sidebar.button("Удалить выбранный контур"):
        if st.session_state.filtered_contours:
            contour_to_delete = st.session_state.filtered_contours[st.session_state.current_contour]
            st.session_state.all_original_contours = [
                contour for contour in st.session_state.all_original_contours
                if not np.array_equal(contour, contour_to_delete)
            ]
            st.session_state.filtered_contours = [
                contour for contour in st.session_state.filtered_contours
                if not np.array_equal(contour, contour_to_delete)
            ]
            st.session_state.current_contour = 0
            st.success("Выбранный контур удалён.")

    # Отображение изображений
    st.image(st.session_state.filtered_image, caption="Фильтрованное изображение", use_container_width=True)

    # Отображение контуров
    selected_contour = st.session_state.current_contour
    result_image = processor.draw_contours(st.session_state.filtered_contours, highlight_index=selected_contour)
    st.image(result_image, caption="Контуры", use_container_width=True)

    # Экспорт G-code
    if st.button("Экспортировать в G-code (.MPF)"):
        gcode_data = processor.export_to_mpf(st.session_state.filtered_contours)
        st.download_button(
            label="Скачать G-code",
            data=gcode_data,
            file_name="contours.mpf",
            mime="text/plain"
        )
