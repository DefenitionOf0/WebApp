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
        """Применяет фильтры размытия, контрастности и медианного фильтра к изображению."""
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
        """Рассчитывает контуры на основе обработанного изображения."""
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
        # Создаём словарь контуров с уникальными ID
        contour_dict = {i: (cv2.approxPolyDP(contour, epsilon=tolerance, closed=True) * scaling_factor).astype(np.int32)
                        for i, contour in enumerate(contours) if len(contour) >= 3}
        return contour_dict

    def filter_contours(self, contours, area_thresh, perimeter_thresh, removed_ids):
        """Фильтрует контуры по площади, длине периметра и удалённым ID."""
        filtered_dict = {id_: contour for id_, contour in contours.items()
                         if id_ not in removed_ids and
                         cv2.contourArea(contour) >= area_thresh and
                         cv2.arcLength(contour, closed=True) >= perimeter_thresh}
        return filtered_dict

    def draw_contours(self, contours, highlight_id=None):
        """Рисует контуры на пустом изображении."""
        result_image = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
        for id_, contour in contours.items():
            color = (0, 255, 0) if id_ == highlight_id else (255, 0, 255)
            cv2.polylines(result_image, [contour.astype(np.int32)], isClosed=True, color=color, thickness=2)
        return result_image

    def export_to_mpf(self, contours):
        """Экспортирует контуры в формате G-code."""
        gcode = []
        gcode.append("BEGIN PGM G-CODE EXPORT\n")
        for contour in contours.values():
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


# Функция для сброса состояния
def reset_state():
    """Сбрасывает состояние контуров и обработанных данных."""
    st.session_state.primary_contours = None  # Все пересчитанные контуры с ID
    st.session_state.secondary_contours = None  # Отфильтрованные контуры (с учётом удалений)
    st.session_state.removed_contour_ids = set()  # ID удалённых контуров
    st.session_state.filtered_image = None
    st.session_state.current_contour_id = None  # ID текущего выделенного контура


# Загрузка изображения
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Сбрасываем состояние при загрузке нового изображения
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file:
        reset_state()  # Сбрасываем состояние
        st.session_state.last_uploaded_file = uploaded_file

    # Обрабатываем изображение
    op_image = Image.open(uploaded_file)
    image = np.array(op_image)
    processor = ImageProcessor(image)

    # Боковая панель с фильтрами
    #blur = st.sidebar.slider("Размытие", 0, 10, 0)
    #contrast = st.sidebar.slider("Контрастность", 1.0, 3.0, 1.0)
    #median_filter = st.sidebar.slider("Медианный фильтр", 0, 10, 0)
    scaling_factor = st.sidebar.slider("Масштаб контуров", 0.1, 1.0, 1.0)
    tolerance = st.sidebar.slider("Упрощение контуров", 0.1, 10.0, 1.0)
    binary_thresh = st.sidebar.slider("Порог бинаризации", 0, 255, 127)
    adaptive_thresh = st.sidebar.checkbox("Адаптивная бинаризация")
    area_thresh = st.sidebar.slider("Минимальная площадь", 1, 1000, 10)
    perimeter_thresh = st.sidebar.slider("Минимальная длина периметра", 1, 500, 10)

    # Применение фильтров к изображению
    ##st.session_state.filtered_image = processor.apply_filters(blur, contrast, median_filter)
    ##processor.filtered_image = st.session_state.filtered_image

    # Пересчёт контуров
    st.session_state.primary_contours = processor.process_image(
        scaling_factor, tolerance, binary_thresh, adaptive_thresh
    )

    # Применение фильтров площади и периметра с учётом удалённых контуров
    st.session_state.secondary_contours = processor.filter_contours(
        st.session_state.primary_contours, area_thresh, perimeter_thresh, st.session_state.removed_contour_ids
    )

    # Выбор текущего контура
    selected_contour_id = st.sidebar.selectbox(
        "Выберите контур:",
        options=list(st.session_state.secondary_contours.keys()),
        format_func=lambda id_: f"Контур {id_ + 1}"  # Для наглядности
    )
    st.session_state.current_contour_id = selected_contour_id

    # Удаление текущего контура
    if st.sidebar.button("Удалить выбранный контур"):
        if st.session_state.current_contour_id in st.session_state.secondary_contours:
            # Добавляем ID контура в список удалённых
            st.session_state.removed_contour_ids.add(st.session_state.current_contour_id)
            st.success(f"Контур {st.session_state.current_contour_id + 1} удалён.")

    # Отображение изображений
    st.image(st.session_state.filtered_image, caption="Фильтрованное изображение", use_container_width=True)

    # Отображение контуров
    result_image = processor.draw_contours(st.session_state.secondary_contours, highlight_id=st.session_state.current_contour_id)
    st.image(result_image, caption="Контуры", use_container_width=True)

    # Экспорт G-code
    if st.button("Экспортировать в G-code (.MPF)"):
        gcode_data = processor.export_to_mpf(st.session_state.secondary_contours)
        st.download_button(
            label="Скачать G-code",
            data=gcode_data,
            file_name="contours.mpf",
            mime="text/plain"
        )
