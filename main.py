from operator import index

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
        contour_dict = {i: (cv2.approxPolyDP(contour, epsilon=tolerance, closed=True) * scaling_factor).astype(np.float32)
                        for i, contour in enumerate(contours) if len(contour) >= 3}
        #print(contour_dict)
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
        if len(highlight_id) is 0:
            highlight_id = [0]

        result_image = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
        #("yep {}".format(highlight_id))

        for id_, contour in contours.items():
            #(id_, contour)
            color = (255, 0, 255)
            cv2.polylines(result_image, [contour.astype(np.int32)], isClosed=True, color=color, thickness=1)
            for itr in highlight_id:
                if np.array_equal(contour, contours[itr]):
                    color = (0, 255, 0)
                    cv2.polylines(result_image, [contour.astype(np.int32)], isClosed=True, color=color, thickness=1)
                    #("if {}{}".format(itr, id_))

        return result_image

    def export_to_mpf(self, contours):
        """Центрирование контура в 0"""
        # Смещение в центр
        height, width = self.binary_image.shape
        center_x, center_y = width // 2, height // 2

        transformed_contours = []
        for contour in contours.values():
            # Перемещаем контур в центр
            contour = np.vstack([contour, contour[0:1]])  # Используем contour[0:1] для сохранения формы (1, 1, 2)
            centered_contour = contour - [center_x * scaling_factor, center_y * scaling_factor]

            # Инвертируем по осям
            inverted_contour = centered_contour * [1, -1]  # Инверсия по обеим осям
            transformed_contours.append(inverted_contour)



        """Экспортирует контуры в формате G-code."""
        gcode = []
        gcode.append(";BEGIN PGM G-CODE EXPORT\n;GENERATED IN CONTOUR TOOL\n;CREATED BY IVAN S. OMP-8\n")
        gcode.append("WORKPIECE(,\"\",,\"CYLINDER\",0,0,-16,-80,80)\n")
        gcode.append("G54 G17\n")
        gcode.append("T=\"CENTERDRILL\"\n")
        gcode.append("M6\n")
        gcode.append("S10000 M3 M8\n\n")
        #gcode.append("G0 Z3\n")  # Подъем инструмента
        for contour in transformed_contours:
            start_point = contour[0][0]
            gcode.append(f"G0 X{start_point[0]} Y{start_point[1]}\n")  # Переход к стартовой точке
            try:
                print(gcode[8])
            except:
                gcode.append("G0 Z1\n")

            gcode.append("G1 F10 Z-0.1\n")  # Опускание инструмента
            gcode.append("G1 F50\n")
            for point in contour:
                x, y = point[0]
                gcode.append(f"X{x} Y{y}\n")
            gcode.append("G0 Z1\n")  # Подъем инструмента после контура

        gcode.append("G75 Z1\nG75 X1 Y1\nM5 M9 M30\nEND PGM\n")
        return "".join(gcode)


# Настройка Streamlit
st.set_page_config(page_title="Интерактивная обработка изображений", layout="wide")
st.title("Интерактивная обработка изображений")


def on_clk():
    # Добавляем ID контура в список удалённых
    st.session_state.filters_locked = True  # Блокируем изменение фильтров
    for current in st.session_state.current_contour_id:
        st.session_state.removed_contour_ids.add(current)
    for selected in st.session_state.selected_contour_id:
        del st.session_state.secondary_contours[selected]

# Функция для сброса состояния
def reset_state():
    """Сбрасывает состояние контуров и обработанных данных."""
    st.session_state.primary_contours = None  # Все пересчитанные контуры с ID
    st.session_state.secondary_contours = None  # Отфильтрованные контуры (с учётом удалений)
    st.session_state.removed_contour_ids = set()  # ID удалённых контуров
    st.session_state.filtered_image = None
    st.session_state.filters_locked = False  # Разрешает применение фильтров
    st.session_state.current_contour_id = None  # ID текущего выделенного контура
    st.session_state.selected_contour_id = None



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
    blur = st.sidebar.slider("Размытие", 0, 10, 0, disabled=st.session_state.filters_locked)
    contrast = st.sidebar.slider("Контрастность", 1.0, 3.0, 1.0, disabled=st.session_state.filters_locked)
    median_filter = st.sidebar.slider("Медианный фильтр", 0, 10, 0, disabled=st.session_state.filters_locked)
    scaling_factor = st.sidebar.slider("Масштаб контуров", 0.01, 1.0, 1.0)
    tolerance = st.sidebar.slider("Упрощение контуров", 0.1, 10.0, 1.0)
    binary_thresh = st.sidebar.slider("Порог бинаризации", 0, 255, 127)
    adaptive_thresh = st.sidebar.checkbox("Адаптивная бинаризация")
    area_thresh = st.sidebar.slider("Минимальная площадь", 0, 1000, 0)
    perimeter_thresh = st.sidebar.slider("Минимальная длина периметра", 0, 500, 0)

    # Применение фильтров к изображению
    st.session_state.filtered_image = processor.apply_filters(blur, contrast, median_filter)
    processor.filtered_image = st.session_state.filtered_image

    # Пересчёт контуров
    st.session_state.primary_contours = processor.process_image(
        scaling_factor, tolerance, binary_thresh, adaptive_thresh
    )

    # Применение фильтров площади и периметра с учётом удалённых контуров
    st.session_state.secondary_contours = processor.filter_contours(
        st.session_state.primary_contours, area_thresh, perimeter_thresh, st.session_state.removed_contour_ids
    )
    # Удаление текущего контура
    if st.sidebar.button("Удалить выбранный контур", key=123, use_container_width=True):
        on_clk()
    #print(st.session_state.selected_contour_id)
    #print(st.session_state.current_contour_id)
    # Выбор текущего контура
    #st.session_state.selected_contour_id = next(iter(st.session_state.secondary_contours))
    st.session_state.selected_contour_id = st.sidebar.multiselect(
        "Выберите контур:",
        options=list(st.session_state.secondary_contours.keys()),
        default=next(iter(list(st.session_state.secondary_contours.keys()))),
        key = 222,
        on_change = None
    )
    st.session_state.current_contour_id = st.session_state.selected_contour_id

    # Отображение изображений
    st.image(st.session_state.filtered_image, caption="Фильтрованное изображение", use_container_width=True)

    # Отображение контуров
    result_image = processor.draw_contours(st.session_state.secondary_contours,
                                           highlight_id=st.session_state.current_contour_id)

    st.image(result_image, caption="Контуры", use_container_width=True)






        #st.session_state.current_contour_id = next(iter(st.session_state.secondary_contours))
        #st.session_state.selected_contour_id = next(iter(st.session_state.secondary_contours))



    # Экспорт G-code
    if st.button("Экспортировать в G-code (.MPF)"):
        gcode_data = processor.export_to_mpf(st.session_state.secondary_contours)
        st.download_button(
            label="Скачать G-code",
            data=gcode_data,
            file_name="contours.mpf",
            mime="text/plain"
        )

