import streamlit as st
import cv2
import numpy as np
from PIL import Image

np.set_printoptions(legacy='1.25')

def apply_filters(image, blur, contrast, median_filter):
    if blur > 0:
        image = cv2.GaussianBlur(image, (blur * 2 + 1, blur * 2 + 1), 0)
    if contrast > 1.0:
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    if median_filter > 0:
        image = cv2.medianBlur(image, median_filter * 2 + 1)
    return image

def process_image(image, scaling_factor, tolerance, binary_thresh, adaptive_thresh):
    if len(image.shape) == 3:  # Если изображение цветное (3 канала)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Если изображение уже в градациях серого (1 канал)
        gray = image

    # Инверсия изображения
    inverted_image = cv2.bitwise_not(gray)

    # Бинаризация
    if adaptive_thresh:
        binary = cv2.adaptiveThreshold(inverted_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    else:
        _, binary = cv2.threshold(inverted_image, binary_thresh, 255, cv2.THRESH_BINARY)

    # Поиск контуров
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Упрощение контуров и масштабирование
    simplified_contours = []
    for contour in contours:
        if len(contour) < 3:
            continue
        approx = cv2.approxPolyDP(contour, epsilon=tolerance, closed=True)
        scaled_contour = (approx * scaling_factor).astype(np.int32)
        simplified_contours.append(scaled_contour)

    return simplified_contours, binary

def clean_contours(contours, area_thresh, perimeter_thresh):
    cleaned_contours = []
    for contour in contours:
        if len(contour) < 3:
            continue
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        if area >= area_thresh and perimeter >= perimeter_thresh:
            cleaned_contours.append(contour)
    return cleaned_contours


def highlight_selected_contour(result_image, contours, selected_index):
    """
    Подсвечивает выбранный пользователем контур на итоговом изображении.
    """
    if 0 <= selected_index < len(contours):
        contour = contours[selected_index].astype(np.int32)
        cv2.drawContours(result_image, [contour], -1, (0, 255, 0), thickness=2)
    return result_image

def apply_filters(image, blur, contrast, median_filter):
    if blur > 0:
        image = cv2.GaussianBlur(image, (blur * 2 + 1, blur * 2 + 1), 0)
    if contrast > 1.0:
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    if median_filter > 0:
        image = cv2.medianBlur(image, median_filter * 2 + 1)
    return image

# Интерфейс Streamlit
st.title("Интерактивная обработка изображений")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Показ изображения
    op_image = Image.open(uploaded_file)
    dpi = op_image.info.get('dpi', (72, 72))
    image = np.array(op_image)
    st.image(image, caption="Исходное изображение", use_container_width=True)

    # Базовые настройки
    st.subheader("Базовые настройки")
    scaling_factor = st.slider("Масштаб контуров", 0.1, 1.0, 1.0)
    tolerance = st.slider("Упрощение контуров", 0.1, 10.0, 1.0)
    binary_thresh = st.slider("Порог бинаризации", 0, 255, 127)
    adaptive_thresh = st.checkbox("Адаптивная бинаризация")

    # Расширенные настройки
    with st.expander("Расширенные фильтры и удаление шума"):
        blur = st.slider("Размытие", 0, 10, 0)
        contrast = st.slider("Контрастность", 1.0, 3.0, 1.0)
        median_filter = st.slider("Медианный фильтр", 0, 10, 0)
        area_thresh = st.slider("Минимальная площадь", 1, 1000, 10)
        perimeter_thresh = st.slider("Минимальная длина периметра", 1, 500, 10)

    # Предварительный просмотр фильтров
    if st.button("Применить фильтры"):
        filtered_image = apply_filters(image, blur, contrast, median_filter)
        st.image(filtered_image, caption="Изображение после фильтрации", use_container_width=True)
    
    # Обработка изображения
    if st.button("Обработать изображение"):
        filtered_image = apply_filters(image, blur, contrast, median_filter)
        contours, binary = process_image(filtered_image, scaling_factor, tolerance, binary_thresh, adaptive_thresh)
        contours = clean_contours(contours, area_thresh, perimeter_thresh)

        # Вывод результатов
        st.image(binary, caption="Бинаризованное изображение", use_container_width=True)
        
        if contours:
            result_image = np.zeroы(image.shape[0], image.shape[1], 3)
            
            for contour in contours:
                contour = contour.astype(np.int32)
                cv2.polylines(result_image, [contour], isClosed=True, color=(255, 0, 255), thickness=1)
            st.image(result_image, caption="Обработанные контуры", use_container_width=True)

            # Сохранение результата
            st.download_button("Скачать обработанное изображение",
                               data=cv2.imencode('.jpg', result_image)[1].tobytes(),
                               file_name="processed_image.jpg",
                               mime="image/jpeg")
        else:
            st.warning("Контуры отсутствуют. Попробуйте изменить параметры.")
