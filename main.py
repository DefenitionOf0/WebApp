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
