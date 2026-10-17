
"""Заготовки задач на NumPy."""

import numpy as np
import matplotlib.pyplot as plt

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    return np.sum(matrices @ vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [
        list(dict.fromkeys(row.tolist()))
        for row in matrix
    ]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [
        list(dict.fromkeys(matrix[:, j].tolist()))
        for j in range(matrix.shape[1])
    ]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns = data.rows, data.columns
    mean, std, seed = data.mean, data.std, data.seed

    rng = np.random.default_rng(seed)
    matrix = rng.normal(mean, std, size=(rows, columns))

    return MatrixStatistics(
        matrix,
        np.mean(matrix, axis=1),
        np.mean(matrix, axis=0),
        np.var(matrix, axis=1),
        np.var(matrix, axis=0),
    )


def plot_matrix_histograms(matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix)

    for i, row in enumerate(matrix):
        plt.figure()
        plt.hist(row)
        plt.title(f"Строка {i}")
        plt.xlabel("Значение")
        plt.ylabel("Частота")
        plt.show()

    for j in range(matrix.shape[1]):
        plt.figure()
        plt.hist(matrix[:, j])
        plt.title(f"Столбец {j}")
        plt.xlabel("Значение")
        plt.ylabel("Частота")
        plt.show()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns = data.rows, data.columns
    first, second = data.first, data.second

    pattern = (np.indices((rows, columns)).sum(axis=0) % 2 == 0)
    return np.where(pattern, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color

    y0, x0 = image_height // 2, image_width // 2

    y_start = y0 - height // 2
    x_start = x0 - width // 2

    y_start = max(0, y_start)
    x_start = max(0, x_start)
    y_end = min(image_height, y_start + height)
    x_end = min(image_width, x_start + width)

    image[y_start:y_end, x_start:x_end] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color

    y0, x0 = image_height // 2, image_width // 2
    y, x = np.ogrid[:image_height, :image_width]

    inside = (
        ((x - x0) / semi_axis_x) ** 2
        + ((y - y0) / semi_axis_y) ** 2
        <= 1
    )

    image[inside] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)

    mean = np.mean(values)
    variance = np.var(values)
    std = np.std(values)

    local_maxima = np.where(
        (values[1:-1] > values[:-2])
        & (values[1:-1] > values[2:])
    )[0] + 1

    local_minima = np.where(
        (values[1:-1] < values[:-2])
        & (values[1:-1] < values[2:])
    )[0] + 1

    moving_average = np.convolve(
        values, np.ones(window) / window, mode="valid"
    )

    return TimeSeriesStatistics(
        mean,
        variance,
        std,
        local_maxima.tolist(),
        local_minima.tolist(),
        moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)

    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0

    result = np.zeros((len(labels), class_count), dtype=int)
    result[np.arange(len(labels)), labels] = 1
    return result
