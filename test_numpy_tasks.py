
import numpy as np
import pytest

from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixStatistics,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
    TimeSeriesStatistics,
)

from numpy_tasks import (
    sum_prod,
    binarize,
    unique_rows,
    unique_columns,
    matrix_statistics,
    chess,
    draw_rectangle,
    draw_ellipse,
    analyze_time_series,
    one_hot,
)


def test_sum_prod():
    matrices = np.array([
        [[1, 0], [0, 1]],
        [[2, 0], [0, 2]],
    ])
    vectors = np.array([
        [[3], [4]],
        [[3], [4]],
    ])

    result = sum_prod(
        MatrixVectorBatchInput(matrices=matrices, vectors=vectors)
    )

    np.testing.assert_array_equal(result, [[9], [12]])


def test_binarize():
    matrix = np.array([[1, 5], [3, 8]])

    result = binarize(
        BinarizeInput(matrix=matrix, threshold=3)
    )

    np.testing.assert_array_equal(result, [[0, 1], [0, 1]])

    # Элемент, равный порогу, должен стать нулём.
    result = binarize(
        BinarizeInput(matrix=np.array([[3]]), threshold=3)
    )
    np.testing.assert_array_equal(result, [[0]])


def test_unique_rows():
    matrix = np.array([
        [1, 2, 1, 3, 2],
        [4, 4, 4, 4, 4],
        [5, 6, 7, 8, 9],
    ])

    result = unique_rows(MatrixInput(matrix=matrix))

    assert result == [
        [1, 2, 3],
        [4],
        [5, 6, 7, 8, 9],
    ]


def test_unique_columns():
    matrix = np.array([
        [1, 2, 1],
        [3, 2, 4],
        [1, 5, 1],
    ])

    result = unique_columns(MatrixInput(matrix=matrix))

    assert result == [
        [1, 3],
        [2, 5],
        [1, 4],
    ]


def test_matrix_statistics():
    data = RandomMatrixInput(
        rows=3,
        columns=4,
        mean=10,
        std=2,
        seed=42,
    )

    result = matrix_statistics(data)

    assert result.matrix.shape == (3, 4)
    assert result.row_means.shape == (3,)
    assert result.column_means.shape == (4,)
    assert result.row_variances.shape == (3,)
    assert result.column_variances.shape == (4,)

    # Повторный запуск с тем же seed даёт ту же матрицу.
    result2 = matrix_statistics(data)
    np.testing.assert_array_equal(result.matrix, result2.matrix)

    np.testing.assert_allclose(
        result.row_means, result.matrix.mean(axis=1)
    )
    np.testing.assert_allclose(
        result.column_means, result.matrix.mean(axis=0)
    )
    np.testing.assert_allclose(
        result.row_variances, result.matrix.var(axis=1)
    )
    np.testing.assert_allclose(
        result.column_variances, result.matrix.var(axis=0)
    )


def test_chess():
    result = chess(
        ChessInput(rows=3, columns=4, first=0, second=1)
    )

    expected = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ])

    np.testing.assert_array_equal(result, expected)

    # Матрица 1 на 1 содержит только первый элемент.
    result = chess(
        ChessInput(rows=1, columns=1, first=7, second=9)
    )
    np.testing.assert_array_equal(result, [[7]])


def test_draw_rectangle():
    result = draw_rectangle(
        RectangleInput(
            width=2,
            height=2,
            image_height=5,
            image_width=5,
            shape_color=(255, 0, 0),
            background_color=(0, 0, 0),
        )
    )

    assert result.shape == (5, 5, 3)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result[2, 2], [255, 0, 0])
    np.testing.assert_array_equal(result[0, 0], [0, 0, 0])


def test_draw_ellipse():
    result = draw_ellipse(
        EllipseInput(
            semi_axis_x=2,
            semi_axis_y=1,
            image_height=5,
            image_width=5,
            shape_color=(255, 0, 0),
            background_color=(0, 0, 0),
        )
    )

    assert result.shape == (5, 5, 3)
    assert result.dtype == np.uint8

    # Центр эллипса закрашен.
    np.testing.assert_array_equal(result[2, 2], [255, 0, 0])

    # Угол изображения вне эллипса.
    np.testing.assert_array_equal(result[0, 0], [0, 0, 0])


def test_analyze_time_series():
    result = analyze_time_series(
        TimeSeriesInput(
            values=[1, 3, 1, 5, 2],
            window=3,
        )
    )

    assert result.mean == pytest.approx(2.4)
    assert result.variance == pytest.approx(2.24)
    assert result.std == pytest.approx(np.sqrt(2.24))
    assert result.local_maxima == [1, 3]
    assert result.local_minima == [2]

    np.testing.assert_allclose(
        result.moving_average,
        [5 / 3, 3, 8 / 3],
    )


def test_analyze_time_series_without_extrema():
    result = analyze_time_series(
        TimeSeriesInput(
            values=[1, 2, 3, 4],
            window=2,
        )
    )

    assert result.local_maxima == []
    assert result.local_minima == []
    np.testing.assert_allclose(result.moving_average, [1.5, 2.5, 3.5])


def test_one_hot():
    result = one_hot(
        OneHotInput(labels=[0, 2, 3, 0], class_count=4)
    )

    expected = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])

    np.testing.assert_array_equal(result, expected)


def test_one_hot_without_class_count():
    result = one_hot(
        OneHotInput(labels=[0, 2, 1], class_count=None)
    )

    expected = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])

    np.testing.assert_array_equal(result, expected)
