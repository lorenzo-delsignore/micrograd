import pytest

from micrograd.backpropagation import Value


@pytest.mark.parametrize(
    "input, output, grad",
    [
        (Value(data=2, label="x1"), 2, 1),
        (Value(data=-2.0, label="x2"), 0, 0),
        (Value(data=1.3, label="x3"), 1.3, 1),
    ],
)
def test_relu(input, output, grad):
    relu = input.relu()
    assert relu.data == output
    relu.backward()
    assert input.grad == grad


@pytest.mark.parametrize(
    "input, output, grad",
    [
        (Value(data=2, label="x1"), 0.9640275800758169, 0.07065082485316443),
        (Value(data=-2.0, label="x2"), -0.9640275800758168, 0.07065082485316465),
        (Value(data=1.3, label="x3"), 0.8617231593133063, 0.25743319670309406),
    ],
)
def test_tanh(input, output, grad):
    tanh = input.tanh()
    assert tanh.data == output
    tanh.backward()
    assert input.grad == grad


@pytest.mark.parametrize(
    "input, output, grad",
    [
        (Value(data=2, label="x1"), 7.38905609893065, 7.38905609893065),
        (Value(data=-2.0, label="x2"), 0.1353352832366127, 0.1353352832366127),
        (Value(data=1.3, label="x3"), 3.6692966676192444, 3.6692966676192444),
    ],
)
def test_exp(input, output, grad):
    exp = input.exp()
    assert exp.data == output
    exp.backward()
    assert input.grad == grad
