import pytest

from micrograd.backpropagation import Value
from tests.utils import create_values_dict


@pytest.mark.parametrize(
    "a, b, output, grad_a, grad_b",
    [
        (Value(-3, label="a"), Value(2, label="b"), -1, 1, 1),
        (Value(4, label="b"), -3, 1, 1, None),
    ],
)
def test_add(a, b, output, grad_a, grad_b):
    sum = a + b
    assert sum.data == output
    sum.backward()
    assert a.grad == grad_a
    if isinstance(b, Value):
        assert b.grad == grad_b


@pytest.mark.parametrize(
    "a, b, output, grad_a, grad_b",
    [
        (Value(-3, label="a"), Value(2, label="b"), -5, 1, -1),
        (Value(4, label="b"), -3, 7, 1, None),
    ],
)
def test_sub(a, b, output, grad_a, grad_b):
    sub = a - b
    assert sub.data == output
    sub.backward()
    assert a.grad == grad_a
    if isinstance(b, Value):
        assert b.grad == grad_b


@pytest.mark.parametrize(
    "a, b, output, grad_a, grad_b",
    [
        (Value(-3, label="a"), Value(2, label="b"), -6, 2, -3),
        (Value(4, label="b"), -3, -12, -3, None),
    ],
)
def test_mul(a, b, output, grad_a, grad_b):
    mul = a * b
    assert mul.data == output
    mul.backward()
    assert a.grad == grad_a
    if isinstance(b, Value):
        assert b.grad == grad_b


@pytest.mark.parametrize(
    "a, b, output, grad_a, grad_b",
    [
        (Value(-3, label="a"), Value(2, label="b"), -1.5, 0.5, 0.75),
        (Value(4, label="b"), -3, -1.3333333333333333, -0.3333333333333333, None),
    ],
)
def test_div(a, b, output, grad_a, grad_b):
    div = a / b
    assert div.data == output
    div.backward()
    assert a.grad == grad_a
    if isinstance(b, Value):
        assert b.grad == grad_b


@pytest.mark.parametrize(
    "a, b, output, grad_a",
    [
        (Value(-3, label="a"), 2, 9, -6),
        (Value(4, label="b"), 3, 64, 48),
    ],
)
def test_pow(a, b, output, grad_a):
    pow = a**b
    assert pow.data == output
    pow.backward()
    a.grad == grad_a


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


@pytest.mark.parametrize(
    "x, output, grad",
    [
        (Value(data=3, label="x1"), -3, -1),
        (Value(data=-2, label="x2"), 2, -1),
    ],
)
def test_neg(x, output, grad):
    z = -x
    assert z.data == output
    z.backward()
    assert x.grad == grad


@pytest.mark.parametrize(
    "x, output, grad",
    [
        (Value(data=3, label="x1"), 6, 2),
        (Value(data=-2, label="x2"), -4, 2),
    ],
)
def test_rmul(x, output, grad):
    rmul = 2 * x
    assert rmul.data == output
    rmul.backward()
    assert x.grad == grad


@pytest.mark.parametrize(
    "input, expression, output, expected_gradients",
    [
        (
            {
                "x1": -1,
                "x2": 4,
                "x3": 0.5,
            },
            lambda x1, x2, x3: (x1 + x2) * x3 + x1**2,
            2.5,
            {"x1": 0.5 + 2 * -1, "x2": 0.5, "x3": 3},
        ),
        (
            {
                "x1": 0.5,
                "x2": -2,
                "x3": 3,
            },
            lambda x1, x2, x3: (x1 - x2) * x3 + x2**3,
            -0.5,
            {"x1": 3, "x2": -3 + 3 * (-2) ** 2, "x3": 0.5 - (-2)},
        ),
    ],
)
def test_complex_expression(input, expression, output, expected_gradients):
    values = create_values_dict(input)
    expr = expression(values["x1"], values["x2"], values["x3"])
    assert expr.data == output
    expr.backward()
    for var, expected_grad in expected_gradients.items():
        assert values[var].grad == expected_grad
