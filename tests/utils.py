from micrograd.backpropagation import Value


def create_values_dict(inputs):
    return {key: Value(data=value, label=key) for key, value in inputs.items()}
