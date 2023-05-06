from src.datasets import *  # noqa: F403
from src.eval.main import eval_dataset

print("Pokec N")
print(pokec_n[0])  # noqa: F405
print(eval_dataset(pokec_n[0]))  # noqa: F405
print()

print("Pokec Z")
print(pokec_z[0])  # noqa: F405
print(eval_dataset(pokec_z[0]))  # noqa: F405
print()

print("German")
print(german[0])  # noqa: F405
print(eval_dataset(german[0]))  # noqa: F405
print()

print("Credit")
print(credit[0])  # noqa: F405
print(eval_dataset(credit[0]))  # noqa: F405
print()

print("Recidivism")
print(bail[0])  # noqa: F405
print(eval_dataset(bail[0]))  # noqa: F405
print()
