from TensorSnake import TensorSnake
import torch as t
from timeit import timeit

print('building')
snake_count = 35 * 10 ** 6
ts = TensorSnake(8, float_type=t.float64)
ts = t.jit.script(ts)

print('init...')
ts.init(snake_count)
print('running')

for _ in range(5):
  actions = t.randint(0,2,(snake_count,))
  print(timeit(lambda: ts(actions), number=1))
