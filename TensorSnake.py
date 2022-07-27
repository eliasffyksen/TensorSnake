import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
from timeit import timeit

class TensorSnake(nn.Module):
  board_size: int

  def __init__(self, board_size, games):
    super().__init__()
    self.board_size = board_size

    self.register_buffer(
      'pos_prev',
      t.tensor([[board_size // 2 - 1] * 2])
        .repeat((games, 1))
    )
    self.register_buffer('pos_cur', self.pos_prev + t.tensor([0, 1]))
    self.register_buffer('idxs', t.arange(games, dtype=t.long))

    self.register_buffer('state', t.zeros((games, board_size, board_size), dtype=t.int8))
    self.state[self.idxs, self.pos_prev[:,0], self.pos_prev[:,1]] = 1
    self.state[self.idxs, self.pos_cur[:,0],  self.pos_cur[:,1] ] = 2
    food = (self.state == 0).flatten(1).to(t.float).multinomial(1).squeeze()
    food_x = t.div(food, board_size, rounding_mode='trunc')
    food_y = food % board_size
    self.state[self.idxs, food_x, food_y] = -1

  def forward(self, action) -> t.Tensor:
    pos_next = self.pos_cur - self.pos_prev

    action_left = (action == 0)
    pos_next[action_left] = pos_next[action_left].flip((1))
    pos_next[action_left, 0] = -pos_next[action_left,0]

    action_right = (action == 2)
    pos_next[action_right] = pos_next[action_right].flip((1))
    pos_next[action_right, 1] = -pos_next[action_right,1]

    pos_next += self.pos_cur

    outside = (pos_next < 0) | (pos_next >= self.board_size)
    outside = t.any(outside, 1)
    pos_next = pos_next.clamp(0, self.board_size - 1)
    dead = outside | (self.state[
      self.idxs,
      pos_next[:,0],
      pos_next[:,1]
    ] > 0)

    feeding = self.state[self.idxs, pos_next[:,0], pos_next[:,1]] == -1

    self.state[dead] = 0
    pos = t.tensor([self.board_size // 2] * 2)
    pos[1] -= 2
    self.pos_prev[dead] = pos
    self.state[dead,pos[0],pos[1]] = 1
    pos[1] += 1
    self.pos_cur[dead] = pos
    self.state[dead,pos[0],pos[1]] = 2
    pos[1] += 1
    pos_next[dead] = pos
    self.state[dead,pos[0],pos[1]] = 1

    spawn_food = dead | feeding
    food = (self.state[spawn_food] == 0).flatten(1).to(t.float).multinomial(1).squeeze()
    food_x = t.div(food, self.board_size, rounding_mode='trunc')
    food_y = food % self.board_size
    self.state[spawn_food, food_x, food_y] = -1

    dec_mask = self.state > 0
    dec_mask[feeding] = False
    self.state[dec_mask] -= 1
    self.state[self.idxs, pos_next[:,0], pos_next[:,1]] = \
      self.state[self.idxs, self.pos_cur[:,0], self.pos_cur[:,1]] + 1

    self.pos_prev = self.pos_cur
    self.pos_cur = pos_next

    return self.state

snake_count = 2 ** 22
ts = TensorSnake(8, snake_count)
ts_jit = t.jit.script(ts.eval())
ts_jit = t.jit.optimize_for_inference(ts_jit)


for _ in range(20):
  actions = t.randint(0,2,(snake_count,))
  print(timeit(lambda: ts_jit(actions), number=1))
