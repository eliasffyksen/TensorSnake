import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
from timeit import timeit

class TensorSnake(nn.Module):
  board_size: int
  float_type: t.dtype

  def __init__(self, board_size, device='cpu', float_type=t.float):
    super().__init__()
    self.board_size = board_size
    self.float_type = t.float
    self.register_buffer('pos_cur', t.tensor(0))
    self.register_buffer('pos_last', t.tensor(0))
    self.register_buffer('pos_food', t.tensor(0))
    self.register_buffer('idxs', t.tensor(0))
    self.register_buffer('state', t.tensor(0))
    self.register_buffer('dirs', t.tensor([
      [ 0,  0],
      [ 0,  0],
      [-1,  0],
      [ 0,  1],
      [ 1,  0],
      [ 0, -1],
    ]))
    self.to(device=device)


  @t.jit.export
  def init(self, games: int):
    device = self.state.device
    self.pos_last = t.tensor(
      [0,0],
      device=device
    ).repeat((games, 1))

    self.pos_cur = t.tensor(
      [0,1],
      device=device
    ).repeat((games, 1))

    self.pos_food = t.tensor(
      [0,2],
      device=device
    ).repeat((games, 1))

    self.idxs = t.arange(games, dtype=t.long, device=device)

    self.state = t.zeros((games, self.board_size, self.board_size), dtype=t.uint8, device=device)
    self.state[:,0,0] = 3
    self.state[:,0,1] = 3
    self.state[:,0,2] = 1

  def forward(self, action) -> t.Tensor:
    device=self.state.device

    action_dirs = self.state[
      self.idxs,
      self.pos_cur[:,0],
      self.pos_cur[:,1]
    ]
    action_dirs[action == 0] -= 1
    action_dirs[action == 2] += 1
    action_dirs[action_dirs == 1] = 5
    action_dirs[action_dirs == 6] = 2
    self.state[
      self.idxs,
      self.pos_cur[:,0],
      self.pos_cur[:,1]
    ] = action_dirs

    pos_next = self.pos_cur + self.dirs[action_dirs.to(t.int64)]
    pos_next %= t.tensor(self.board_size, device=device)
    self.pos_cur = pos_next

    dead = self.state[
      self.idxs,
      pos_next[:,0],
      pos_next[:,1]
    ] > 1

    feeding = (pos_next == self.pos_food).all(1)

    pos_next_last = self.pos_last[~feeding] + self.dirs[
      self.state[
        ~feeding,
        self.pos_last[~feeding,0],
        self.pos_last[~feeding,1]
      ].to(t.int64)
    ]
    self.state[
      ~feeding,
      self.pos_last[~feeding,0],
      self.pos_last[~feeding,1],
    ] = 0
    self.pos_last[~feeding] = pos_next_last

    food = (self.state[feeding] == 0) \
      .flatten(1).to(self.float_type) \
      .multinomial(1).squeeze()
    food_x = t.div(food, self.board_size, rounding_mode='trunc')
    food_y = food % self.board_size

    self.pos_food[feeding,0] = food_x
    self.pos_food[feeding,1] = food_y

    self.state[feeding, food_x, food_y] = 1
    self.state[
      self.idxs,
      pos_next[:,0],
      pos_next[:,1]
    ] = action_dirs

    self.state[dead] = 0
    self.state[dead,0,0] = 3
    self.state[dead,0,1] = 3
    self.state[dead,0,2] = 1

    return self.state

