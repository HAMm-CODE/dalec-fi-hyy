import sys; sys.path[:0] = ["src", "showcase"]
import model as show
from dalec.acm import acm_from_config; from dalec.config import load_config
from dalec.sampler import model_from_config
cfg = load_config("showcase/config.yaml"); data = show.load_data(cfg, 1997, 2010)
real = model_from_config(load_config(), data).model
mine, _ = show.build_model(data, acm_from_config(cfg), "hemisurface")
p = real.initial_point(); print(real.compile_logp()(p), mine.compile_logp()(p))
