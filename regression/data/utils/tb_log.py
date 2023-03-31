# MIT License

# Copyright (c) 2022 Tung Nguyen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import gzip
from torch.utils.tensorboard import SummaryWriter
import copy

def get_tb_logger(args):
    if args.enable_tensorboard:
        return TensorboardLogger(args)
    else:
        return Logger(args)

class Logger:
    def __init__(self, args):
        self.data = {}
        self.args = copy.deepcopy(vars(args))
        self.context = ""

    def set_context(self, context):
        self.context = context

    def add_scalar(self, key, value, iter_idx=None, use_context=False):
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]

    def add_object(self, key, value, use_context=False):
        if use_context:
            key = self.context + '/' + key
        self.data[key] = value

    def save(self, save_path, args):
        pickle.dump({'logged_data': self.data, 'args': self.args}, gzip.open(save_path, 'wb'))


class TensorboardLogger(Logger):
    def __init__(self, args):
        self.data = {}
        self.context = ""
        self.args = copy.deepcopy(vars(args))
        self.writer = SummaryWriter(log_dir=args.tb_log_dir, comment=f"{args.expid}")
        print(f"SAVING IN {args.tb_log_dir}")
        print(self.args)
        self.writer.add_hparams(self.args, {})

    def set_context(self, context):
        self.context = context

    def add_scalar(self, key, value, iter_idx=None, use_context=False):
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]

        if iter_idx is not None:
            self.writer.add_scalar(key, value, iter_idx)
        else:
            self.writer.add_scalar(key, value, len(self.data[key]))

    def add_object(self, key, value, use_context=False):
        if use_context:
            key = self.context + '/' + key
        self.data[key] = value

    def save(self, save_path, args):
        pickle.dump({'logged_data': self.data, 'args': self.args}, gzip.open(save_path, 'wb'))
        self.writer.flush()
