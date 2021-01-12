import time
from tqdm import tqdm, trange
import random

def t1():
    for i in tqdm(range(100)):
        time.sleep(0.1)

def t2():
     pbar = tqdm(["a", "b", "c", "d"])
     for char in pbar:
          # 设置描述
          pbar.set_description("Processing %s" % char)
          time.sleep(1)
def t3():
    with tqdm(total=100, ncols=100) as pbar:
        for i in range(10):
            time.sleep(0.1)
            pbar.update(10)
def t4():
    with trange(100, ncols=150) as t:
        for i in t:
            # Description will be displayed on the left
            t.set_description('GEN %i' % i)
            # Postfix will be displayed on the right,
            # formatted automatically based on argument's datatype
            t.set_postfix(loss=random.random(), gen=random.randint(1, 999), str='h',
                          lst=[1, 2])
            time.sleep(0.1)

def t5():
    for i in trange(4, ncols=150, desc='1st loop'):
        for j in trange(5,  ncols=150, desc='2nd loop'):
            for k in trange(50,  ncols=150, desc='3rd loop', leave=False):
                time.sleep(0.01)

if __name__ == "__main__":
    # t1()
    # t2()
    # t3()
    # t4()
    t5()
