import  torch
import gops


gops.init()

my_rank = gops.get_rank()
a = torch.ones([16, 16]) * my_rank / (256.)
a = a.cuda()
b = torch.zeros_like(a).cuda()

print('pre {} {}'.format(my_rank, a.sum().item()))
for i in range(8):
    gops.ring_pass(a.data_ptr(), b.data_ptr(), 256, 1)
    print('pass {} post {} {}'.format(i, my_rank, b.sum().item()))
    if i == 4:
        gops.wait()
    c = a
    a = b
    b = c

gops.finalize()
