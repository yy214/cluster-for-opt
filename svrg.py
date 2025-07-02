# from https://github.com/ishaqadenali/SVRG-Pytorch/

import torch
from torch.optim.optimizer import Optimizer, required


class SVRG(Optimizer):
    r""" implement SVRG """ 

    def __init__(self, params, lr=required, freq=10):
        
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, freq=freq)
        self.counter = 0 # counts inner / outer iterations
        super(SVRG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SVRG, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('m', )

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            freq = group['freq']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                
                if 'large_batch' not in param_state:
                    buf = param_state['large_batch'] = torch.zeros_like(p.data)
                    buf.add_(d_p) #add first large, low variance batch
                    #need to add the second term in the step equation; the gradient for the original step!
                    buf2 = param_state['small_batch'] = torch.zeros_like(p.data)

                buf = param_state['large_batch']
                buf2 = param_state['small_batch']

                if self.counter == freq:
                    buf.data = d_p.clone() #copy new large batch. Begining of new inner loop
                    buf2.data = torch.zeros_like(p.data)
                    
                if self.counter == 1:
                    buf2.data.add_(d_p) #first small batch gradient for inner loop!

                #dont update parameters when computing large batch (low variance gradients)
                if self.counter != freq and self.counter != 0:
                    p.data.add_(-group['lr'], (d_p - buf2 + buf) )

        if self.counter == freq:
            self.counter = 0

        self.counter += 1    

        return loss



alpha = 1e-2
freq = 100 #how often to recompute large gradient
            #The optimizer will not update model parameters on iterations
            #where the large batches are calculated

lg_batch = 3000 #size of large gradient batch (replaces full batch)
min_batch = 300 #size of mini batch

optimizer = SVRG(model.parameters(), lr = alpha, freq = freq)

epochs = 50
iterations = int (epochs * (60000/min_batch))

#SVRG Training
counter = 0
total = time.time()
while(counter < iterations):
    #compute large batch gradient
    large_batch_indices = np.random.choice(x.size()[0], lg_batch, replace = False)
    large_batch_indices = torch.from_numpy(large_batch_indices)
    x_batch = torch.index_select(x, 0, large_batch_indices)
    y_pred = model(x_batch)
    y_batch = torch.index_select(y, 0, large_batch_indices).type(torch.LongTensor)
    loss = loss_fn(y_pred, torch.max(y_batch, 1)[1])
    optimizer.zero_grad()
    loss.backward()
    counter+=1
    optimizer.step()

    #update models using mini batch gradients
    for i in range(freq-1):
        batch_indices = np.random.choice(x.size()[0], min_batch, replace = False)
        batch_indices = torch.from_numpy(batch_indices)
        x_batch = torch.index_select(x, 0, batch_indices)
        y_pred = model(x_batch)
        y_batch = torch.index_select(y, 0, batch_indices).type(torch.LongTensor)
        loss = loss_fn(y_pred, torch.max(y_batch, 1)[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter += 1
        if (counter == iterations):
            break
print('time for SVRG ' + str(iterations) +' steps')
print(time.time()-total)
print('')