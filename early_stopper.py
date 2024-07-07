class EarlyStopper:

    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.count = 0
        self.min_loss = float('inf')
    
    def reset(self):
        self.count = 0
        self.min_loss = float('inf')
    
    def stop(self, loss):
        if (loss < self.min_loss):
            self.min_loss = loss
            self.count = 0
        elif loss >= self.min_loss + self.min_delta:
            self.count += 1
            if self.count == self.patience:
                return True
        return False