import random
import numpy as np

class TextRNN:
    data = None
    num_chars = None
    char_to_idx = None
    idx_to_char = None
    hidden_size = None
    Wxh = None
    Whh = None
    Why = None
    Bh = None
    By = None

    def __init__(self, text_file_str, hidden_size):
        self.data = open(text_file_str, 'r').read()

        chars = list(set(self.data))
        self.num_chars = len(chars)
        self.hidden_size = hidden_size

        self.char_to_idx = { c:i for i,c in enumerate(chars) }
        self.idx_to_char = { i:c for i,c in enumerate(chars) }

        self.Wxh = np.random.randn(hidden_size, self.num_chars) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(self.num_chars, hidden_size) * 0.01
        self.Bh = np.zeros((hidden_size, 1))
        self.By = np.zeros((self.num_chars, 1))

    def forwardPass(self, inputs, targets, h_prev):
        assert(len(inputs) == len(targets))

        x, h, y, p = {}, {}, {}, {}
        h[-1] = np.copy(h_prev)
        loss = 0
        for idx in range(len(inputs)):
            x[idx] = np.zeros((self.num_chars, 1))
            x[idx][inputs[idx]] = 1
            Wh = np.dot(self.Whh, h[idx - 1])
            Wx = np.dot(self.Wxh, x[idx])
            h[idx] = np.tanh(Wh + Wx + self.Bh)
            y[idx] = np.dot(self.Why, h[idx]) + self.By
            exp_y = np.exp(y[idx])
            p[idx] = exp_y / np.sum(exp_y)
            loss += -np.log(p[idx][targets[idx]])

        return (loss, x, h, p)

    def backwardPass(self, targets, x, h, p, learning_rate):
        dL_dY = np.copy(p)
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dBh, dBy = np.zeros_like(self.Bh), np.zeros_like(self.By)
        dh_dh = np.zeros_like(h[0])
        
        for idx in reversed(range(len(targets))):
            dL_dY = np.copy(p[idx])
            dL_dY[targets[idx]] -= 1

            # Why
            dY_dWhy = h[idx]
            dWhy += np.dot(dL_dY, dY_dWhy.T)

            # By
            dBy += dL_dY

            dY_dh = self.Why
            dL_dh = np.dot(dY_dh.T, dL_dY)

            dh_dtanh = 1 - (h[idx] * h[idx])
            dh = dh_dtanh * (dL_dh + dh_dh)
            dh_dh = np.dot(self.Whh.T, dh)

            # Bh
            dBh += dh

            # Whh
            dh_dWhh = h[idx - 1]
            dWhh += np.dot(dh, dh_dWhh.T)

            # Wxh
            dh_dWxh = x[idx]
            dWxh += np.dot(dh, dh_dWxh.T)

        return dWxh, dWhh, dWhy, dBh, dBy, h[len(targets) - 1]

    def sample(self, h, seed_char_idx, chars_to_sample):
        x = np.zeros((self.num_chars, 1))
        x[seed_char_idx] = 1
        char_indices = [seed_char_idx]
        for t in range(chars_to_sample):
            Wh = np.dot(self.Whh, h)
            Wx = np.dot(self.Wxh, x)
            h = np.tanh(Wh + Wx + self.Bh)
            y = np.dot(self.Why, h) + self.By
            exp_y = np.exp(y)
            p = exp_y / np.sum(exp_y)

            idx = np.random.choice(range(self.num_chars), p=p.ravel())
            x = np.zeros((self.num_chars, 1))
            x[idx] = 1
            char_indices.append(idx)
        string = ''.join(self.idx_to_char[idx] for idx in char_indices)
        return string

    def updateAdagrad(self, learning_rate, params):
        for param, dparam, mem, _ in params:
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    def updateAdam(self, beta1, beta2, learning_rate, params, iteration):
        for param, dparam, mom1, mom2 in params:
            mom1 = beta1 * mom1 + (1 - beta1) * dparam
            mom2 = beta2 * mom2 + (1 - beta2) * dparam * dparam
            first_unbias = mom1 / (1 - beta1 ** (iteration + 1))
            second_unbias = mom2 / (1 - beta2 ** (iteration + 1))
            param -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-8)

    def train(self, seq_len, learning_rate, optimizer, beta1=0.9, beta2=0.999):
        # Starting characters for sampling
        starting_chars = [line[0] for line in self.data.split('\n') if len(line) > 0]

        # Variables for Optimizer
        m1Wxh, m1Whh, m1Why = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        m1Bh, m1By = np.zeros_like(self.Bh), np.zeros_like(self.By)
        m2Wxh, m2Whh, m2Why = np.zeros_like(m1Wxh), np.zeros_like(m1Whh), np.zeros_like(m1Why)
        m2Bh, m2By = np.zeros_like(m1Bh), np.zeros_like(m1By)

        cur_idx = 0
        iteration = 0
        smooth_loss = -np.log(1.0 / self.num_chars) * seq_len # loss at iteration 0
        while True:
            if cur_idx + seq_len + 1 >= len(self.data) or iteration == 0: 
                h_prev = np.zeros((self.hidden_size, 1))
                cur_idx = 0
            inputs = [self.char_to_idx[c] for c in self.data[cur_idx : cur_idx + seq_len]]
            targets = [self.char_to_idx[c] for c in self.data[cur_idx + 1 : cur_idx + seq_len + 1]]
            loss, x, h, p = self.forwardPass(inputs, targets, h_prev)
            dWxh, dWhh, dWhy, dBh, dBy, h_prev = self.backwardPass(targets, x, h, p, learning_rate)

            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if iteration % 100 == 0:
                print("Iter %d  |  Loss: %f" % (iteration, smooth_loss))
                seed_char_idx = self.char_to_idx[random.choice(starting_chars)]
                sample = self.sample(h_prev, seed_char_idx, 50)
                print("---------------------------")
                print(sample)
                print("---------------------------")

            params = zip([self.Wxh, self.Whh, self.Why, self.Bh, self.By], 
                                          [dWxh, dWhh, dWhy, dBh, dBy], 
                                          [m1Wxh, m1Whh, m1Why, m1Bh, m1By],
                                          [m2Wxh, m2Whh, m2Why, m2Bh, m2By])

            if optimizer == 'adam':
                self.updateAdam(beta1, beta2, learning_rate, params, iteration)
            elif optimizer == 'adagrad':
                self.updateAdagrad(learning_rate, params)
            else:
                assert(False)

            cur_idx += seq_len
            iteration += 1

if __name__ == '__main__':
    rnn = TextRNN("easy.txt", 10)
    rnn.train(25, learning_rate=1e-1, optimizer='adagrad')

