import random
import numpy as np

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y ** 2

class TextLSTM:
    data = None
    num_chars = None
    char_to_idx = None
    idx_to_char = None
    W = None
    B = None
    Why = None
    By = None

    def __init__(self, text_file_str):
        self.data = open(text_file_str, 'r').read()

        chars = list(set(self.data))
        self.num_chars = len(chars)

        self.char_to_idx = { c:i for i,c in enumerate(chars) }
        self.idx_to_char = { i:c for i,c in enumerate(chars) }

        self.W = np.random.randn(4, self.num_chars, self.num_chars * 2) * 0.01
        self.B = np.random.randn(4, self.num_chars, 1) * 0.01
        self.Why = np.random.randn(self.num_chars, self.num_chars) * 0.01
        self.By = np.zeros((self.num_chars, 1))

    def forwardPass(self, inputs, targets, h_prev, c_prev):
        x, h, i, f, o, g, p, c, y = {}, {}, {}, {}, {}, {}, {}, {}, {}
        h[-1] = np.copy(h_prev)
        c[-1] = np.copy(c_prev)
        loss = 0
        for idx in range(len(inputs)):
            x[idx] = np.zeros((self.num_chars, 1))
            x[idx][inputs[idx]] = 1
            xh = np.concatenate((x[idx], h[idx - 1]))
            Wxhb = np.dot(self.W, xh) + self.B
            i[idx] = sigmoid(Wxhb[0])
            f[idx] = sigmoid(Wxhb[1])
            o[idx] = sigmoid(Wxhb[2])
            g[idx] = tanh(Wxhb[3])
            c[idx] = np.multiply(f[idx], c[idx - 1]) + np.multiply(i[idx], g[idx])
            h[idx] = np.multiply(o[idx], tanh(c[idx]))
            y[idx] = np.dot(self.Why, h[idx]) + self.By

            # Loss
            exp_y = np.exp(y[idx])
            p[idx] = exp_y / np.sum(exp_y)
            loss += -np.log(p[idx][targets[idx]])

        return loss, x, h, c, p, i, f, o, g

    def backwardPass(self, targets, x, h, c, p, i, f, o, g):
        dh_dh = np.zeros_like(h[0])
        dc_dc = np.zeros_like(c[0])

        dW = np.zeros_like(self.W)
        dB = np.zeros_like(self.B)
        dWhy = np.zeros_like(self.Why)
        dBy = np.zeros_like(self.By)
        
        for idx in reversed(range(len(targets))):
            # dWhy
            dL_dy = np.copy(p[idx])
            dL_dy[targets[idx]] -= 1
            dy_dWhy = h[idx]
            dWhy += np.dot(dL_dy, dy_dWhy.T)

            # dBy
            dBy += dL_dy

            dy_dh = self.Why
            dL_dh = np.dot(dy_dh.T, dL_dy)
            dh = dL_dh + dh_dh

            dtanhc = o[idx] + dh
            dtanhc_dc = 1 - tanh(c[idx]) ** 2
            dh_dc = dtanhc * dtanhc_dc

            dc = dc_dc + dh_dc
            dc_dc = np.multiply(dc, f[idx])

            xh = np.concatenate((x[idx], h[idx - 1]))

            # dWi/dBi
            di = dc * g[idx] * dsigmoid(i[idx])
            dWi = np.dot(di, xh.T)
            dW[0] += dWi
            dB[0] += di

            # dWf/dBf
            df = dc * c[idx - 1] * dsigmoid(f[idx])
            dWf = np.dot(df, xh.T)
            dW[1] += dWf
            dB[1] += df

            # dWo/dBo
            do = dh * tanh(c[idx]) * dsigmoid(o[idx])
            dWo = np.dot(do, xh.T)
            dW[2] += dWo
            dB[2] += do

            # dWg/dBg
            dg = dc * i[idx] * dtanh(g[idx])
            dWg = np.dot(dg, xh.T)
            dW[3] += dWg
            dB[3] += dg

            di_dh = np.dot(self.W[0, :, self.num_chars:].T, di)
            df_dh = np.dot(self.W[1, :, self.num_chars:].T, df)
            do_dh = np.dot(self.W[2, :, self.num_chars:].T, do)
            dg_dh = np.dot(self.W[3, :, self.num_chars:].T, dg)

            dh_dh = di_dh + df_dh + do_dh + dg_dh

        return dW, dB, dWhy, dBy, h[len(targets) - 1], c[len(targets) - 1]

    def sample(self, h, c, seed_char_idx, chars_to_sample):
        x = np.zeros((self.num_chars, 1))
        x[seed_char_idx] = 1
        char_indices = [seed_char_idx]
        for t in range(chars_to_sample):
            xh = np.concatenate((x, h))
            Wxhb = np.dot(self.W, xh) + self.B
            i = sigmoid(Wxhb[0])
            f = sigmoid(Wxhb[1])
            o = sigmoid(Wxhb[2])
            g = tanh(Wxhb[3])
            c = np.multiply(f, c) + np.multiply(i, g)
            h = np.multiply(o, tanh(c))
            y = np.dot(self.Why, h) + self.By

            # Loss
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

    def train(self, seq_len, learning_rate, optimizer, sample_size, beta1=0.9, beta2=0.999):
        # Starting characters for sampling
        starting_chars = [line[0] for line in self.data.split('\n') if len(line) > 0]

        # Variables for Optimizer
        m1W = np.zeros_like(self.W)
        m1B = np.zeros_like(self.B)
        m1Why = np.zeros_like(self.Why)
        m1By = np.zeros_like(self.By)
        m2W = np.zeros_like(self.W)
        m2B = np.zeros_like(self.B)
        m2Why = np.zeros_like(self.Why)
        m2By = np.zeros_like(self.By)

        cur_idx = 0
        iteration = 0
        smooth_loss = -np.log(1.0 / self.num_chars) * seq_len # loss at iteration 0
        while True:
            if cur_idx + seq_len + 1 >= len(self.data) or iteration == 0: 
                h_prev = np.zeros((self.num_chars, 1))
                c_prev = np.zeros((self.num_chars, 1))
                cur_idx = 0
            inputs = [self.char_to_idx[c] for c in self.data[cur_idx : cur_idx + seq_len]]
            targets = [self.char_to_idx[c] for c in self.data[cur_idx + 1 : cur_idx + seq_len + 1]]
            loss, x, h, c, p, i, f, o, g = self.forwardPass(inputs, targets, h_prev, c_prev)
            dW, dB, dWhy, dBy, h_prev, c_prev = self.backwardPass(targets, x, h, c, p, i, f, o, g)

            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if iteration % 100 == 0:
                print("Iter %d  |  Loss: %f" % (iteration, smooth_loss))
                seed_char_idx = self.char_to_idx[random.choice(starting_chars)]
                sample = self.sample(h_prev, c_prev, seed_char_idx, sample_size)
                print("---------------------------")
                print(sample)
                print("---------------------------")

            params = zip([self.W, self.B, self.Why, self.By], 
                         [dW, dB, dWhy, dBy], 
                         [m1W, m1B, m1Why, m1By],
                         [m2W, m2B, m2Why, m2By])

            if optimizer == 'adam':
                self.updateAdam(beta1, beta2, learning_rate, params, iteration)
            elif optimizer == 'adagrad':
                self.updateAdagrad(learning_rate, params)
            else:
                assert(False)

            cur_idx += seq_len
            iteration += 1

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

    def backwardPass(self, targets, x, h, p):
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

    def train(self, seq_len, learning_rate, optimizer, sample_size, beta1=0.9, beta2=0.999):
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
            dWxh, dWhh, dWhy, dBh, dBy, h_prev = self.backwardPass(targets, x, h, p)

            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if iteration % 100 == 0:
                print("Iter %d  |  Loss: %f" % (iteration, smooth_loss))
                seed_char_idx = self.char_to_idx[random.choice(starting_chars)]
                sample = self.sample(h_prev, seed_char_idx, sample_size)
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
    np.set_printoptions(precision=3, suppress=True)

    # rnn = TextRNN("easy.txt", 10)
    # rnn.train(25, learning_rate=1e-1, optimizer='adagrad', sample_size=5)

    lstm = TextLSTM("char-lstm.py")
    lstm.train(500, learning_rate=1e-3, optimizer='adam', sample_size=500)

