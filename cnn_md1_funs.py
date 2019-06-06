from scipy import stats
import gzip
import numpy as np

## Returns idexes of maximum value of the array
def nanargmax(a):
    idx = np.argmax(a, axis=None)
    multi_idx = np.unravel_index(idx, a.shape)
    if np.isnan(a[multi_idx]):
        nan_count = np.sum(np.isnan(a))
        idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
        multi_idx = np.unravel_index(idx, a.shape)
    return multi_idx

def maxpool(X, f, s):
    """
    :param X: input image
    :param f: pool size
    :param s: stride
    :return: max pooled image
    uses valid padding
    """
    (l, w, w) = X.shape
    pool2d = np.zeros((l, int(np.floor((w-f)/s+1)),int(np.floor((w-f)/s+1))))
    for c in range(0,l):
        for i in range(0, w-f+s,s):
            for j in range(0, w-f+s,s):
                pool2d[c,i,j] = np.max(X[c,i:i+f,j:j+f])
    return pool2d


def softmax_entropy(out, y):
    """
    :param out: output of fully connected layer
    :param y: true labels as one-hot vector
    :return: loss as cross entropy and softmax probabilities
    """
    eout = np.exp(out - np.max(out), dtype=np.float)
    probs = eout / sum(eout)

    p = sum(y * probs)
    loss = -np.log(p + 1e-9) / out.shape[0]  ## (Only data loss. No regularised loss)
    return loss, probs

def get_truncnorm(sigma = 0.1, mu = 0, intvl=2):
    """
    Define a truncated normal distribution for weight initialization
    :param sigma_lm1: standard deviation
    :param mu_lm1: mean
    :param intvl: interva for truncation
    :return: a truncated normal distribution
    """
    return stats.truncnorm(a=(-intvl * sigma - mu) / sigma, b=(intvl * sigma - mu) / sigma, loc=mu, scale=sigma)

def initialize_weight_tnorm(shape):
    """
    Get random variables with desired shape
    :param shape: size of filter
    :return: random variables with input shape
    """
    truncated_normal = get_truncnorm()
    return truncated_normal.rvs(size=shape)

def conv2d(image, w, filt, bias, fc, f, s):
    """
    #do a convolution between image and filt
    :param image: input image
    :param w: size of image
    :param filt: filter
    :param fc: filter channels
    :param f: filter size
    :param s: strides
    :return: returns 2d output of convolution
    """
    #define convolution output size
    w1 = int(np.floor((w - f + s) / s))
    conv1 = np.zeros((fc, w1, w1))

    # apply fcxfxf filter and apply relu
    for c in range(0, fc):
        for i in range(0, w - f + s, s):
            for j in range(0, w - f + s, s):
                conv1[c, i, j] = np.sum(image[:, i:i + f, j:j + f] * filt[c]) + bias[c]
    #apply relu activation
    conv1[conv1 <= 0] = 0
    return conv1

def backprop_dconv(dpool,dconv, conv, l, w, fp, s):
    """
    backpropagation for pool to conv
    :param dpool: pool backprop
    :param dconv: conv backprop to be initialized
    :param conv: conv
    :param l: channels for conv
    :param w: size for conv
    :param fp: pool size
    :param s: stride
    :return: dconv, backprop for conv
    """
    for c in range(0, l):
        for i in range(0, w - fp + s, s):
            for j in range(0, w - fp + s, s):
                (a, b) = nanargmax(conv[c, i:i+fp, j:j+fp])  ## Getting indexes of maximum value in the array
                dconv[c, i+a, j+b] = dpool[c, i, j]
    dconv[conv <= 0] = 0
    return dconv


def extract_data(filename, num_images, IMAGE_WIDTH):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels