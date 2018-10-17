import tensorflow as tf
import keras.backend as K


def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)


def adjust_binary_cross_entropy(y_true, y_pred):
    return K.binary_crossentropy(y_true, K.pow(y_pred, 2))


def MMD_Loss_func(num_source, sigmas=None):
    """
    MMD loss of multiple sources
    :param num_source: number of source domain
    :param sigmas: sigma need to use, default: [1, 5, 10]
    :return:
    """
    if sigmas is None:
        sigmas = [1, 5, 10]

    def loss(y_true, y_pred):
        cost = []

        for i in range(num_source):
            for j in range(num_source):
                domain_i = K.tf.where(K.tf.equal(y_true, i))[:, 0]
                domain_j = K.tf.where(K.tf.equal(y_true, j))[:, 0]
                single_res = mmd_two_distribution(K.gather(y_pred, domain_i),
                                                  K.gather(y_pred, domain_j),
                                                  sigmas=sigmas)
                cost.append(single_res)
        cost = K.concatenate(cost)
        return K.mean(cost)
    return loss


def mmd_two_distribution(source, target, sigmas):
    """
    compute mmd loss between two distributions
    :param source: [num_samples, num_features]
    :param target: [num_samples, num_features]
    :return:
    """

    sigmas = K.constant(sigmas)
    xy = rbf_kernel(source, target, sigmas)
    xx = rbf_kernel(source, source, sigmas)
    yy = rbf_kernel(target, target, sigmas)
    return xx + yy - 2 * xy


def rbf_kernel(x, y, sigmas):
    """
    compute the rbf kernel value
    :param x: [num_x_samples, num_features]
    :param y: [num_y_samples, num_features]
    :param sigmas: sigmas need to use
    :return: single value of x, y kernel
    """
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    dot = -K.dot(beta, K.reshape(dist, (1, -1)))
    exp = K.exp(dot)
    return K.mean(exp, keepdims=True)


def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]
    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples].
    """
    norm = lambda x: K.sum(K.square(x), axis=1)
    return norm(K.expand_dims(x, 2) - K.transpose(y))