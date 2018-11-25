import numpy


def samp_down(image):
    u = image
    n_x, n_y = u.shape
    m_x, m_y = n_x // 2 * 2, n_y // 2 * 2
    u_s = (u[0:m_x:2, 0:m_y:2] + u[0:m_x:2, 1:m_y:2] + u[1:m_x:2, 0:m_y:2] + u[1:m_x:2, 1:m_y:2]) / 4.0
    return u_s
