
class SR_node(object):

    def __init__(self, cell,  height, m_value):
        self.height = height
        self.value = m_value
        self.cell = cell

def Subroutine(mfobject, n, theta, alpha):
    d = mfobject.domain_dim
    cell = tuple([(0, 1)] * d)
    active_cells = []

    root = SR_node(cell, 0, 0)

    # split root into 2
    pcell = list(root.cell)
    # get the range of the span
    span = [abs(pcell[i][1] - pcell[i][0]) for i in range(len(pcell))]





