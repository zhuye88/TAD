import scipy.io as scio


def load_graph(file_path):
    graph = scio.loadmat(file_path)
    attr = graph["Attributes"].A
    adj = graph["Network"]
    label = graph["Label"]
    return attr, adj, label
