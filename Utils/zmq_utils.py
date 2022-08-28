import zmq


def zmq_nonblocking_recv(socket):
    raw_data_list = []
    while True:
        try:
            data = socket.recv(zmq.NOBLOCK)
            raw_data_list.append(data)
        except zmq.ZMQError as e:
            break
    return raw_data_list


def zmq_nonblocking_multipart_recv(socket):
    raw_data_list = []
    while True:
        try:
            data = socket.recv_multipart(zmq.NOBLOCK)
            raw_data_list.append(data)
        except zmq.ZMQError as e:
            break
    return raw_data_list


# 上面两个函数都是接收数据，第一个是一次只能接一条，第二个可以一次性接多条
