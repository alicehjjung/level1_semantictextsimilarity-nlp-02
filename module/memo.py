def memo(path, **kwargs):
    """ Hyperparameter 를 txt로 기록하는 함수
    """
    with open(path, 'a') as f:
        f.write("{\n")
        for k, v in kwargs.items():
            text = '{:<15}:{:>15}\n'.format(k,v)
            f.write(text)
        f.write("}\n")