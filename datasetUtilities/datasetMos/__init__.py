if __name__=='__main__' :
    from simpledatasetloader import SimpleDatasetLoader
    print('INFO:dataset init')
else:
    from .simpledatasetloader import SimpleDatasetLoader
    print('INFO:dataset init')

