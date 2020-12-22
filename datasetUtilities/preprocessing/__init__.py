if __name__=='__main__' :
    from simplepreprocessor import SimplePreprocessor
    from imagetoarraypreprocessor import ImageToArrayPreprocessor
    print('INFO:simplepreprocessor init')
else:
    from .simplepreprocessor import SimplePreprocessor
    from .imagetoarraypreprocessor import ImageToArrayPreprocessor
    print('INFO:simplepreprocessor init')
