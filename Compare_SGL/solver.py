import commands
import os

for ypheno_root, ypheno_dirs, files in os.walk("./data/pheno", topdown=False):
    print 'Current dir: ' + ypheno_root
    for ypheno_file_name in files:
        if ypheno_file_name[-10:-4] != 'result' and ypheno_file_name[-4:] == '.csv':
            print 'Current ypheno file: ' + ypheno_file_name

            root = ypheno_root[-3:] if ypheno_root[-3:-1] == 'ex' else ''
            file = os.path.join(root, ypheno_file_name)

            print '==================================='
            print '----Has--PCA'
            t = commands.getoutput('python test_cv_0.py {0} true'.format(file))
            print t
            if 'error' in t.lower():
                break
            print '==================================='
            print '----NO--PCA'
            t = commands.getoutput('python test_cv_0.py {0} false'.format(file))
            print t
            if 'error' in t.lower():
                break
            print '==================================='
