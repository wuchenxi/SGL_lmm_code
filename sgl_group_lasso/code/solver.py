import commands
import threading
import argparse
import os
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', default = 'data/pheno', help = 'the root path of the ypheno files, default is data/pheno/')
    parser.add_argument('--use-group-lasso', '-g', default = 'true', help = 'whether use group lasso, default is true, value is true/false')
    parser.add_argument('--use-lasso', '-l', default = 'true', help = 'whether use lasso, default is true, value is true/false')
    args = parser.parse_args()

    threads = []

    for ypheno_root, ypheno_dirs, files in os.walk(args.dir, topdown=False):
        for ypheno_file_name in files:
            if ypheno_file_name[-10:-4] != 'result' and ypheno_file_name[-4:] == '.csv':
                root = ypheno_root[5:]
                file = os.path.join(root, ypheno_file_name)

                t = threading.Thread(target = commands.getoutput,
                                     args = ('python test_cv_0.py {0} {1} {2}'.format(
                                         file, args.use_group_lasso, args.use_lasso),))
                print 'python test_cv_0.py {0} {1} {2}'.format(
                                         file, args.use_group_lasso, args.use_lasso)
                threads.append(t)

    for t in threads:
        t.setDaemon(True)
        t.start()
    t.join()
    print "all over %s" % time.ctime()
