import commands
import threading
import argparse
import os
import time


def excec(comm):
    result = commands.getoutput(comm)
    state_file = open('data/' + comm.split()[2].replace('.csv',
                                                        '{0}_stat.txt'.format(
                                                            '_PCA' if comm.split()[3] == 'true' else '')), 'w')
    state_file.write(str(result))
    state_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', default='data/pheno',
                        help='the root path of the ypheno files, default is data/pheno/')
    parser.add_argument('--use-group-lasso', '-g', default='true',
                        help='whether use group lasso, default is true, value is true/false')
    parser.add_argument('--use-lasso', '-l', default='true',
                        help='whether use lasso, default is true, value is true/false')
    args = parser.parse_args()

    threads = []

    for ypheno_root, ypheno_dirs, files in os.walk(args.dir, topdown=False):
        for ypheno_file_name in files:
            if ypheno_file_name[-10:-4] != 'result' and ypheno_file_name[-4:] == '.csv':
                root = ypheno_root[5:]
                file = os.path.join(root, ypheno_file_name)

                for choice in ['true', 'false']:
                    t = threading.Thread(target=excec,
                                         args=('python test_cv_0.py {0} {1}'.format(
                                             file, choice),))
                    print 'python test_cv_0.py {0} {1}'.format(
                        file, choice)
                    threads.append(t)

    p = threads[0]
    for t in threads:
        t.setDaemon(True)
        t.start()
        p = t
    p.join()
    print "all over %s" % time.ctime()