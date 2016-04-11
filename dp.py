import numpy as np
table = np.zeros((31,2))

def f(s1, s2, n):
    #print "in f", n, s1, s2
    if (s1 == 1 and s2 == 0) and (n > 1):
        #print .5
        return .5
    else:
        #print 1
        return 1 


def r(n, s):
    if (n == 0):
        table[n][s] = 1
        return 1
    else:
        #print "counts", table[n-1][0], table[n-1][1]
        table[n][s] = table[n-1][0]*f(0,s,n) + round(table[n-1][1]*f(1,s,n))
        return table[n][s]

def c(n):
    return r(n, 0) + r(n, 1)
    

def main(n):
    for i in xrange(31):
        print i, c(i)
    
    
    