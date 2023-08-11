import numpy as np
import numpy.matlib as mlib
import numpy.testing as npt
import pdb

def generate_matrix(M,N,fname):
    A = np.random.rand(M,N)
    #a1 = np.arange(1,N+1)
    #A = mlib.repmat(a1,M,1)

    np.savetxt(fname,A)
    pass

def test(fname1,fname2,ansf):
    A = np.loadtxt(fname1) 
    B = np.loadtxt(fname2) 
    ans = np.loadtxt(ansf)
    C = np.matmul(np.transpose(A),B)

    print(npt.assert_array_equal(ans,C))


if __name__=="__main__":
    fname1 = "A.TXT"
    fname2 = "B.TXT"
    #generate_matrix(512,128,fname1)
    #generate_matrix(128,256,fname2)
    #test(fname1,fname2,"blk_4_4.out")
    test(fname1,fname2,"ans.out")
