# Utility functions for second-quantization calculations using the SOLAX library.
# source: https://zenodo.org/records/14740476
from itertools import combinations
from itertools import product
import solax as sx
import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp

def create_full_determinant_basis(N_el,N_orb,S=None):
    # S is a list with possible spin entries
    assert N_el <= 2 * N_orb, \
        'Requesting more electrons than orbitals.'

    basis = []
    for positions in combinations(range(2 * N_orb), N_el):
        base_list = [0] * 2 * N_orb
        for pos in positions:
            base_list[pos] = 1
        basis.append(''.join(map(str, base_list)))

    # start with choosing N_el_up from N_orb for spin up and then N_el_dn from N_orb for spin dn 
    if S:
        S = np.array(S)*2
        S = S.astype(int)
        basis_S = []
        for det in basis:
            up = 0
            dn = 0
            for i,num in enumerate(det):
                if num == '1':
                    if i%2 == 0:
                        up += 1
                    elif i%2 == 1:
                        dn += 1
            if (up - dn) in S:
                basis_S.append(det)
        basis = basis_S

    basis = sx.Basis(basis)

    return basis

def create_model_U(N_orb,num_U_vals,seed=1234):
    
    all_indices = np.array(list(product(range(N_orb), repeat=4)))
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    indices = all_indices[:num_U_vals]
    normal_sample = np.random.normal(loc=0.0, scale=0.0005, size=int(0.99*num_U_vals))
    uniform_sample = np.random.uniform(low=-1, high=1, size=int(0.01*num_U_vals))
    U_mod_vals = np.abs(np.concatenate((normal_sample, uniform_sample)))
    U_mod_tensor = np.hstack((indices, U_mod_vals.reshape(-1, 1)))
    return U_mod_tensor
    
def create_rand_basis(N_orb,N_el,num_basis):

    char_list = ['1']*N_el + ['0']*(2*N_orb-N_el)
    permutations = [] 

    for _ in range(num_basis): 
        np.random.shuffle(char_list) 
        permutations.append(''.join(char_list))

    return sx.Basis(permutations)


def num_offdiag(mat,tresh):
    
    counter = 0
    imax,jmax=np.shape(mat)
    for i in range(imax):
        for j in range(jmax):
            if i != j and abs(mat[i,j])>tresh:
                counter += 1
                
    return counter


def find_lowest_states(hermition_matrix,k,rand_keys):
    
    np.random.seed(jax.random.randint(rand_keys,(1,),0,jnp.iinfo(jnp.int32).max)[0])
    v0 = np.random.rand(min(hermition_matrix.shape))
    eval, evec = sp.sparse.linalg.eigsh(hermition_matrix, k=k, which="SA",v0=v0)
    
    return eval, evec


def print_leading_coefficients(state, number=10):
    
    coeff = state.coeffs
    basis = state.basis
    
    sorter = np.argsort(np.abs(coeff))[-number :]

    print('Configuration-----Coefficient')

    for i in sorter[::-1]:
        to_log = '%-100s-----%10.6f' %(basis[int(i)], coeff[i])
        print(to_log.replace(' ', '-'))
        
def build_op_0(V_values,N_orb,daggers):
    
    H_0_vals = []
    H_0_posits = []

    # noninteracting Hamiltonian: 2*N_orb entries
    for i in range(N_orb):
        for j in range(N_orb):
            # spin preserving hopping
            H_0_vals += [V_values[i,j], V_values[i,j]]
            H_0_posits += [ [2 * i     , 2 * j    ],
                            [2 * i + 1 , 2 * j + 1]]
    
    return sx.Operator(daggers, np.array(H_0_posits), np.array(H_0_vals))

def build_op_int(U_values,N_orb,daggers):
    
    H_int_vals = []
    H_int_posits = []
    
    for indices,value in U_values.items():
        i,j,k,l = indices
        
        if i < N_orb and j < N_orb and k < N_orb and l < N_orb:

            H_int_vals += [ -0.5 * value,
                            -0.5 * value,
                            -0.5 * value,
                            -0.5 * value]
            
            H_int_posits += [
                                [2 * i     , 2 * j     , 2 * k     , 2 * l    ],
                                [2 * i + 1 , 2 * j + 1 , 2 * k + 1 , 2 * l + 1],
                                [2 * i + 1 , 2 * j     , 2 * k + 1 , 2 * l    ],
                                [2 * i     , 2 * j + 1 , 2 * k     , 2 * l + 1]
                                ]
            
    return sx.Operator(daggers, np.array(H_int_posits), np.array(H_int_vals))

def build_op_MF(U_values,Rho):
    
    H_MF_vals = []
    H_MF_posits = []
    dE_MF = 0
    
    daggers = (1,0)

    N_orb = int(np.shape(Rho)[0]/2)

    # if not Rho:
    #     Rho = np.diag([1.0] * N_el + [0.0] * (2 * N_orb - N_el))
    
    for indices,value in U_values.items():
        i,j,k,l = indices
        
        if i < N_orb and j < N_orb and k < N_orb and l < N_orb:
        
            dE_MF += (   
                        -0.5 * value * Rho[2 * i     , 2 * k    ] * Rho[2 * j     , 2 * l    ]
                        +0.5 * value * Rho[2 * i     , 2 * l    ] * Rho[2 * j     , 2 * k    ]
                        -0.5 * value * Rho[2 * i + 1 , 2 * k + 1] * Rho[2 * j + 1 , 2 * l + 1]
                        +0.5 * value * Rho[2 * i + 1 , 2 * l + 1] * Rho[2 * j + 1 , 2 * k + 1]

                        -0.5 * value * Rho[2 * i     , 2 * k    ] * Rho[2 * j + 1 , 2 * l + 1]
                        +0.5 * value * Rho[2 * i     , 2 * l + 1] * Rho[2 * j + 1 , 2 * k    ]
                        -0.5 * value * Rho[2 * i + 1 , 2 * k + 1] * Rho[2 * j     , 2 * l    ]
                        +0.5 * value * Rho[2 * i + 1 , 2 * l    ] * Rho[2 * j     , 2 * k + 1]
                        )

            H_MF_vals += [
                        -0.5 * value * Rho[2 * j     , 2 * k    ],
                        0.5 * value * Rho[2 * j     , 2 * l    ],
                        -0.5 * value * Rho[2 * i     , 2 * l    ],
                        0.5 * value * Rho[2 * i     , 2 * k    ],
                        -0.5 * value * Rho[2 * j + 1 , 2 * k + 1],
                        0.5 * value * Rho[2 * j + 1 , 2 * l + 1],
                        -0.5 * value * Rho[2 * i + 1 , 2 * l + 1],
                        0.5 * value * Rho[2 * i + 1 , 2 * k + 1],

                        -0.5 * value * Rho[2 * j + 1 , 2 * k    ],
                        0.5 * value * Rho[2 * j + 1 , 2 * l + 1],
                        -0.5 * value * Rho[2 * i     , 2 * l + 1],
                        0.5 * value * Rho[2 * i     , 2 * k    ],
                        -0.5 * value * Rho[2 * j     , 2 * k + 1],
                        0.5 * value * Rho[2 * j     , 2 * l    ],
                        -0.5 * value * Rho[2 * i + 1 , 2 * l    ],
                        0.5 * value * Rho[2 * i + 1 , 2 * k + 1]
                        ]
            
            H_MF_posits += [    
                                [2 * i     , 2 * l    ],
                                [2 * i     , 2 * k    ],
                                [2 * j     , 2 * k    ],
                                [2 * j     , 2 * l    ],
                                [2 * i + 1 , 2 * l + 1],
                                [2 * i + 1 , 2 * k + 1],
                                [2 * j + 1 , 2 * k + 1],
                                [2 * j + 1 , 2 * l + 1],

                                [2 * i     , 2 * l + 1],
                                [2 * i     , 2 * k    ],
                                [2 * j + 1 , 2 * k    ],
                                [2 * j + 1 , 2 * l + 1],
                                [2 * i + 1 , 2 * l    ],
                                [2 * i + 1 , 2 * k + 1],
                                [2 * j     , 2 * k + 1],
                                [2 * j     , 2 * l    ]
                                ]
        
    return sx.Operator(daggers, np.array(H_MF_posits), np.array(H_MF_vals)) + dE_MF


def drop_herm_conj(Opp):
    
    Opp = (Opp + Opp.hconj)/2
        
    try:
        Opp_half = sx.Operator(Opp["scalar"]/2)
    except:
        Opp_half = sx.Operator()
    
    for key in Opp.keys():
        
        if isinstance(Opp[key],sx.OperatorTerm):
        
            if key == tuple(1 if dag == 0 else 0 for dag in key)[::-1]:
                
                # use only upper triangle
                Opp_half += sx.Operator(Opp[key][[True if x[0] > x[len(key)-1] else False for x in Opp[key].posits]])
                # divide diagonal by 2
                Opp_half += 1/2 * sx.Operator(Opp[key][[True if x[0] == x[len(key)-1] else False for x in Opp[key].posits]])
                
            else:
                print("Operator structure does not allow to be converted.")
                return None
        
    return Opp_half
        

# rather build BD in one add herm conj and subtract D
def update_matrix(opp_hm,A,old_basis,new_basis,det_batch_size,op_batch_size):
    
    # function automatically includes hermitian conjugate of operator

    # want to compute matrix of form    (A B)
    #                                   (C D)
    # A = <old_basis|opp_hm|old_basis> = hm
    # B = <old_basis|opp_hm|new_basis-old_basis>
    # C = <new_basis-old_basis|opp_hm|old_basis>
    # D = <new_basis-old_basis|opp_hm|new_basis-old_basis>
    
    # opp_hm = opp_hm_full - opp_hm.hconj
    
    #                           B1                                        C1                      B2 & C2
    # B & C = (<old_basis|opp_hm|new_basis-old_basis> + <new_basis-old_basis|opp_hm|old_basis>) + h.c.
    # D = <new_basis-old_basis|opp_hm|new_basis-old_basis> + h.c.
        
    B1 = opp_hm.build_matrix(old_basis, new_basis % old_basis,det_batch_size=det_batch_size,op_batch_size=op_batch_size,multiple_devices=True).displace(0, len(old_basis))
    C1 = opp_hm.build_matrix(new_basis % old_basis, old_basis,det_batch_size=det_batch_size,op_batch_size=op_batch_size,multiple_devices=True).displace(len(old_basis), 0)
    
    B2 = C1.hconj
    C2 = B1.hconj
    
    D1 = opp_hm.build_matrix(new_basis % old_basis, new_basis % old_basis,det_batch_size=det_batch_size,op_batch_size=op_batch_size,multiple_devices=True).displace(len(old_basis), len(old_basis))
    D2 = D1.hconj
    
    return A + B1 + B2 + C1 + C2 + D1 + D2


def csr_to_Matrix(hm):
    """
    Transforms a scipy coo object to a solax Matrix object.
    """

    hm = hm.tocoo()

    coord = np.concatenate((hm.col.reshape(-1, 1), hm.row.reshape(-1, 1)), axis=1)
    val = hm.data
    size = hm.shape

    return sx.Matrix(coord,val,size)


def chop(Opp,tresh):

    try:
        if np.abs(Opp["scalar"]) > tresh:
            Opp_chop = sx.Operator(Opp["scalar"])
        else:
            Opp_chop = sx.Operator()
    except KeyError:
        Opp_chop = sx.Operator()

    for key in Opp:
        if isinstance(Opp[key],sx.OperatorTerm):
            Opp_chop += Opp[key].chop(tresh)

    return Opp_chop


def full_len(Opp):
    
    try:
        sx.Operator(Opp["scalar"])
        total_num = 1
    except:
        total_num = 0

    for key in Opp:
        if isinstance(Opp[key],sx.OperatorTerm):
            total_num += len(Opp[key])
        
    return total_num



def variance(Opp_half,psi,batch_sizes):

    Opp_half_psi = Opp_half(psi,**batch_sizes)
    Opp_half_dag_psi = Opp_half.hconj(psi,**batch_sizes)

    psi_Opp_sqrd_psi = 2*Opp_half_dag_psi*Opp_half_psi + Opp_half_dag_psi*Opp_half_dag_psi + Opp_half_psi*Opp_half_psi

    psi_Opp_psi = psi * (Opp_half_psi + Opp_half_dag_psi)

    expOpp = psi_Opp_psi
    varOpp = psi_Opp_sqrd_psi - psi_Opp_psi**2

    return expOpp, varOpp


def density_matrix(state):

    N_spin_orb = state.basis._bitlen
    
    denstiy_mat = np.zeros((N_spin_orb,N_spin_orb))

    for i in range(N_spin_orb):
        for j in range(N_spin_orb):

            n_ij = sx.OperatorTerm(daggers=(1,0),posits=np.array([[i,j]]),coeffs=np.ones(1))

            denstiy_mat[i,j] = state * n_ij(state)

    return denstiy_mat