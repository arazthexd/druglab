from typing import List
import itertools

import numpy as np

from druglab.pharm import PharmBittifier, PharmProfile

class PharmFingerprinter:
    def __init__(self,
                 type_bittifier: PharmBittifier | List[PharmBittifier] = None,
                 pairs1_bittifier: PharmBittifier | List[PharmBittifier] = None,
                 pairs2_bittifier: PharmBittifier | List[PharmBittifier] = None):
        if isinstance(type_bittifier, PharmBittifier):
            type_bittifier = [type_bittifier]
        if isinstance(pairs1_bittifier, PharmBittifier):
            pairs1_bittifier = [pairs1_bittifier]
        if isinstance(pairs2_bittifier, PharmBittifier):
            pairs2_bittifier = [pairs2_bittifier]

        self.tb = type_bittifier
        self.pb1 = pairs1_bittifier
        self.pb2 = pairs2_bittifier
    
    def fingerprint(self,
                    profile: PharmProfile,
                    fpsize: int = 1024,
                    merge_confs: bool = False):
        
        bits_list: List[List[np.ndarray]] = []
        maxbit_list: List[List[int]] = []

        if self.tb is not None:
            c = 0
            b = []
            for tb in self.tb:
                bits, maxbit = tb.bittify(profile.tys, 
                                          profile.tyids, 
                                          profile.n_tyids)
                print(maxbit)
                bits = bits + c
                c += maxbit
                b.append(bits)
            
            bits_list.append(b)
            maxbit_list.append(c)
        
        if self.pb1 is not None:
            c = 0
            b = []
            for pb in self.pb1:
                print(pb)
                bits, maxbit = pb.bittify(profile.pair1tys,
                                          profile.pair1tyids,
                                          profile.n_pair1tyids,
                                          profile.pair1vals)
                print(maxbit)
                bits = bits + c
                c += maxbit
                b.append(bits)
                print(b, c)
            
            bits_list.append(b)
            maxbit_list.append(c)
        
        if self.pb2 is not None:
            c = 0
            b = []
            for pb in self.pb2:
                bits, maxbit = pb.bittify(profile.pair2tys,
                                          profile.pair2tyids,
                                          profile.n_pair2tyids,
                                          profile.pair2vals)
                bits = bits + c
                c += maxbit
                b.append(bits)
            
            bits_list.append(b)
            maxbit_list.append(c)

        print(maxbit_list)

        if merge_confs:
            fp = np.zeros((fpsize, ))
        else:
            fp = [np.zeros((fpsize, )) 
                  for _ in range(len(profile.subborderids))]
        
        indexer = np.flip(np.array(maxbit_list))
        indexer = indexer.cumprod().astype(np.uint64)
        indexer = np.flip(indexer)
        indexer[:-1] = indexer[1:]
        indexer[-1] = 1

        for bits_choice in itertools.product(*bits_list):
            bits = np.stack(bits_choice, axis=-1)
            bits = (bits * indexer).sum(axis=-1) % fpsize
            bits = bits.astype(np.uint64)
            
            if merge_confs:
                fp[bits] = 1
            
            else:
                for i, (j, k) in enumerate(zip([0]+profile.subborderids, 
                                               profile.subborderids)):
                    fp[i][bits[j:k]] = 1
        
        return fp


        

        # if self.pb1 is not None:
        #     bit_choices.append(list())
        #     maxbit_choices.append(list())
        #     for pb in self.pb1:
        #         bits, maxbit = pb.bittify(profile.pair1tys, 
        #                                   profile.pair1tyids, 
        #                                   profile.n_pair1tyids,
        #                                   profile.pair1vals)
        #         bit_choices[-1].append(bits)
        #         maxbit_choices[-1].append(maxbit)

        # if self.pb2 is not None:
        #     bit_choices.append(list())
        #     maxbit_choices.append(list())
        #     for pb in self.pb2:
        #         bits, maxbit = pb.bittify(profile.pair2tys, 
        #                                   profile.pair2tyids, 
        #                                   profile.n_pair2tyids,
        #                                   profile.pair2vals)
        #         bit_choices[-1].append(bits)
        #         maxbit_choices[-1].append(maxbit)

        # bit_choices = list(itertools.product(*bit_choices))
        # maxbit_choices = list(itertools.product(*maxbit_choices))

        # if merge_confs:
        #     fp = np.zeros((fpsize, ))
        # else:
        #     fp = [np.zeros((fpsize, )) 
        #           for _ in range(len(profile.subborderids))]
            
        # for bits_list, maxbit_list in zip(bit_choices, maxbit_choices):
        #     maxbit = np.flip(np.array(maxbit_list))
        #     maxbit = maxbit.cumprod().astype(np.uint64)
        #     maxbit = np.flip(maxbit)
        #     maxbit[0:-1] = maxbit[1:]
        #     maxbit[-1] = 1

        #     bits = np.stack(bits_list, axis=-1)
        #     bits = (bits * maxbit).sum(axis=-1, dtype=np.uint64) % fpsize

        #     if merge_confs:
        #         fp[bits] = 1
            
        #     else:
        #         for i, (j, k) in enumerate(zip([0]+profile.subborderids, 
        #                                        profile.subborderids)):
        #             fp[i][bits[j:k]] = 1
        
        # return fp


            
