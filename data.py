import pandas as pd
import numpy as np

def preprocess():

    metadata = pd.read_csv('data/metadata.csv', usecols = ['sample-id', 'group'])
    print(metadata['group'].value_counts())

    nim_aminoacids = pd.read_csv('data/nim-aminoacids_400.csv', index_col=0)
    nim_aminoacids = nim_aminoacids.drop(nim_aminoacids.iloc[:, 7:], axis = 1)
    nim_aminoacidsD = pd.read_csv('data/nim-aminoacidsD_400.csv', index_col=0)
    nim_sugars = pd.read_csv('data/nim-sugars_400.csv', index_col=0)
    nim_vitamins = pd.read_csv('data/nim-vitamins_400.csv', index_col=0)

    print(f"NIM Amino Acids Shape: {nim_aminoacids.shape}")
    print(f"NIM Amino Acids D Shape: {nim_aminoacidsD.shape}")
    print(f"NIM Sugars Shape: {nim_sugars.shape}")
    print(f"NIM Vitamins Shape: {nim_vitamins.shape}")

    nim = nim_aminoacids.merge(nim_aminoacidsD, how = 'inner', on = 'taxonomy')
    nim = nim.merge(nim_sugars, how = 'inner', on = 'taxonomy')
    nim = nim.merge(nim_vitamins, how = 'inner', on = 'taxonomy')
    print('nim', nim)

    taxonomy = pd.read_csv('data/taxonomy_400.csv', index_col=0)
    taxonomy = taxonomy.replace('%', '', regex = True).astype(np.float64)
    print('taxonomy', taxonomy)

    normal = metadata.loc[metadata.group=='NORMAL']
    normal = normal.drop(['group'], axis = 1)
    normal_ids = normal.to_numpy()

    deviant = metadata.loc[metadata.group=='DEVIANT']
    deviant = deviant.drop(['group'], axis = 1)
    deviant_ids = deviant.to_numpy()


    # Taxonomies of normal and deviant samples
    normal_taxonomy = taxonomy[normal_ids.reshape(-1)]
    deviant_taxonomy = taxonomy[deviant_ids.reshape(-1)]

    # Dataframes to numpy arrays
    normal_taxonomy = normal_taxonomy.to_numpy()
    deviant_taxonomy_np = deviant_taxonomy.to_numpy()

    # Store vhigh / vlow in arrays
    # - with an additional allowance of 5% on both ends
    vlow = np.percentile(normal_taxonomy, 10, axis=1)
    vhigh = np.percentile(normal_taxonomy, 90, axis=1) + 5

    print("Faecalibacterium prausnitzii")
    print(f"v_low : {vlow[0]}\nv_high: {vhigh[0]}")

    def violations(u):
        """ 
        Computes number of ASVs with abundance outside vlow and vhigh (i.e. violation)
        Returns an array of integers representing how many violations each sample has
        """
        
        if len(u.shape) == 1:
            u = u[np.newaxis, :]
            
        assert u.shape[1] == len(vlow) == len(vhigh)
        
        vio_low = u < vlow[np.newaxis, :]
        vio_high = u > vhigh[np.newaxis, :]
        vio = (vio_low | vio_high).astype(np.int32).sum(axis=1)
        
        return vio

    normal_taxonomy_violoation = violations(normal_taxonomy.transpose((1, 0)))
    deviant_taxonomy_violoation = violations(deviant_taxonomy_np.transpose((1, 0)))

    print(f"Normal Taxonomy \n{normal_taxonomy_violoation}\nMean Violoations: {normal_taxonomy_violoation.mean()}\n")
    print(f"Deviant Taxonomy \n{deviant_taxonomy_violoation}\nMean Violoations: {deviant_taxonomy_violoation.mean()}\n")
    
    nim_np = nim.to_numpy()
    return violations, vlow, vhigh, deviant_taxonomy_np, nim_np, deviant_taxonomy, nim