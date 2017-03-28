from CLASS.AGN_Autoencoder import *
from CLASS.MNA_Autoencoder import *
from CLASS.VAE_Autoencoder import *

def main(name = None):
    if name == 'AGN':
        AGN_main()
    elif name == 'MNA':
        MNA_main()
    elif name == 'VAE':
        VAE_main()
    else:
        print('You should choose from the AGN MNA or VAE,please check it.')
        raise Exception('Error...')
        
if __name__ == '__main__':
    main('AGN')
    #main('MNA')
    #main('VAE')