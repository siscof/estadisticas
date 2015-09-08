#!/usr/bin/python3

import time
import multiprocessing
import multiprocessing.pool
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import os
import pprint
import gc
import re
import seaborn as sb

def buscar_string(archivo, palabra):

    linea = None
    for i in archivo:
        if i.startswith(palabra):
            linea = i
            break
    if linea == None:
        return ''
    else:
        return linea.split('=')[1].strip(' \n\t\r')


def cargar_datos(origen_datos,archivos):
    
    datos_cargados = dict()
    pool = multiprocessing.Pool(processes=1)
    pool_result = []
    lista_nombres = []
    for directorio_exp in origen_datos:
        args = []
        
        #nombre la prueba
        aux = directorio_exp.split('/')
        if aux[len(aux)-1] == '' :
            lista_nombres.append(aux[len(aux)-2])
        else:
            lista_nombres.append(aux[len(aux)-1])
            
        for bench in os.listdir(directorio_exp):
            
            args.append((bench,archivos, ["tunk"], directorio_exp))

        pool_result.append(pool.starmap_async(loadworker, args))

    for r in pool_result:
        datos_bench = r.get()
        datos_cargados[lista_nombres[i]]= dict(zip(BENCHMARKS,datos_bench))
        i = i + 1

    return datos_cargados
    
    
def cargar_datos_sequencial(origen_datos,archivos):
    
    datos_cargados = dict()
    i = 0
    lista_nombres = []
    pool_result = []
    for directorio_exp in origen_datos:
        datos_bench = []
        
        #nombre la prueba
        aux = directorio_exp.split('/')
        if aux[len(aux)-1] == '' :
            lista_nombres.append(aux[len(aux)-2])
        else:
            lista_nombres.append(aux[len(aux)-1])
        
        for bench in os.listdir(directorio_exp):
             
            datos_bench.append(loadworker(bench,archivos, ["tunk"], directorio_exp))
        
        datos_cargados[lista_nombres[i]] = dict(zip(BENCHMARKS,datos_bench))
        i = i + 1

    return datos_cargados

def loadworker(bench,archivos, TESTS, directorio_exp):
    dict_general = dict()

    print('cargando -> '+bench)
    

    #dict_general[archivo] = contenedor()
    try:
        pprint.pprint(directorio_exp+'/condor_log/'+bench+'.err')
        f = open(directorio_exp+'/condor_log/'+bench+'.err')
        simend = buscar_string(f, 'SimEnd')
        f.close()
        if 'ContextsFinished' == simend :
            for archivo in archivos :
                dict_general[archivo] = pd.read_csv(directorio_exp+'/'+bench+'/'+archivo,sep = ',', header = 0)

    except Exception as e:
        print('Fallo al cargar el archivo -> '+directorio_exp+'/'+bench+'/'+archivo)
        print(e)
            
    return dict_general
    

if __name__ == '__main__':

    BENCHMARKS = ['BinarySearch','BinomialOption','BlackScholes','DCT','DwtHaar1D','FastWalshTransform','FloydWarshall','MatrixMultiplication','MatrixTranspose','MersenneTwister','QuasiRandomSequence','RadixSort','RecursiveGaussian','Reduction','ScanLargeArrays','SimpleConvolution']
    test = "tunk"
    
    pprint.pprint("hola")
    
    datos = cargar_datos_sequencial(["/nfs/gap/fracanma/benchmark/resultados/09-07_nmoesi_mshr32_tunk_conL1/"],["device-spatial-report","extra-report_ipc"])
    
    pprint.pprint(datos)
    
    #pprint.pprint(datos.keys())
    
    
    