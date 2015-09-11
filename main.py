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

def sorted_nicely( l ):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

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
            
        for bench in  sorted_nicely(os.listdir(directorio_exp)):
            if bench in BENCHMARKS :
                args.append((bench,archivos, ["Ttunk"], directorio_exp))

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
        try:
            datos_bench = []
            
            #nombre la prueba
            aux = directorio_exp.split('/')
            if aux[len(aux)-1] == '' :
                lista_nombres.append(aux[len(aux)-2])
            else:
                lista_nombres.append(aux[len(aux)-1])
            
            for bench in  sorted_nicely(os.listdir(directorio_exp)):
                if bench in BENCHMARKS :
                    datos_bench.append(loadworker(bench,archivos, ["Ttunk"], directorio_exp))
            
            datos_cargados[lista_nombres[i]] = dict(zip(BENCHMARKS,datos_bench))
            i = i + 1
        except Exception as e:
            print('Fallo al cargar el experimento -> '+directorio_exp)
            print(e)

    return datos_cargados

def loadworker(bench,archivos, TESTS, directorio_exp):
    dict_general = dict()

    #dict_general[archivo] = contenedor()
    try:
        pprint.pprint(directorio_exp+'/'+bench+'.err')
        f = open(directorio_exp+'/condor_log/'+bench+'.err')
        simend = buscar_string(f, 'SimEnd')
        f.close()
        if 'ContextsFinished' == simend :
            for archivo in archivos :
                pprint.pprint(directorio_exp+'/'+bench+'/'+archivo)
                dict_general[archivo] = pd.read_csv(directorio_exp+'/'+bench+'/'+archivo,sep = ',', header = 0)

    except Exception as e:
        print('Fallo al cargar el archivo -> '+directorio_exp+'/'+bench+'/'+archivo)
        print(e)
            
    return dict_general
    
    
def plot_opc(axis,datos,index):
    
    intervalo = datos['cycle'][0]
    datos2 = datos.set_index(datos[index].cumsum())
    df_mean = pd.rolling_mean(datos2.loc[:,['simd_op', 'scalar_i','v_mem_op', 's_mem_i','lds_op']].sum(1)/intervalo, 20)
    df_mean.plot(ax=axis)
    
def plot_mshr_size(axis,datos,index):
    
    datos2 = datos.set_index(datos[index].cumsum())
    df_mean = pd.rolling_mean(datos2.loc[:,['MSHR_size']], 20)
    df_mean.plot(ax=axis)
    
def plot_latencia_memoria(axis,datos,index):
    
    datos2 = datos.set_index(datos[index].cumsum())
    df_mean = pd.rolling_mean(datos2['mem_acc_lat']/datos2['mem_acc_end'], 20)
    df_mean.plot(ax=axis)
    
def plot_wg_unmapped(axis,datos,index):
    
    datos2 = datos.set_index(datos[index].cumsum())
    df_mean = pd.rolling_mean(datos2['unmappedWG'], 20)
    df_mean.plot(ax=axis)
    
def axis_config(axis,title = '', ylabel = '', yticks = [], xlabel = '', xticks = []):
    axis.set_title(title)
    plt.legend()
    axis.set_ylabel(ylabel)
    axis.set_ylim(bottom = 0)
    axis.set_xticks(xticks)
    axis.set_ylabel(xlabel)
    
def ajustar_resolucion(datos,num_puntos = 0, tamanyo_grupo = 0)
    for test in  datos:
        for bench in  test:
            for df in bench:
                if type(df) == pd.DataFrame :
                    if tamanyo_grupo != 0 :
                        df = df.groupby( lambda x :  (x // tamanyoGrupo) * tamanyoGrupo).sum()
    
    
    
def comprobar_estructura_datos(datos):
    for key in  list(datos):
        if type(datos[key]) is dict:
            dict_vacio = comprobar_estructura_datos(datos[key])
            if dict_vacio :
                datos.pop(key, None)
            
    if bool(datos) :
        return False
    else:
        return True
            
    
    
            

if __name__ == '__main__':

    BENCHMARKS = ['BinarySearch','BinomialOption','BlackScholes','DCT','DwtHaar1D','FastWalshTransform','FloydWarshall','MatrixMultiplication','MatrixTranspose','MersenneTwister','QuasiRandomSequence','RadixSort','RecursiveGaussian','Reduction','ScanLargeArrays','SimpleConvolution']
    test = "tunk"
    bench = 'tunk'
    
    pprint.pprint("hola")
    
    dir_experimentos = ["/nfs/gap/fracanma/benchmark/resultados/09-09_nmoesi_mshr32_tunk_conL1/",
    "/nfs/gap/fracanma/benchmark/resultados/09-09_nmoesi_mshr16_tunk_conL1/",
    "/nfs/gap/fracanma/benchmark/resultados/09-09_nmoesi_mshr256_tunk_conL1/"]
    
    datos = cargar_datos_sequencial(dir_experimentos,["device-spatial-report","extra-report_ipc"])
    
    comprobar_estructura_datos(datos)
    
    ajustar_resolucion(datos)

    f, t = plt.subplots(2,2)
    f.set_size_inches(15, 10)
    f.set_dpi(300)
    
    for bench in BENCHMARKS:
        for test in datos.keys():
            try:
                plot_opc(t[0][0],datos[test][bench]['device-spatial-report'], index='total_i')
                axis_config(t[0][0],title = 'OPC')
            except KeyError as e:
                print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
            try:
                plot_mshr_size(t[1][0],datos[test][bench]['device-spatial-report'], index='total_i')
                axis_config(t[1][0], title='mshr size')
            except KeyError as e:
                print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')   
                
            try:
                plot_latencia_memoria(t[1][1],datos[test][bench]['device-spatial-report'], index='total_i')
                axis_config(t[1][1], title = 'latencia de memoria')
            except KeyError as e:
                print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
            try:
                plot_wg_unmapped(t[0][1],datos[test][bench]['device-spatial-report'], index='total_i')
                axis_config(t[0][1], title = 'WGs finalizados')
            except KeyError as e:
                print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
        f.savefig('/home/sisco/workspace/pruebas/'+bench+'_'+'opc.eps',format='eps')
        for l in t:
            for axis in l:
                axis.cla()
    
    directorio_salida = "/nfs/gap/fracanma/benchmark/resultados/09-07_nmoesi_mshr32_tunk_conL1/"
    
    
    
    
    