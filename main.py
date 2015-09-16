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
    
    
def plot_opc(axis,datos,index, legend_label=''):
    
    intervalo = datos['cycle'][0]
    datos2 = datos.set_index(datos[index].cumsum())
    df_mean = pd.rolling_mean(datos2.loc[:,['simd_op', 'scalar_i','v_mem_op', 's_mem_i','lds_op','branch_i']].sum(1)/intervalo, 20)
    
    if legend_label != '':
        df_mean.columns = [legend_label]
    
    df_mean.plot(ax=axis)
    #axis.set_ylim(bottom = 60,top = 140)
    
def plot_mshr_size(axis,datos,index, legend_label=''):
    
    datos2 = datos.set_index(datos[index].cumsum())
    df_mean = pd.rolling_mean(datos2.loc[:,['MSHR_size']], 20)
    #df_mean = datos2.loc[:,['MSHR_size']]
    
        
    df_mean.plot(ax=axis,legend=False)
    
def plot_latencia_memoria(axis,datos,index, legend_label=''):
    
    datos2 = datos.set_index(datos[index].cumsum())
    df_mean = pd.rolling_mean(datos2['mem_acc_lat']/datos2['mem_acc_end'], 20)
    
    if legend_label != '':
        df_mean.columns = [legend_label]
    
    df_mean.plot(ax=axis)
    
def plot_wg_unmapped(axis,datos,index, legend_label=''):
    
    datos2 = datos.set_index(datos[index].cumsum())
    df_mean = pd.rolling_mean(datos2['unmappedWG'], 20)
    #df_mean = datos2['unmappedWG']
    
    if legend_label != '':
        df_mean.columns = [legend_label]
    
    df_mean.plot(ax=axis)
    
def plot_wg_active(axis,datos,index, legend_label=''):
    
    datos2 = datos.set_index(datos[index].cumsum())
    df = pd.DataFrame((datos2['mappedWG'] - datos2['unmappedWG']).cumsum(),columns=[legend_label])
    #df_mean = datos2['unmappedWG']
    
    #if legend_label != '':
    #    df.columns = [legend_label]
    
    df.plot(ax=axis)
    
def plot_opc_accu(axis,datos,index, legend_label=''):
    
    datos2 = datos.set_index(datos[index].cumsum())
    df_mean = pd.rolling_mean(datos2.loc[:,['simd_op', 'scalar_i','v_mem_op', 's_mem_i','lds_op','branch_i']].sum(1).cumsum().div(datos2.loc[:,['cycle']].cumsum()['cycle']), 20)
    #df_mean = datos2['unmappedWG']
    
    if legend_label != '':
        df_mean.columns = [legend_label]
    
    df_mean.plot(ax=axis)
        
def plot_try_points(axis,datos,index, legend_label=''):
    
    aux = []
    
    for i in datos['MSHR_size']:
        if i == 0:
            aux.append(1)
        else:
            aux.append(0)
    
    df= pd.DataFrame(aux)
    
    if legend_label != '':
        df.columns = [legend_label]
    
    df.plot(ax=axis)
    
def axis_config(axis,title = '', ylabel = '', yticks = [], xlabel = '', xticks = []):
    axis.set_title(title)
    plt.legend()
    axis.set_ylabel(ylabel)
    #axis.set_ylim(bottom = 0)
    axis.set_xticks(xticks)
    axis.set_ylabel(xlabel)
    
def ajustar_resolucion(datos,num_puntos = 0, tamanyo_grupo = 0):
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
        
def generar_hoja_calculo(datos):
    
    df = pd.DataFrame()
    
    for bench in sorted_nicely(BENCHMARKS):
        for test in sorted_nicely(datos.keys()):
         
            try:    
            #df = df.append(pd.DataFrame([(test),(bench),(datos[test][bench]['device-spatial-report']['cycle'].sum())],columns=['test','benchmark','cycles']))
                df = df.append(pd.DataFrame([(test , bench,datos[test][bench]['device-spatial-report']['cycle'].sum())],columns=['test','benchmark','cycles']))
            except KeyError as e:
                print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
    df.set_index(['benchmark','test'],inplace=True)

    df.to_excel('/nfs/gap/fracanma/benchmark/resultados/09-13/tunk.xlsx',engine='xlsxwriter')
    
    return
            
    
    
            

if __name__ == '__main__':
    
    sb.set_style("whitegrid")
    #cmap = sb.color_palette("Greys_r", 3)
    #sb.set_palette(cmap, n_colors=3)
    
    #
    BENCHMARKS = ['DwtHaar1D']

    #BENCHMARKS = ['BinarySearch','BinomialOption','BlackScholes','DCT','DwtHaar1D','FastWalshTransform','FloydWarshall','MatrixMultiplication','MatrixTranspose','MersenneTwister','QuasiRandomSequence','RadixSort','RecursiveGaussian','Reduction','ScanLargeArrays','SimpleConvolution']
    test = "tunk"
    bench = 'tunk'
    
    pprint.pprint("hola")
    
    '''dir_experimentos = ["/nfs/gap/fracanma/benchmark/resultados/09-13_nmoesi_mshr16_mshr_dinamico_conL1/",
    "/nfs/gap/fracanma/benchmark/resultados/09-13_nmoesi_mshr32_mshr_dinamico_conL1/",
    "/nfs/gap/fracanma/benchmark/resultados/09-13_nmoesi_mshr256_mshr_dinamico_conL1/","/nfs/gap/fracanma/benchmark/resultados/09-09_nmoesi_mshr32_tunk_conL1/",
    "/nfs/gap/fracanma/benchmark/resultados/09-09_nmoesi_mshr16_tunk_conL1/",
    "/nfs/gap/fracanma/benchmark/resultados/09-09_nmoesi_mshr256_tunk_conL1/"]
    '''
    '''dir_experimentos = ["/nfs/gap/fracanma/benchmark/resultados/09-09_nmoesi_mshr32_tunk_conL1/",
    "/nfs/gap/fracanma/benchmark/resultados/09-09_nmoesi_mshr16_tunk_conL1/",
    "/nfs/gap/fracanma/benchmark/resultados/09-09_nmoesi_mshr256_tunk_conL1/"]
    '''
    
    experimentos = ['09-14_nmoesi_mshr16_mshr_dinamico_conL1','09-14_nmoesi_mshr256_mshr_dinamico_conL1','09-14_nmoesi_mshr32_mshr_dinamico_conL1']

    #experimentos = ['09-15_nmoesi_mshr16_estatico_conL1','09-15_nmoesi_mshr32_estatico_conL1','09-15_nmoesi_mshr64_estatico_conL1','09-15_nmoesi_mshr128_estatico_conL1','09-15_nmoesi_mshr256_estatico_conL1']
        
    f, t = plt.subplots(4,1)
    f.set_size_inches(10, 15)
    f.set_dpi(300)
    
    index_x = 'cycle' #'total_i'
    directorio_resultados = '/nfs/gap/fracanma/benchmark/resultados'
    directorio_salida = '/nfs/gap/fracanma/benchmark/resultados/09-16/'
    dir_experimentos = []
    
    for exp in experimentos:
        dir_experimentos.append(directorio_resultados+'/'+exp)
        
    datos = cargar_datos_sequencial(dir_experimentos,["device-spatial-report","extra-report_ipc"])
    
    comprobar_estructura_datos(datos)
    
    #ajustar_resolucion(datos)


    
    for bench in sorted_nicely(BENCHMARKS):
        for test in sorted_nicely(datos.keys()):
            
            '''try:
                plot_wg_active(t[1][2],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
                axis_config(t[1][2],title = 'wg active')
            except KeyError as e:
                print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
            '''
            try:
                plot_opc(t[0],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
                axis_config(t[0],title = 'OPC')
            except KeyError as e:
                print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
            try:
                plot_mshr_size(t[1],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
                axis_config(t[1], title='mshr size')
            except KeyError as e:
                print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')   
                
            '''try:
                plot_latencia_memoria(t[1][1],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
                axis_config(t[1][1], title = 'latencia de memoria')
            except KeyError as e:
                print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
            try:
                plot_wg_unmapped(t[0][1],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
                axis_config(t[0][1], title = 'WGs finalizados')
            except KeyError as e:
                print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
            ''' 
            try:
                plot_opc_accu(t[2],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
                axis_config(t[2], title = 'opc acumulado')
            except KeyError as e:
                print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
              
            try:
                plot_try_points(t[3],datos[test][bench]['device-spatial-report'], index='cycle', legend_label=test)
                axis_config(t[3], title = 'opc acumulado')
            except KeyError as e:
                print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
            
            
                
        f.savefig(directorio_salida+bench+'_'+'opc.pdf',format='pdf')
        for l in t:
            #for axis in l:
            l.cla()
    
    generar_hoja_calculo(datos)
    
    directorio_salida = "/nfs/gap/fracanma/benchmark/resultados/09-13/"
    
    
    
    
    