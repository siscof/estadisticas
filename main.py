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
from itertools import cycle, islice

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
                if archivo == 'extra-report_ipc':
                    dict_general[archivo] = pd.read_csv(directorio_exp+'/'+bench+'/'+archivo,sep = ' ', header = 0)
                else:
                    dict_general[archivo] = pd.read_csv(directorio_exp+'/'+bench+'/'+archivo,sep = ',', header = 0)

    except Exception as e:
        print('Fallo al cargar el archivo -> '+directorio_exp+'/'+bench+'/'+archivo)
        print(e)
            
    return dict_general
    
def plot_prediccion_opc(axis,datos,index, legend_label=''):
    
    if datos[index].size > 100:
        tamanyoGrupos = datos[index].size // 100
        datos = datos.groupby(lambda x : (x// tamanyoGrupos) * tamanyoGrupos).sum()
    
    intervalo = datos['cycle'][0]
    datos2 = datos.set_index(datos[index].cumsum())
    #df_mean = pd.DataFrame(pd.rolling_mean(datos2.loc[:,['simd_op', 'scalar_i','v_mem_op', 's_mem_i','lds_op','branch_i']].sum(1)/intervalo, 20))
    df_mean = pd.DataFrame(datos2.loc[:,['simd_op', 'scalar_i','v_mem_op', 's_mem_i','lds_op','branch_i']].sum(1)/intervalo)
 
    
    df_mean.replace(0, np.nan,inplace=True)
    
    '''if legend_label != '':
        df_mean.columns = [legend_label]
    '''
    df_mean.plot(ax=axis,style=['k--'])
    
def plot_opc(axis,datos,index, legend_label=''):
    
    if datos[index].size > 100:
        tamanyoGrupos = datos[index].size // 100
        datos = datos.groupby(lambda x : (x// tamanyoGrupos) * tamanyoGrupos).sum()
    
    intervalo = datos['cycle'][0]
    datos2 = datos.set_index(datos[index].cumsum())
    #df_mean = pd.DataFrame(pd.rolling_mean(datos2.loc[:,['simd_op', 'scalar_i','v_mem_op', 's_mem_i','lds_op','branch_i']].sum(1)/intervalo, 20))
    df_mean = pd.DataFrame(datos2.loc[:,['simd_op', 'scalar_i','v_mem_op', 's_mem_i','lds_op','branch_i']].sum(1)/intervalo)
 
    
    df_mean.replace(0, np.nan,inplace=True)
    
    '''if legend_label != '':
        df_mean.columns = [legend_label]
    '''
    df_mean.plot(ax=axis,legend=False)
    
    #df_mean = pd.DataFrame(datos2.loc[:,['simd_op', 'scalar_i','v_mem_op', 's_mem_i','lds_op','branch_i']].sum(1)/intervalo).groupy
    #df.groupby( lambda x :  (x // tamanyoGrupo) * tamanyoGrupo).sum()
    
def plot_mshr_size(axis,datos,index, legend_label=''):
    
    datos2 = datos.copy()
    for i in datos2.index:
        if np.isnan(datos2['MSHR_size'][i]):
            try:
                datos2.loc[i,'MSHR_size'] = datos2.loc[(i+1),'MSHR_size']
            except Exception as e:
                break
                
    datos2 = datos2.set_index(datos2[index].cumsum())
    df_mean = datos2.loc[:,['MSHR_size']] #.replace(0,np.nan)
    #df_mean = datos2.loc[:,['MSHR_size']]
        
             
    df_mean.interpolate().plot(ax=axis,legend=False)
    
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
    
def plot_latencia_write(axis,datos,index, legend_label=''):
    
    datos2 = ajustar_resolucion(datos)
    datos2 = datos2.set_index(datos2[index].cumsum())
    df_mean = datos2['write_lat']/datos2['write_end']
    
    if legend_label != '':
        df_mean.columns = [legend_label]
    
    df_mean.plot(ax=axis)
    
def plot_latencia_load(axis,datos,index, legend_label=''):

    datos2 = ajustar_resolucion(datos)
    datos2 = datos2.set_index(datos2[index].cumsum())
    df_mean = datos2['load_lat']/datos2['load_end']
    
    if legend_label != '':
        df_mean.columns = [legend_label]
    
    df_mean.plot(ax=axis)
    
def plot_retries(axis_lat,axis_num_retries,datos,index, legend_label=''):

    datos2 = ajustar_resolucion(datos)
    datos2 = datos2.set_index(datos2[index].cumsum())
    df_mean = datos2['cache_retry_lat']/datos2['cache_retry_cont']
    
    if legend_label != '':
        df_mean.columns = [legend_label]
    
    df_mean.plot(ax=axis_lat)
    
    if len(datos.index) > 100:
        tamanyoGrupos = len(datos.index) // 100
        pd.DataFrame(datos.groupby(lambda x : (x// tamanyoGrupos) * tamanyoGrupos).mean()['cache_retry_cont']).plot(ax=axis_num_retries)
        
        
def plot_wavefronts(axis,datos,index, legend_label=''):

   
    if len(datos.index) > 100:
        tamanyoGrupos = len(datos.index) // 100
        pd.DataFrame(datos.groupby(lambda x : (x// tamanyoGrupos) * tamanyoGrupos).mean()['active_wavefronts']).plot(ax=axis)
        return
        
    pd.DataFrame(datos.groupby(lambda x : (x// tamanyoGrupos) * tamanyoGrupos).mean()['active_wavefronts']).plot(ax=axis)
    return
    
def plot_wavefronts_waiting(axis,datos,index, legend_label=''):

   
    if len(datos.index) > 100:
        tamanyoGrupos = len(datos.index) // 100
        pd.DataFrame(datos.groupby(lambda x : (x// tamanyoGrupos) * tamanyoGrupos).mean()['wavefronts_waiting_mem']).plot(ax=axis)
        return
        
    pd.DataFrame(datos.groupby(lambda x : (x// tamanyoGrupos) * tamanyoGrupos).mean()['wavefronts_waiting_mem']).plot(ax=axis)
    return
    
def plot_load_envuelo(axis,datos,index, legend_label=''):

    datos2 = ajustar_resolucion(datos)
    datos2 = datos2.set_index(datos2[index].cumsum())
    df_mean = datos2['vcache_load_start'].cumsum()-datos2['vcache_load_finish'].cumsum()
    
    if legend_label != '':
        df_mean.columns = [legend_label]
    
    df_mean.plot(ax=axis)
    
def plot_write_envuelo(axis,datos,index, legend_label=''):

    datos2 = ajustar_resolucion(datos)
    datos2 = datos2.set_index(datos2[index].cumsum())
    df_mean = datos2['scache_start'].cumsum()-datos2['scache_finish'].cumsum()
    
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
    df_mean = pd.DataFrame(datos2.loc[:,['simd_op', 'scalar_i','v_mem_op', 's_mem_i','lds_op','branch_i']].sum(1).cumsum().div(datos2.loc[:,['cycle']].cumsum()['cycle']),columns=[legend_label])
    #df_mean = datos2['unmappedWG']
    
    if legend_label != '':
        df_mean.columns = [legend_label]
    
    df_mean.plot(ax=axis)

    
def plot_lim_accu(axis, datos,bench,index,legend_label=''):
    
    
    aux = pd.DataFrame()
    for test in datos.keys() :
        df = datos[test][bench]['device-spatial-report']
        df2 = df.set_index(df[index].cumsum())
        
        dato = pd.DataFrame(df2.loc[:,['simd_op', 'scalar_i','v_mem_op', 's_mem_i','lds_op','branch_i']].sum(1).cumsum().div(df2.loc[:,['cycle']].cumsum()['cycle']),columns=[test])
        
        aux = aux.join(pd.DataFrame(dato,columns=[test]), how = 'outer')
        #dato.plot(ax=axis)
        
    
    aux.interpolate().min(1).plot(ax=axis,style=['k--'])
    aux.interpolate().max(1).plot(ax=axis,style=['k--'])
        

        
def plot_train_points(axis,datos,index, legend_label=''):
    a = [0]
    a.extend(datos[index].cumsum())
    pintar = True
    for i in zip(a,datos['MSHR_size'].values):
        if (np.isnan(i[1]) or i[1] == 0) and pintar :
            axis.axvline(x=i[0],linewidth=1, color='k',ls='--')   
            pintar = False
        if not(np.isnan(i[1]) or i[1] == 0):
            pintar = True

def plot_gpu_idle(axis,datos,index, legend_label=''):
    for i in zip(datos[index].cumsum(),datos['gpu_idle'].values):
        if i[1] == 1:
            axis.axvline(x=i[0],linewidth=1, color='r',ls='-')  

def plot_opc_barras(axis,datos,file_input, benchmarks,index, legend_label=''):
    
    df = pd.DataFrame()
    
    opc = pd.DataFrame(index=benchmarks, columns=sorted_nicely(datos.keys()))
    
    for bench in benchmarks :
        for test in sorted_nicely(datos.keys()):
            try:
                opc[test][bench] = datos[test][bench][file_input][['scalar_i','simd_op','s_mem_i','v_mem_op','lds_op']].sum(0).sum() / float(datos[test][bench][file_input]['cycle'].sum())
            except KeyError as e:
                print('tunk')
         
        try:
            opc.ix[bench] = opc.ix[bench] / opc['10-13_nmoesi_mshr32_estatico_conL1'][bench]
        except KeyError as e:
                print('tunk')
        

    opc.plot(ax=axis,kind='bar')

    
def axis_config(axis,title = '', ylabel = '', yticks = [], y_lim = None, xlabel = '', xticks = [], x_lim = None,legend = None):
    axis.set_title(title)
    plt.legend()
    axis.set_ylabel(ylabel)
    axis.set_ylim(bottom = 0)
    if not(y_lim is None):
        axis.set_ylim(bottom = y_lim[0],top = y_lim[1])
        
    if not(x_lim is None):
        axis.set_xlim(x_lim)
    #axis.set_xticks(xticks)
    
    if not(legend is None):
        axis.legend(legend,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    elif not(axis.legend() is None):
        axis.legend().remove()
    
    axis.set_ylabel(xlabel)
    
def plot_distribucion_lat(axis,datos,index,bench, legend_label=''):
    
    #aux = pd.DataFrame().join(pd.DataFrame(datos,columns=[test]), how = 'outer')
    #aux
    
    memEventsLoad = pd.DataFrame(index=experimentos,columns=['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load','wavedfront_sync_load'])
    #memEventsLoad.join(pd.DataFrame(datos2,columns=[test]), how = 'outer')
    for test in experimentos:
        datos2 = datos[test][bench]['extra-report_ipc'].copy()
        datos3 = datos[test][bench]['device-spatial-report']
        #datos2['sg_sync_load'] = (datos3['load_lat']/datos3['load_end']) - (datos2[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum(0).sum(0)/datos2['access_load'].sum(0))
        
        memEventsLoad.ix[test] = pd.DataFrame(datos2[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum(0)/ datos2['access_load'].sum(0),columns=[test]).transpose().ix[test]
        
        memEventsLoad.ix[test]['wavedfront_sync_load'] = datos3['load_lat']/datos3['load_end'] - memEventsLoad.ix[test].sum(1)
                
    memEventsLoad.plot(ax=axis, kind='bar',stacked=True,title='memEventsLoad')
    
def plot_distribucion_lat_continua(datos,bench, legend_label=''):
    
    #aux = pd.DataFrame().join(pd.DataFrame(datos,columns=[test]), how = 'outer')
    #aux
    f_lat, t_lat = plt.subplots(len(experimentos),1)
    f_lat.set_size_inches(10, 15)
    f_lat.set_dpi(300)
    
    memEventsLoad = pd.DataFrame(index=experimentos,columns=['queue_load_miss','lock_mshr_load_miss','lock_dir_load_miss','eviction_load_miss','retry_load_miss','miss_load_miss','finish_load_miss'])
    #memEventsLoad.join(pd.DataFrame(datos2,columns=[test]), how = 'outer')
    for test in zip(experimentos,t_lat):
        ipc = datos[test[0]][bench]['extra-report_ipc'].copy()
        ipc = ipc.set_index('esim_time')
        device = datos[test[0]][bench]['device-spatial-report'].copy()
        device = device.set_index('esim_time')
        device = device.join(ipc[['queue_load_miss','lock_mshr_load_miss','lock_dir_load_miss','eviction_load_miss','retry_load_miss','miss_load_miss','finish_load_miss']].div( ipc.loc[:,'access_load_miss'].values, axis='index'),how = 'outer')
        
        device[['queue_load_miss','lock_mshr_load_miss','lock_dir_load_miss','eviction_load_miss','retry_load_miss','miss_load_miss','finish_load_miss']] = device[['queue_load_miss','lock_mshr_load_miss','lock_dir_load_miss','eviction_load_miss','retry_load_miss','miss_load_miss','finish_load_miss']].interpolate()
        #interpolate(metho)='index'  dropduplicated
        device.set_index(device['cycle'].interpolate().cumsum())[['queue_load_miss','lock_mshr_load_miss','lock_dir_load_miss','eviction_load_miss','retry_load_miss','miss_load_miss','finish_load_miss']].plot(ax=test[1],linewidth=0.1,kind='area',stacked=True,title=test[0])
    
    f_lat.tight_layout()
    f_lat.savefig(directorio_salida+bench+'-memoria-continua.pdf',format='pdf',bbox_inches='tight')
    
    t_lat[0].set_xlim([1000000,2000000])
    t_lat[1].set_xlim([1000000,2000000])
    t_lat[2].set_xlim([1000000,2000000])
    
    f_lat.savefig(directorio_salida+bench+'-memoria-continua-zoom.pdf',format='pdf',bbox_inches='tight')
    
    
    plt.close(f_lat)

    
def ajustar_resolucion(df):
     
    if len(df.index) > 100:
        tamanyoGrupos = len(df.index) // 100
        return pd.DataFrame(df.groupby(lambda x : (x// tamanyoGrupos) * tamanyoGrupos).sum())
        
    return df
    
    
    
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
    cmap = sb.color_palette("Set2", 15)
    sb.set_palette(cmap, n_colors=15)
    
    mpl.rcParams['lines.linewidth'] = 2
    
    
    #BENCHMARKS = ['DwtHaar1D']
    
    #BENCHMARKS = ['BinarySearch','BinomialOption','BlackScholes','DCT','DwtHaar1D','FastWalshTransform','FloydWarshall','MatrixMultiplication','MatrixTranspose','MersenneTwister','QuasiRandomSequence','RadixSort','RecursiveGaussian','Reduction','ScanLargeArrays','SimpleConvolution','SobelFilter']

    BENCHMARKS = ['BlackScholes','DwtHaar1D','FastWalshTransform','MatrixMultiplication','MersenneTwister','QuasiRandomSequence','SimpleConvolution','SobelFilter']
    test = "tunk"
    bench = 'tunk'
    
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
    
    #experimentos = ['09-14_nmoesi_mshr16_mshr_dinamico_conL1','09-14_nmoesi_mshr256_mshr_dinamico_conL1','09-14_nmoesi_mshr32_mshr_dinamico_conL1']
    
    #experimentos = ['09-16_nmoesi_mshr16_estatico_scalar8_conL1','09-16_nmoesi_mshr32_estatico_scalar8_conL1','09-16_nmoesi_mshr64_estatico_scalar8_conL1','09-16_nmoesi_mshr128_estatico_scalar8_conL1','09-16_nmoesi_mshr256_estatico_scalar8_conL1']
    
    #experimentos = ['09-16_nmoesi_mshr16_din_scalar8_conL1','09-16_nmoesi_mshr32_din_scalar8_conL1','09-16_nmoesi_mshr64_din_scalar8_conL1','09-16_nmoesi_mshr128_din_scalar8_conL1','09-16_nmoesi_mshr256_din_scalar8_conL1','09-17_nmoesi_mshr32_B30000_conL1','09-17_nmoesi_mshr32_B_conL1']

    #experimentos = ['09-15_nmoesi_mshr16_estatico_conL1','09-15_nmoesi_mshr32_estatico_conL1','09-15_nmoesi_mshr64_estatico_conL1','09-15_nmoesi_mshr128_estatico_conL1','09-15_nmoesi_mshr256_estatico_conL1']
    
    #experimentos = ['09-17_nmoesi_mshr16_A_conL1','09-17_nmoesi_mshr32_B30000_conL1','09-17_nmoesi_mshr64_A_conL1','09-17_nmoesi_mshr128_B_conL1','09-17_nmoesi_mshr256_A_conL1']
        
    #experimentos = ['09-16_nmoesi_mshr16_estatico_scalar8_conL1','09-16_nmoesi_mshr32_estatico_scalar8_conL1','09-16_nmoesi_mshr64_estatico_scalar8_conL1','09-16_nmoesi_mshr128_estatico_scalar8_conL1','09-16_nmoesi_mshr256_estatico_scalar8_conL1','09-17_nmoesi_mshr32_B30000_conL1'] 
    
    #experimentos = ['09-28_nmoesi_mshr32_b5000_conL1','09-28_nmoesi_mshr32_b10000_conL1','09-28_nmoesi_mshr32_b20000_conL1','09-28_nmoesi_mshr32_b30000_conL1'] 
    
    #experimentos = ['09-30_nmoesi_mshr32_5000_conL1','09-30_nmoesi_mshr32_10000_conL1','09-30_nmoesi_mshr32_20000_conL1','09-30_nmoesi_mshr32_30000_conL1']    
    #experimentos = ['09-30_nmoesi_mshr32_b5000_conL1','09-30_nmoesi_mshr32_b10000_conL1','09-30_nmoesi_mshr32_b20000_conL1']
    #experimentos = ['10-02_nmoesi_mshr32_20000_conL1','10-01_nmoesi_mshr32_lat300estatico_conL1','10-05_nmoesi_mshr32_control_mshr_32_trucado_conL1']
    #experimentos = ['10-02_nmoesi_mshr32_20000_conL1','10-01_nmoesi_mshr32_lat300estatico_conL1','10-05_nmoesi_mshr32_control_mshr_32_trucado_conL1','10-07_nmoesi_mshr32_dinamico_20000_conL1','10-07_nmoesi_mshr32_estatico_20000_conL1','10-07_nmoesi_mshr32_dinamico_trucado_conL1']
    
    #experimentos = ['10-07_nmoesi_mshr32_dinamico_20000_conL1','10-07_nmoesi_mshr32_estatico_20000_conL1','10-07_nmoesi_mshr32_dinamico_trucado_conL1','10-07_nmoesi_mshr32_retry_conL1']
    
    experimentos = ['10-08_nmoesi_mshr32_dinamico20000_conL1','10-08_nmoesi_mshr32_trucado20000_conL1','10-13_nmoesi_mshr32_dinamico_conL1','10-13_nmoesi_mshr32_dinamico_forzado_conL1','10-13_nmoesi_mshr32_estatico_conL1']
    
    experimentos = ['10-19_nmoesi_mshr16_lat100_estatico_conL1','10-19_nmoesi_mshr32_lat100_estatico_conL1','10-19_nmoesi_mshr128_lat100_estatico_conL1']
    
    experimentos = ['10-23_nmoesi_mshr16_test_conL1','10-23_nmoesi_mshr32_test_conL1','10-23_nmoesi_mshr128_test_conL1']
    
    
    #experimentos = ['10-13_nmoesi_mshr8_estatico8_conL1','10-13_nmoesi_mshr32_estatico_conL1']


    
    #legend = ['dinamico_anterior','trucado_anterior','dinamico_nuevo','trucado_nuevo','estatico']
    
    legend = ['estatico_mshr16', 'estatico_mashr32','estatico_mshr128']
    
    index_x = 'cycle' #'total_i'
    directorio_resultados = '/nfs/gap/fracanma/benchmark/resultados'
    directorio_salida = '/nfs/gap/fracanma/benchmark/resultados/10-20_distribucion_latencia/'
    dir_experimentos = []
    
    for exp in experimentos:
        dir_experimentos.append(directorio_resultados+'/'+exp)
        
    datos = cargar_datos_sequencial(dir_experimentos,["device-spatial-report","extra-report_ipc"])
        
    #df_prediccion = cargar_datos_sequencial([directorio_resultados+'/10-05_nmoesi_mshr32_predicion_opc_20000_conL1'],["device-spatial-report","extra-report_ipc"])   
    
    dir_estaticos = []
    
    #experimentos_baseline = ['09-16_nmoesi_mshr16_estatico_scalar8_conL1','09-16_nmoesi_mshr32_estatico_scalar8_conL1','09-16_nmoesi_mshr64_estatico_scalar8_conL1','09-16_nmoesi_mshr128_estatico_scalar8_conL1']
    
    '''experimentos_baseline = ['10-01_nmoesi_mshr32_lat300estatico_conL1']
    #experimentos_baseline = ['10-13_nmoesi_mshr32_estatico_conL1']
    
    
    for exp in experimentos_baseline:
        dir_estaticos.append(directorio_resultados+'/'+exp)  
    
    '''
    
    prestaciones_estatico = cargar_datos_sequencial(dir_estaticos, ["device-spatial-report"])
    
    comprobar_estructura_datos(datos)
    
    #ajustar_resolucion(datos)

    for bench in sorted_nicely(BENCHMARKS):
        test = 0
        
            
        f, t = plt.subplots(4,1)
        f.set_size_inches(10, 15)
        f.set_dpi(300)
    
        f2, t2 = plt.subplots(9,1)
        f2.set_size_inches(10, 15)
        f2.set_dpi(300)
        
        f3, t3 = plt.subplots()
        f3.set_size_inches(10, 15)
        f3.set_dpi(300)
        
        for test in experimentos:#sorted_nicely(datos.keys()): 
        #,sorted_nicely(prestaciones_estatico.keys())):
            
            '''try:
                plot_wg_active(t[1][2],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
                axis_config(t[1][2],title = 'wg active')
            except KeyError as e:
                print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
            '''
            try:
                plot_opc(t[0],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
            except KeyError as e:
                print('WARNING1: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
            try:
                plot_mshr_size(t[1],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
                
            except KeyError as e:
                print('WARNING2: KeyError in datos['+test+']['+bench+'][device-spatial-report]')   
                
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
            except KeyError as e:
                print('WARNING3: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
            try:
                plot_write_envuelo(t2[0],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
            except KeyError as e:
                print('WARNING4: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
            
            try:
                plot_latencia_write(t2[1],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
            except KeyError as e:
                print('WARNING5: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
            try:
                plot_load_envuelo(t2[2],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
            except KeyError as e:
                print('WARNING6: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
            
            try:
                plot_latencia_load(t2[3],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
            except KeyError as e:
                print('WARNING7: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
            
            try:
                plot_retries(t2[4],t2[5],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
            except KeyError as e:
                print('WARNING8: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
            try:
                plot_opc_accu(t2[6],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
                
            except KeyError as e:
                print('WARNING9: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
            try:
                plot_wavefronts(t2[7],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
            except KeyError as e:
                print('WARNING10: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
            
            try:
                plot_wavefronts_waiting(t2[8],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
            except KeyError as e:
                print('WARNING11: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
        try:
            plot_distribucion_lat(t3,datos,index_x,bench, 'tunk')
        except KeyError as e:
            print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')  
            
        try:   
             plot_distribucion_lat_continua(datos,bench, legend_label='')
        except KeyError as e:
            print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
                
        for test in sorted_nicely(prestaciones_estatico.keys()): 
            try:
                plot_opc_accu(t[3],prestaciones_estatico[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
                
            except KeyError as e:
                print('WARNING12: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
              
        '''try:
            #plot_prediccion_opc(t[0],df_prediccion['10-05_nmoesi_mshr32_predicion_opc_20000_conL1'][bench]['device-spatial-report'],index=index_x)
            plot_train_points(t[1],datos[experimentos[0]][bench]['device-spatial-report'], index=index_x, legend_label=test)
            plot_train_points(t[0],datos[experimentos[0]][bench]['device-spatial-report'], index=index_x, legend_label=test)
            #plot_gpu_idle(t[0],datos[experimentos[0]][bench]['device-spatial-report'], index=index_x, legend_label=test)
            
            #plot_lim_accu(t[2], prestaciones_estatico,bench,index=index_x,legend_label=test)
            #axis_config(t[2], title = 'training points')
        except KeyError as e:
            print('WARNING13: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
        '''
            
        if not os.path.exists(directorio_salida):
            os.mkdir(directorio_salida)
            
        axis_config(t[0],title = 'OPC')
        axis_config(t[1], title='mshr size',y_lim = [0,256])
        axis_config(t[2], title = 'opc acumulado')
        axis_config(t[3], title = 'opc acumulado estaticos')
            
        #plot_lim_accu(t[1],prestaciones_estatico,bench, index_x)
        f.suptitle(bench, fontsize=25)
        f.tight_layout()
        f.savefig(directorio_salida+bench+'.pdf',format='pdf',bbox_inches='tight')
       
        
        
        axis_config(t2[0],title = 'store en vuelo',legend = legend)
        axis_config(t2[1], title= 'latencia store')
        axis_config(t2[2], title = 'load en vuelo')
        axis_config(t2[3], title = 'latencia load')
        axis_config(t2[4], title = 'retry latency')
        axis_config(t2[5], title = 'retries')
        axis_config(t2[6], title = 'OPC acumulado')
        axis_config(t2[7], title = 'wavefronts')
        axis_config(t2[8], title = 'wavefronts waiting mem')
            
        #plot_lim_accu(t[1],prestaciones_estatico,bench, index_x)
        f2.suptitle(bench, fontsize=25)
        f2.tight_layout()
        f2.savefig(directorio_salida+bench+'_latencias.pdf',format='pdf',bbox_inches='tight')
        plt.close(f2)
        
        #zoom
        if bench == BENCHMARKS[1]:
            axis_config(t[0],title = 'OPC',x_lim = [0,2000000])
            axis_config(t[1], title='mshr size',y_lim = [0,256],x_lim = [0,2000000])
            axis_config(t[2], title = 'opc acumulado',x_lim = [0,2000000])
            axis_config(t[3], title = 'opc acumulado estaticos',x_lim = [0,2000000])
        f.tight_layout()
        f.savefig(directorio_salida+bench+'-ZOOM.pdf',format='pdf',bbox_inches='tight')
        plt.close(f)
        
        axis_config(t3,title = 'latencias',ylabel='ciclos',legend = legend)
        f3.tight_layout()
        f3.savefig(directorio_salida+bench+'-memoria.pdf',format='pdf',bbox_inches='tight')
        plt.close(f3)
        '''for l in t:
            #for axis in l:
            l.cla()
            
        for l in t2:
            #for axis in l:
            l.cla()
        '''
    f, t = plt.subplots() 
    datos.update(prestaciones_estatico)       
    plot_opc_barras(t,datos,'device-spatial-report', BENCHMARKS,index_x, legend_label='')
    axis_config(t, title='speedup',y_lim = [0.6,1.6])
    t.legend(legend,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    f.savefig(directorio_salida+'opc.pdf',format='pdf',bbox_inches='tight')
    
    #generar_hoja_calculo(datos)
    
    
    
    
    