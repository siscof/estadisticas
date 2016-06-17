'''
Created on 16 de jun. de 2016

@author: sisco
'''


import time
import multiprocessing
import multiprocessing.pool
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import os
import sys
import pprint
import gc
import re
import seaborn as sb
from itertools import cycle, islice
import math


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
        #pprint.pprint(directorio_exp+'/'+bench+'.err')
        f = open(directorio_exp+'/condor_log/'+bench+'.err')
        simend = buscar_string(f, 'SimEnd')
        f.close()
        if 'ContextsFinished' == simend or 'SouthernIslandsMaxInst' == simend  :
            for archivo in archivos :
                #pprint.pprint(directorio_exp+'/'+bench+'/'+archivo)
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
    
def plot_latencia_memoria_VMB(datos,directorio_salida, legend_label=''):
    
    f, t = plt.subplots()
    f.set_size_inches(10, 15)
    f.set_dpi(300)
    
    f2, t2 = plt.subplots()
    f2.set_size_inches(10, 15)
    f2.set_dpi(300)
    
    df_resumen = pd.DataFrame()

    for bench in sorted_nicely(BENCHMARKS):
        try:
            df_bench = pd.DataFrame()
            for test in sorted_nicely(datos.keys()):
                try:
                    datos2 = datos[test][bench]['device-spatial-report'][['uop_load_lat','uop_load_end','uop_load_vmb_lat','uop_load_mm_lat','cycle']]
                    datos3 = datos2.sum()
                    datos4 = datos[test][bench]['extra-report_ipc'][['Coalesces_gpu','Coalesces_L1']].sum().sum()
                    datos4 = datos[test][bench]['extra-report_ipc']['accesos_L1'].sum().sum()
                    #datos2 = datos2.set_index('cycle')
                    #df_result = pd.DataFrame([],index=datos2.index, columns=['VMB','L1-L2','MM'])
                
                    #df_result['VMB'] = datos2['uop_load_vmb_lat']/datos2['uop_load_end']
                    #df_result['MM'] = datos2['uop_load_mm_lat']/datos2['uop_load_end']
                    #df_result['L1-L2'] = (datos2['uop_load_lat']/datos2['uop_load_end']) - df_result['VMB'] - df_result['MM']
                    VMB = 0
                    MM = 0
                    L1_L2 = 0
                    VMB = datos3['uop_load_vmb_lat']/datos3['uop_load_end']
                    MM = datos3['uop_load_mm_lat']/datos3['uop_load_end']
                    L1_L2 = (datos3['uop_load_lat']/datos3['uop_load_end']) - VMB - MM
                    df_resumen = df_resumen.append(pd.DataFrame([[bench,VMB,L1_L2,MM,datos4]],index=[test],columns=['BENCHMARKS','VMB','L1-L2','MM','accesses_L1']))
                    df_bench = df_bench.append(pd.DataFrame([[VMB,L1_L2,MM]],index=[test],columns=['VMB','L1-L2','MM']))
                    
                    #if legend_label != '':
                    #    df_mean.columns = [legend_label]
                    #    datos2 = datos2.set_index(datos2['cycle'].cumsum())
                    #   (datos2[['uop_load_lat','uop_load_vmb_lat','uop_load_mm_lat']].div(datos2['uop_load_end'], axis='index')).plot(ax=t,kind='area',stacked=True,title='memory latency')
                except Exception as e:
                    df_resumen = df_resumen.append(pd.DataFrame([[bench,0,0,0,0]],index=[test],columns=['BENCHMARKS','VMB','L1-L2','MM','accesses_L1']))
                    pass
                
                #f.tight_layout()
                #if not os.path.exists(directorio_salida+'/'+test):
                #    os.mkdir(directorio_salida+'/'+test)
                #f.savefig(directorio_salida+'/'+test+'/'+bench+'-memory-latency.pdf',format='pdf',bbox_inches='tight')
                #for l in t.ravel():
                #t.cla()
        
            df_bench.plot(ax=t2,kind='bar',stacked=True,title=bench+' memory latency')
            f2.tight_layout()
            dir = directorio_salida+'/bench-memory-latency-stacked'
            if not os.path.exists(dir):
                os.mkdir(dir)
            f2.savefig(dir+'/'+bench+'.pdf',format='pdf',bbox_inches='tight')
            #for l in t.ravel():
            t2.cla()
            plt.close(f2)
        except Exception as e:
            pass
        
    df_resumen.to_excel(directorio_salida+'/resumen_memoria.xlsx',engine='xlsxwriter')
    plt.close(f)
        
    
def plot_wg_unmapped(axis,datos,index, legend_label=''):
    
    datos2 = datos.set_index(datos[index].cumsum())
    #df_mean = pd.rolling_mean(datos2['unmappedWG'], 20)
    df_mean = datos2['unmappedWG']
    
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
   
def plot_wait_for_mem(axis,datos,index, legend_label=''):    
    if datos[index].size > 100:
        tamanyoGrupos = datos[index].size // 100
        datos = datos.groupby(lambda x : (x// tamanyoGrupos) * tamanyoGrupos).sum()
    
    datos2 = datos.set_index(datos[index].cumsum())
    #df_mean = pd.DataFrame(pd.rolling_mean(datos2.loc[:,['simd_op', 'scalar_i','v_mem_op', 's_mem_i','lds_op','branch_i']].sum(1)/intervalo, 20))
    df_mean = pd.DataFrame(datos2['wait_for_mem_time']/datos2['wait_for_mem_counter'])
 
    
    df_mean.replace( np.nan, 0,inplace=True)
    
    '''if legend_label != '':
        df_mean.columns = [legend_label]
    '''
    df_mean.plot(ax=axis,legend=False)

def plot_opc_barras(axis,datos,file_input, benchmarks,index, legend_label=''):
    
    df = pd.DataFrame()
    
    opc = pd.DataFrame(index=benchmarks, columns=sorted_nicely(datos.keys()))
    
    for bench in benchmarks :
        for test in sorted_nicely(datos.keys()):
            try:
                opc[test][bench] = datos[test][bench][file_input][['scalar_i','simd_op','s_mem_i','v_mem_op','lds_op']].sum(0).sum() / float(datos[test][bench][file_input]['cycle'].sum())
            except KeyError as e:
                print('tunk1')
         
        try:
            i = 0
            while math.isnan(opc[sorted_nicely(datos.keys())[i]][bench]) and i < len(datos.keys()) :
                i = i + 1
            
            opc.ix[bench] = opc.ix[bench] / opc[sorted_nicely(datos.keys())[i]][bench]
        except Exception as e:
                print('tunk2')
        

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
    
    labels = []
    
    df_index = []
    for i in sorted_nicely(datos.keys()): 
        df_index.append(i) 
        #df_index.append(i+'-ponderada') 
        df_index.append(i+'-wait_for_mem') 
    
    #memEventsLoad = pd.DataFrame(index=df_index,columns=['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load','miss_lat','hit_lat','wait_for_mem_latency'])
    memEventsLoad = pd.DataFrame(index=df_index,columns=['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load','wait_for_mem_latency'])
    hitratio = pd.DataFrame(index=df_index,columns=['hit ratio'])
    #memEventsLoad.join(pd.DataFrame(datos2,columns=[test]), how = 'outer')
    for test in sorted_nicely(datos.keys()):
        try:
            datos2 = datos[test][bench]['extra-report_ipc'].copy()
            datos3 = datos[test][bench]['device-spatial-report'].copy()
            #datos2['sg_sync_load'] = (datos3['load_lat']/datos3['load_end']) - (datos2[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum(0).sum(0)/datos2['access_load'].sum(0))
            
            
            df_miss = datos2[['queue_load_miss','lock_mshr_load_miss','lock_dir_load_miss','eviction_load_miss','retry_load_miss','miss_load_miss','finish_load_miss','access_load_miss']]
            df_hit = datos2[['queue_load_hit','lock_mshr_load_hit','lock_dir_load_hit','eviction_load_hit','retry_load_hit','miss_load_hit','finish_load_hit','access_load_hit']]
            df_critical_miss = datos2[['queue_load_critical_miss','lock_mshr_load_critical_miss','lock_dir_load_critical_miss','eviction_load_critical_miss','retry_load_critical_miss','miss_load_critical_miss','finish_load_critical_miss','access_load_critical_miss']]
            df_critical_hit = datos2[['queue_load_critical_hit','lock_mshr_load_critical_hit','lock_dir_load_critical_hit','eviction_load_critical_hit','retry_load_critical_hit','miss_load_critical_hit','finish_load_critical_hit','access_load_critical_hit']]
            
            col = ['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load','access_load']
            df_miss.columns = col
            df_hit.columns = col
            df_critical_miss.columns = col
            df_critical_hit.columns = col
            
            df_sum = df_critical_miss + df_critical_hit        
            
            hitratio.loc[test,'hit ratio'] = df_critical_hit['access_load'].sum() / df_sum['access_load'].sum()
            
            #df_sum = df_critical_miss + df_critical_hit
            #memEventsLoad.ix[test] = pd.DataFrame([df_sum[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum(0)/ df_sum['access_load'].sum(0)],index=[test]).ix[test]
            
            memEventsLoad.ix[test] = pd.DataFrame([(df_critical_miss + df_miss)[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum(0)/ (df_critical_miss + df_miss)['access_load'].sum(0)],index=[test]).ix[test]
            
            thit = df_critical_hit[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum(0).sum(0)/ df_critical_miss['access_load'].sum(0)
            tmiss = df_critical_miss[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum(0).sum(0)/ df_critical_miss['access_load'].sum(0)
            #memEventsLoad.ix[test+'-ponderada'] = pd.DataFrame([(tmiss * (1-hitratio.ix[test]['hit ratio']),thit * hitratio.ix[test]['hit ratio']) ],index=[test+'-ponderada'],columns=['miss_lat','hit_lat']).ix[test+'-ponderada']
            
    #        memEventsLoad.ix[test+'-wait_for_mem'] = pd.DataFrame([datos3['wait_for_mem_time'].sum(0)/ datos3['wait_for_mem_counter'].sum(0)],index=[test+'-wait_for_mem'],columns=['wait_for_mem_latency']).ix[test+'-wait_for_mem']
            memEventsLoad.ix[test+'-wait_for_mem'] = pd.DataFrame([(df_critical_miss)[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum(0).sum(0)/ (df_critical_miss)['access_load'].sum(0)],index=[test+'-wait_for_mem'],columns=['wait_for_mem_latency']).ix[test+'-wait_for_mem']
            
            labels.append('')
            labels.append('')
            #labels.append(str(int(datos3['wait_for_mem_time'].sum(0)/ datos3['wait_for_mem_counter'].sum(0))))
            
            '''
            miss_lat = ((df_critical_miss)[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum(0).sum(0)/(df_miss+df_critical_miss)['access_load'].sum())* (1-hitratio.ix[test]['hit ratio'])
            hit_lat = ((df_critical_hit)[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum(0).sum(0)/(df_hit+df_critical_hit)['access_load'].sum())* (hitratio.ix[test]['hit ratio'])
            memEventsLoad.ix[test+'-ponderada'] = pd.DataFrame([(miss_lat,hit_lat)],index=[test+'-ponderada'],columns=['miss_lat','hit_lat']).ix[test+'-ponderada']
            '''
            
            
            #memEventsLoad.ix[test]['wavedfront_sync_load'] = memEventsLoad[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum(1)[test] -  df_sum[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum().sum() / df_sum['access_load'].sum()
                    
        except KeyError as e:
            print('plot_distribucion_lat() ERROR')
   
                
    memEventsLoad.plot(ax=axis[0], kind='bar',stacked=True,title='memEventsLoad')
    
    for p,value in zip(axis[0].patches,labels):
        axis[0].annotate(value, (p.get_x() , p.get_height() * 1.005))
    
    axis_config(axis[0],title = 'latencias',ylabel='ciclos',legend = memEventsLoad.columns)
    
    hitratio.plot(ax=axis[1], kind='bar',stacked=True,title='Hit Ratio')
    
    axis_config(axis[1],title = 'hit ratio',y_lim=[0,1])
    
    
    
def plot_distribucion_lat_continua(datos,bench, legend_label=''):
    
    #aux = pd.DataFrame().join(pd.DataFrame(datos,columns=[test]), how = 'outer')
    #aux
    f_lat, t_lat = plt.subplots(len(experimentos),2)
    f_lat.set_size_inches(10, 15)
    f_lat.set_dpi(300)
    
    memEventsLoad = pd.DataFrame(index=datos.keys(),columns=['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load'])
    
    gpu_latencias = pd.DataFrame(index=datos.keys(),columns=['queue_load_critical_miss','lock_mshr_load_critical_miss','lock_dir_load_critical_miss','eviction_load_critical_miss','retry_load_critical_miss','miss_load_critical_miss','finish_load_critical_miss'])
        
    #memEventsLoad.join(pd.DataFrame(datos2,columns=[test]), how = 'outer')
    for test in zip(datos.keys(),t_lat.transpose()[0]):
        ipc = datos[test[0]][bench]['extra-report_ipc'].copy()
        ipc = ipc.set_index('esim_time')
        
        df_miss = ipc[['queue_load_miss','lock_mshr_load_miss','lock_dir_load_miss','eviction_load_miss','retry_load_miss','miss_load_miss','finish_load_miss','access_load_miss']]
        df_hit = ipc[['queue_load_hit','lock_mshr_load_hit','lock_dir_load_hit','eviction_load_hit','retry_load_hit','miss_load_hit','finish_load_hit','access_load_hit']]
        df_critical_miss = ipc[['queue_load_critical_miss','lock_mshr_load_critical_miss','lock_dir_load_critical_miss','eviction_load_critical_miss','retry_load_critical_miss','miss_load_critical_miss','finish_load_critical_miss','access_load_critical_miss']]
        df_critical_hit = ipc[['queue_load_critical_hit','lock_mshr_load_critical_hit','lock_dir_load_critical_hit','eviction_load_critical_hit','retry_load_critical_hit','miss_load_critical_hit','finish_load_critical_hit','access_load_critical_hit']]
        
        col = ['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load','access_load']
        df_miss.columns = col
        df_hit.columns = col
        df_critical_miss.columns = col
        df_critical_hit.columns = col
        
        df_sum = df_miss + df_hit + df_critical_miss + df_critical_hit
    
        
        device = pd.DataFrame(datos[test[0]][bench]['device-spatial-report'][['esim_time','cycle']]).copy()
        device = device.set_index('esim_time')
        
        device = device.join(df_sum[col[0:7]].div(df_sum.loc[:,'access_load'].values, axis='index'),how = 'outer')
        
        device[memEventsLoad.columns] = device[memEventsLoad.columns].interpolate(method='index')
        #interpolate(metho)='index'  dropduplicated
        device.set_index(device['cycle'].cumsum().interpolate(method='index'))[memEventsLoad.columns].plot(ax=test[1],linewidth=0.1,kind='area',stacked=True,title=test[0])
        
    for test in zip(experimentos,t_lat.transpose()[1]):
        ipc = datos[test[0]][bench]['extra-report_ipc'].copy()
        ipc = ipc.set_index('esim_time')
        device = datos[test[0]][bench]['device-spatial-report'].copy()
        device = device.set_index('esim_time')
        device = device.join(ipc[gpu_latencias.columns].div( ipc.loc[:,'access_load_critical_miss'].values, axis='index'),how = 'outer')
        
        device[gpu_latencias.columns] = device[gpu_latencias.columns].interpolate(method='index')
        device.set_index(device['cycle'].cumsum().interpolate(method='index'))[gpu_latencias.columns].plot(ax=test[1],linewidth=0.1,kind='area',stacked=True,title=test[0])
        
        
    
    ylim_max = 0
    for t in t_lat.ravel():
        if t.get_ylim()[1] > ylim_max:
            ylim_max = t.get_ylim()[1]
            
    for t in t_lat.ravel():
        t.set_ylim([0,ylim_max])
        
    
    f_lat.tight_layout()
    f_lat.savefig(directorio_salida+bench+'-memoria-continua.pdf',format='pdf',bbox_inches='tight')
    
    for t in t_lat.ravel():
        t.set_xlim([1000000,2000000])
    
    f_lat.savefig(directorio_salida+bench+'-memoria-continua-zoom.pdf',format='pdf',bbox_inches='tight')
    
    
    for l in t_lat.ravel():
        l.cla()
    
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
        
def generar_hoja_calculo(datos,output_dir,file_input):
    
    
    #i_exp = []
    #i_ben = []
    #for experimentos in sorted_nicely(datos.keys()):
    #    for bench in sorted_nicely(datos[experimentos].keys()):
    #        i_exp.append(experimentos)
    #        i_ben.append(bench)
            
    opc = 0
    #df = pd.DataFrame(index=[sorted_nicely(datos.keys()),sorted_nicely(BENCHMARKS)],columns=['OPC'])
    df = pd.DataFrame()
    
    for bench in sorted_nicely(BENCHMARKS):
        for test in sorted_nicely(datos.keys()):
         
            try:    
            #df = df.append(pd.DataFrame([(test),(bench),(datos[test][bench]['device-spatial-report']['cycle'].sum())],columns=['test','benchmark','cycles']))
                #df = df.append(pd.DataFrame([(test , bench,datos[test][bench]['device-spatial-report']['cycle'].sum())],columns=['test','benchmark','cycles']))
                opc = datos[test][bench][file_input][['scalar_i','simd_op','s_mem_i','v_mem_op','lds_op']].sum(0).sum() / float(datos[test][bench][file_input]['cycle'].sum())
                latencia_gpu = datos[test][bench]['extra-report_ipc']['lat_loads_gpu'].sum(0).sum() / datos[test][bench]['extra-report_ipc']['num_loads_gpu'].sum(0).sum()
                latencia_mem = datos[test][bench]['extra-report_ipc']['lat_loads_mem'].sum(0).sum() / datos[test][bench]['extra-report_ipc']['num_loads_mem'].sum(0).sum()
                ht = datos[test][bench]['extra-report_ipc']['hits_L1'].sum(0).sum() / datos[test][bench]['extra-report_ipc']['accesos_L1'].sum(0).sum()
                accesos = datos[test][bench]['extra-report_ipc']['accesos_L1'].sum(0).sum()
                mpko = datos[test][bench]['extra-report_ipc']['misses_L1'].sum(0).sum() / datos[test][bench][file_input][['scalar_i','simd_op','s_mem_i','v_mem_op','lds_op']].sum(0).sum() 
                mpki = datos[test][bench]['extra-report_ipc']['misses_L1'].sum(0).sum() / datos[test][bench][file_input]['total_i'].sum(0).sum() 
                df = df.append(pd.DataFrame([[bench,opc,latencia_gpu,latencia_mem,ht,accesos,mpko,mpki]],index=[test],columns=['BENCHMARKS','OPC','latencia_gpu','latencia_mem','ht','accesos','mpko','mpki']))
                #print("ok: ")
                #df.loc[(test,bench),'OPC'] = datos[test][bench][file_input][['scalar_i','simd_op','s_mem_i','v_mem_op','lds_op']].sum(0).sum() / float(datos[test][bench][file_input]['cycle'].sum())
            except Exception as e:
                #print('WARNING generar_hoja_calculo : KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                df = df.append(pd.DataFrame([[bench,0]],index=[test],columns=['BENCHMARKS','OPC']))
                #print("error: ")
                
            #print("bench = "+bench+", test = "+test+" opc = ")
                
    #df.set_index(['benchmark','test'],inplace=True)

    df.to_excel(output_dir+'/resumen.xlsx',engine='xlsxwriter')
    
    return
            
    
def grafico_latencia_finalizacion_wg(axis, datos,output_dir):
   
    try:
        #df = datos.set_index("esim_time")
        (datos['op_counter']/datos['interval_cycles']).plot(ax=axis,title='OPC por WorkGroups')
    
    except KeyError as e:
        print('grafico_latencia_finalizacion_wg() ERROR')

    return
    
def plot_ipc_wf(datos,output_dir,bench):
    
    f6, t = plt.subplots(4,1)
    f6.set_size_inches(10, 15)
    f6.set_dpi(300)
    try:
    
        df = datos.set_index(datos['cycle'].cumsum())
        (df['wfop0'][df['wfop0'] >= 0]/df['cycle'][df['wfop0'] >= 0]).plot(ax=t[0])
        (df['wfvmop0'][df['wfop0'] >= 0]/df['cycle'][df['wfop0'] >= 0]).plot(ax=t[0])
        (df['mem_acc_lat'][df['wfop0'] >= 0]/df['mem_acc_end'][df['wfop0'] >= 0]).interpolate().plot(ax=t[0],secondary_y=True,legend=True)
        
        (df['wfop1'][df['wfop1'] >= 0]/df['cycle'][df['wfop1'] >= 0]).plot(ax=t[1])
        (df['wfvmop1'][df['wfop1'] >= 0]/df['cycle'][df['wfop1'] >= 0]).plot(ax=t[1])
        (df['mem_acc_lat'][df['wfop1'] >= 0]/df['mem_acc_end'][df['wfop1'] >= 0]).interpolate().plot(ax=t[1],secondary_y=True,legend=True)
        
        (df['wfop2'][df['wfop2'] >= 0]/df['cycle'][df['wfop2'] >= 0]).plot(ax=t[2])
        (df['wfvmop2'][df['wfop2'] >= 0]/df['cycle'][df['wfop2'] >= 0]).plot(ax=t[2])
        (df['mem_acc_lat'][df['wfop2'] >= 0]/df['mem_acc_end'][df['wfop2'] >= 0]).interpolate().plot(ax=t[2],secondary_y=True,legend=True)
        
        (df['wfop3'][df['wfop3'] >= 0]/df['cycle'][df['wfop3'] >= 0]).plot(ax=t[3])
        (df['wfvmop3'][df['wfop3'] >= 0]/df['cycle'][df['wfop3'] >= 0]).plot(ax=t[3])
        (df['mem_acc_lat'][df['wfop3'] >= 0]/df['mem_acc_end'][df['wfop3'] >= 0]).interpolate().plot(ax=t[3],secondary_y=True,legend=True)
    
        for l in t.ravel():
            #l.set_xlim(left = 0)
            l.set_ylim(bottom = 0)     
        
        f6.tight_layout()
        f6.savefig(output_dir+bench+'_wavefront_ipc.pdf',format='pdf',bbox_inches='tight')
    except Exception as e:
        pass
        
    for l in t.ravel():
        l.cla()
    
    plt.close(f6)
    
    return
    
def graficas_misses_x_stall(datos,output_dir):

    f, t = plt.subplots(5,4)
    f.set_size_inches(30, 18)
    f.set_dpi(300)
    
    f2, t2 = plt.subplots(5,4)
    f2.set_size_inches(30, 18)
    f2.set_dpi(300)
    
    f3, t3 = plt.subplots(1)
    f3.set_size_inches(10, 15)
    f3.set_dpi(300)
    
    index = 0
    df_wf_opc = pd.DataFrame({})
    for bench in sorted_nicely(BENCHMARKS):
        df_dist_misses = pd.DataFrame({})
        df_dist_missesxwf = pd.DataFrame({})
        
        table = t.ravel()[index]  
        table2 = t2.ravel()[index]  
        
        for test in sorted_nicely(datos.keys()):        
            try:
                df = datos[test][bench]['wf-spatial-report']
            
                df_dist_misses = df_dist_misses.append(pd.DataFrame(df.sum()[['dist_misses_1-5','dist_misses_6-10','dist_misses_11-15','dist_misses_16-20','dist_misses_21']],columns=[test]).T)
            except Exception as e:
                df_dist_misses = df_dist_misses.append(pd.DataFrame({},columns=[test]).T)
                
            try:
                df_dist_missesxwf = df_dist_missesxwf.append(pd.DataFrame(df.sum()[['dist_misses_1-5','dist_misses_6-10','dist_misses_11-15','dist_misses_16-20','dist_misses_21']]/len(df.index),columns=[test]).T)
            except Exception as e:
                df_dist_missesxwf = df_dist_missesxwf.append(pd.DataFrame({},columns=[test]).T)
                
                #df_wf_opc = df_wf_opc.join(pd.DataFrame([df.sum()['opc']/df.sum()['exec_cycles']],index=[test]))
                #df_dist_misses = df_dist_misses.join(pd.DataFrame([df.sum()['opc']/df.sum()['exec_cycles']],index=[test],columns=['OPC']))
            try:
                df_wf_opc = df_wf_opc.join(pd.DataFrame([df.sum()['opc']/df.sum()['exec_cycles']],index=[bench],columns=[test]),how='outer')
            except Exception as e:
                pass    
                
    
        df_dist_misses.plot(ax=table,kind='bar',title=bench,legend=True)
        df_dist_missesxwf.plot(ax=table2,kind='bar',title=bench,legend=True)
        #df_wf_opc.plot(ax=table,kind='bar',secondary_y='',legend=True)
        #f.tight_layout()
        #f.savefig(output_dir+bench+'_misses_x_stall.pdf',format='pdf',bbox_inches='tight')
        
        #df_wf_opc = df_wf_opc.join(pd.DataFrame([df.sum()['opc']/df.sum()['exec_cycles']],index=[test]))
        
    
        #for l in t.ravel():
        #   l.cla()
        
        index = index + 1
        
    df_wf_opc.plot(ax=t3,kind='bar',title=bench,legend=True)
        
    f.tight_layout()
    f.savefig(output_dir+'_misses_x_stall.pdf',format='pdf',bbox_inches='tight')
    
    f2.tight_layout()
    f2.savefig(output_dir+'_misses_x_stallxwf.pdf',format='pdf',bbox_inches='tight')
    
    f3.tight_layout()
    f3.savefig(output_dir+'_opc4.pdf',format='pdf',bbox_inches='tight')
    return 
    
 
if __name__ == '__main__':
   
    sb.set_style("whitegrid")
    cmap = sb.color_palette("Set2", 15)
    sb.set_palette(cmap, n_colors=15)
    
    mpl.rcParams['lines.linewidth'] = 2
    
    BENCHMARKS = ['BinarySearch','BinomialOption','BlackScholes','DCT','DwtHaar1D','FastWalshTransform','FloydWarshall','MatrixMultiplication','MatrixTranspose','MersenneTwister','QuasiRandomSequence','RadixSort','RecursiveGaussian','Reduction','ScanLargeArrays','SimpleConvolution','SobelFilter']

    #BENCHMARKS = ['BlackScholes','DwtHaar1D','FastWalshTransform','FloydWarshall','MatrixMultiplication','MersenneTwister']
    test = "tunk"
    bench = 'tunk'
    
    dir_resultados = "/nfs/gap/fracanma/benchmark/resultados"
    
    #experimentos = '05-31_si'
    #experimentos = '06-02_si_antiguo'
    experimentos = '06-15_simd_vmb1024_bigwfqueue'
    experimentos = '06-16'
    
    #legend = ['dinamico_anterior','trucado_anterior','dinamico_nuevo','trucado_nuevo','estatico']
    
    legend = ['estatico_mshr16', 'estatico_mshr32','estatico_mshr128','estatico_L2MSHR_mshr16', 'estatico_L2MSHR_mshr32','estatico_L2MSHR_mshr128']
    
    legend = ['mshr16','mshr32','mshr128']
    
    index_x = 'cycle' #'total_i'
    directorio_resultados = '/nfs/gap/fracanma/benchmark/resultados/'+experimentos+'/'
    
    directorio_salida = '/nfs/gap/fracanma/benchmark/resultados/'+experimentos+'_graficas/'
    
    if not os.path.exists(directorio_salida):
        os.mkdir(directorio_salida)
    
    dir_experimentos = []
    
    print('Empezamos a leer los archivos...')
    load_data_start = time.time()
    
    for exp in os.listdir(directorio_resultados):
        #dir_experimentos.append(directorio_resultados+'/'+exp)
        print(exp)
        if(os.path.isdir(directorio_resultados + exp)):
            print("ok")
            dir_experimentos.append(directorio_resultados + exp)
        
    datos = cargar_datos_sequencial(dir_experimentos,["device-spatial-report","extra-report_ipc","wf-spatial-report"])
        
    #df_prediccion = cargar_datos_sequencial([directorio_resultados+'/10-05_nmoesi_mshr32_predicion_opc_20000_conL1'],["device-spatial-report","extra-report_ipc"])   
    
    dir_estaticos = []
    
    #experimentos_baseline = ['09-16_nmoesi_mshr16_estatico_scalar8_conL1','09-16_nmoesi_mshr32_estatico_scalar8_conL1','09-16_nmoesi_mshr64_estatico_scalar8_conL1','09-16_nmoesi_mshr128_estatico_scalar8_conL1']
    
    '''experimentos_baseline = ['10-01_nmoesi_mshr32_lat300estatico_conL1']
    #experimentos_baseline = ['10-13_nmoesi_mshr32_estatico_conL1']
    
    
    for exp in experimentos_baseline:
        dir_estaticos.append(directorio_resultados+'/'+exp)  
    
    '''   
    
    prestaciones_estatico = cargar_datos_sequencial(dir_estaticos, ["device-spatial-report","device-spatial-report_wg"])
    
    print('Datos leidos en :',int(time.time() - load_data_start),'segundos')
    
    comprobar_estructura_datos(datos)
    
    #ajustar_resolucion(datos)
    generar_hoja_calculo(datos,directorio_salida,'device-spatial-report')
   
    graficas_misses_x_stall(datos,directorio_salida)

    #sys.exit(0)
    try:
        plot_latencia_memoria_VMB(datos,directorio_salida, legend_label='')
    except Exception as e:
        print('WARNING: KeyError plot_latencia_memoria_VMB()')
        
    for bench in sorted_nicely(BENCHMARKS):
        test = 0
        
            
        f, t = plt.subplots(5,1)
        f.set_size_inches(10, 15)
        f.set_dpi(300)
    
        f2, t2 = plt.subplots(9,1)
        f2.set_size_inches(10, 15)
        f2.set_dpi(300)
        
        f3, t3 = plt.subplots(2)
        f3.set_size_inches(10, 15)
        f3.set_dpi(300)
        
        f4, t4 = plt.subplots(5)
        f4.set_size_inches(10, 15)
        f4.set_dpi(300)
        
        f5, t5 = plt.subplots(4)
        f5.set_size_inches(10, 7)
        f5.set_dpi(300)
        
        index_axis_opc_salva = -1
        for test in sorted_nicely(datos.keys()): 
        #,sorted_nicely(prestaciones_estatico.keys())):
            index_axis_opc_salva = index_axis_opc_salva + 1 
            
            try:
                directorio_salida_wf = directorio_salida+test+'/'
                if not os.path.exists(directorio_salida_wf):
                    os.mkdir(directorio_salida_wf)
                
                df2 = datos[test][bench]['device-spatial-report'][['cycle','mappedWG','unmappedWG']]
                df2 = df2.set_index(df2['cycle'].cumsum())
                df3 = pd.DataFrame((df2['mappedWG'] - df2['unmappedWG']).cumsum(),columns=['wgs'])
                
                '''for i in reversed(df3.index):
                    if df3['wgs'][i] == df3['wgs'][df3.index[0]]:
                        start_finish_wg_x = i
                        break
                '''
                
                
                device_spatial_report_wg = pd.DataFrame(datos[test][bench]['device-spatial-report'][['esim_time','cycle','mappedWG','unmappedWG']]).copy()
                device_spatial_report_wg = device_spatial_report_wg.set_index('esim_time')        
                tunk = datos[test][bench]['device-spatial-report_wg'].set_index('esim_time')
                device_spatial_report_wg = device_spatial_report_wg.join(tunk,how ='outer')
                device_spatial_report_wg['cycle'] = device_spatial_report_wg['cycle'].cumsum().interpolate(method='index')
                device_spatial_report_wg = device_spatial_report_wg.join(pd.DataFrame(datos[test][bench]['device-spatial-report'].set_index('esim_time')[['simd_op', 'scalar_i','v_mem_op', 's_mem_i','lds_op','branch_i']].sum(1).cumsum(),columns=['op']),how = 'outer')
                device_spatial_report_wg[['op','cycle']] = device_spatial_report_wg[['op','cycle']].interpolate(method='index')
                device_spatial_report_wg2 = device_spatial_report_wg.set_index('cycle')
                
                grafico_latencia_finalizacion_wg(t4[0],device_spatial_report_wg2 ,directorio_salida_wf)
                
                opc_salva = pd.DataFrame(device_spatial_report_wg2[device_spatial_report_wg2.op_counter.notnull()]['op']/device_spatial_report_wg2[device_spatial_report_wg2.op_counter.notnull()].index)
                opc_salva.columns = [test]
                opc_salva.plot(ax=t4[1],title='OPC salva')
                if index_axis_opc_salva < 4 :
                    opc_salva.plot(ax=t5[index_axis_opc_salva])
                t4[1].set_xlim(left = 0)
                t4[1].set_ylim(bottom = 0)
                
                device_spatial_report_wg = pd.DataFrame(datos[test][bench]['device-spatial-report'][['esim_time','cycle','mappedWG','unmappedWG']]).copy()
                device_spatial_report_wg = device_spatial_report_wg.set_index('esim_time')        
                tunk = datos[test][bench]['device-spatial-report_wg'].drop_duplicates(subset=['esim_time']).set_index('esim_time')
                device_spatial_report_wg = device_spatial_report_wg.join(tunk,how ='outer')
                device_spatial_report_wg['cycle'] = device_spatial_report_wg['cycle'].cumsum().interpolate(method='index')
                device_spatial_report_wg = device_spatial_report_wg.join(pd.DataFrame(datos[test][bench]['device-spatial-report'].set_index('esim_time')[['simd_op', 'scalar_i','v_mem_op', 's_mem_i','lds_op','branch_i']].sum(1).cumsum(),columns=['op']),how = 'outer').interpolate(method='index')
                
                
                for i in reversed(df3.index):
                    if df3['wgs'][i] == df3['wgs'][df3.index[0]]:
                        start_finish_wg_x = i
                        break
                '''
                opc_julio_antes = pd.DataFrame([],columns=['cycle','op'])
                opc_julio_despues = pd.DataFrame([],columns=['cycle','op'])
                for i in np.arange(device_spatial_report_wg.index.size - 1):
                
                    if (device_spatial_report_wg['unmappedWG'][device_spatial_report_wg.index[i]] == 0) and (device_spatial_report_wg['unmappedWG'][device_spatial_report_wg.index[i+1]] > 0):
                        opc_julio_antes = opc_julio_antes.append(device_spatial_report_wg.loc[device_spatial_report_wg.index[i]])
                    
                    if (device_spatial_report_wg['unmappedWG'][device_spatial_report_wg.index[i]] > 0) and (device_spatial_report_wg['unmappedWG'][device_spatial_report_wg.index[i+1]] == 0):
                        opc_julio_despues = opc_julio_despues.append(device_spatial_report_wg.loc[device_spatial_report_wg.index[i]])
                        
                opc_julio_antes = opc_julio_antes.set_index('cycle')
                opc_julio_despues = opc_julio_despues.set_index('cycle')
                opc_julio_antes.columns = ['antes', 'interval_cycles', 'op_counter', 'mappedWG','unmappedWG']
                (opc_julio_antes['antes']/opc_julio_antes.index).plot(ax=t4[2],title='OPC julio',legend=True) 
                opc_julio_despues.columns = ['despues', 'interval_cycles', 'op_counter', 'mappedWG','unmappedWG']
                (opc_julio_despues['despues']/opc_julio_despues.index).plot(ax=t4[2],title='OPC julio',legend=True) 
                
           
                t4[2].set_xlim(left = 0)
                t4[2].set_ylim(bottom = 0)     
                '''
                #(device_spatial_report_wg['op']/device_spatial_report_wg['cycle']).plot(ax=t4[1],title='OPC salva')
            
                #axis_config(t[3], title = 'opc acumulado estaticos')
                plot_opc(t4[3],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
                t4[3].set_xlim(left = 0)
                t4[3].set_ylim(bottom = 0)  
                
                plot_wg_unmapped(t4[4],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
              
                df_wg_vertical = (device_spatial_report_wg['mappedWG'] - device_spatial_report_wg['unmappedWG']).cumsum()
                #for i in zip(np.arange(df_wg_vertical)):
                
                #    if (device_spatial_report_wg['unmappedWG'][device_spatial_report_wg.index[i]] == 0) and (device_spatial_report_wg['unmappedWG'][device_spatial_report_wg.index[i+1]] > 0):
                    
                
                t4[0].axvline(x=start_finish_wg_x, linewidth=2, color='r')
                t4[1].axvline(x=start_finish_wg_x, linewidth=2, color='r')
                t4[2].axvline(x=start_finish_wg_x, linewidth=2, color='r')
                t4[3].axvline(x=start_finish_wg_x, linewidth=2, color='r')
                t4[4].axvline(x=start_finish_wg_x, linewidth=2, color='r')
                
                t4[0].set_ylabel("opc")
                t4[1].set_ylabel("opc")
                t4[2].set_ylabel("opc")
                t4[3].set_ylabel("opc")
                t4[4].set_ylabel("gw finalizados cada 1kciclo")
                #f.suptitle(bench, fontsize=25)
                plot_wg_active(t4[4],datos[test][bench]['device-spatial-report'],'cycle')
                f4.tight_layout()
                f4.savefig(directorio_salida_wf+bench+'_opc_wg.pdf',format='pdf',bbox_inches='tight')
                plot_ipc_wf(datos[test][bench]['device-spatial-report'],directorio_salida_wf,bench)
            except KeyError as e:
                print('WARNING: KeyError plot_wg_unmapped()')
            
            for l in t4.ravel():
                l.cla()
            
                
            
        
            
            '''try:
                plot_wg_active(t[1][2],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
                axis_config(t[1][2],title = 'wg active')
            except KeyError as e:
                print('WARNING: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
            '''
            '''
            try:
                plot_opc(t[0],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
            except KeyError as e:
                print('WARNING1: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
            try:
                plot_mshr_size(t[3],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
                
            except KeyError as e:
                print('WARNING2: KeyError in datos['+test+']['+bench+'][device-spatial-report]')   
            '''
            '''    
            try:
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
            '''try:
                plot_opc_accu(t[2],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
            except KeyError as e:
                print('WARNING3: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
             
            try:
                plot_wait_for_mem(t[1],datos[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
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
            '''
        
        
        try:
            plot_distribucion_lat(t3,datos,index_x,bench, 'tunk')
        except KeyError as e:
            print('WARNING12: KeyError in datos['+test+']['+bench+'][device-spatial-report]')  
            
        try:   
             plot_distribucion_lat_continua(datos,bench, legend_label='')
        except KeyError as e:
            print('WARNING13: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
                
                
        for test in sorted_nicely(prestaciones_estatico.keys()): 
            try:
                plot_opc_accu(t[3],prestaciones_estatico[test][bench]['device-spatial-report'], index=index_x, legend_label=test)
                
            except KeyError as e:
                print('WARNING14: KeyError in datos['+test+']['+bench+'][device-spatial-report]')
        
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
            
        axis_config(t[0],title = 'OPC')
        axis_config(t[3], title='mshr size',y_lim = [0,256])
        axis_config(t[1], title='wait_for_mem')
        axis_config(t[2], title = 'opc acumulado')
        #axis_config(t[3], title = 'opc acumulado estaticos')
            
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
        for l in t2.ravel():
            l.cla()
        plt.close(f2)
        
        #zoom
        '''if bench == BENCHMARKS[1]:
            axis_config(t[0],title = 'OPC',x_lim = [0,2000000])
            axis_config(t[1], title='mshr size',y_lim = [0,256],x_lim = [0,2000000])
            axis_config(t[2], title = 'opc acumulado',x_lim = [0,2000000])
            axis_config(t[3], title = 'opc acumulado estaticos',x_lim = [0,2000000])
        '''
        f.tight_layout()
        f.savefig(directorio_salida+bench+'-ZOOM.pdf',format='pdf',bbox_inches='tight')
        for l in t.ravel():
            l.cla()
        plt.close(f)
        
        #axis_config(t3[0],title = 'latencias',ylabel='ciclos')
        f3.tight_layout()
        f3.savefig(directorio_salida+bench+'-memoria.pdf',format='pdf',bbox_inches='tight')
        for l in t3.ravel():
            l.cla()
        plt.close(f3)

        plt.close(f4)
        
        f5.tight_layout()
        f5.savefig(directorio_salida+bench+'-4opcaccu.pdf',format='pdf',bbox_inches='tight')
        plt.close(f5)
        
    f, t = plt.subplots() 
    #datos.update(prestaciones_estatico)       
    plot_opc_barras(t,datos,'device-spatial-report', BENCHMARKS,index_x, legend_label='')
    axis_config(t, title='speedup')
    t.set_ylim(bottom = 0)
    #t.legend(legend,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    t.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    f.savefig(directorio_salida+'opc.pdf',format='pdf',bbox_inches='tight')
    t.cla()
    plt.close(f)
    #generar_hoja_calculo(datos)
    
    exit()
    