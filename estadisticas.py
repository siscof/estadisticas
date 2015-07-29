#!/usr/bin/python3.4

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

#funciones

def sorted_nicely( l ):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

def leer_columnas(f):
    archivo = open(f)
    nombre_columnas = archivo.readline().split(',')

    archivo.close()
    return True

def cargarDatos():

    loadDictCompleto('fran_general')

    return

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

def buscar_valor(archivo, palabra):

    linea = 'FALLO'
    for i in archivo:
        if i.startswith(palabra):
            linea = i
            break
    if linea == 'FALLO':
        return 0
    else:
        print(linea.split('=')[1])
        print(float(linea.split('=')[1]))
        return float(linea.split('=')[1])

def buscar_y_acumular(archivo, inicio, fin, palabra):
    reads = 0
    for linea in archivo:
        if linea.startswith(inicio):
            break
    for linea in archivo:
        if linea.startswith(palabra+' ='):
            reads = reads + float(linea.split('=')[1])
        elif linea.startswith(fin):
            break
    return reads

def buscar_y_acumular_porcentage(archivo, inicio, fin, palabra):
    reads = 0
    aux = 0
    for linea in archivo:
        if linea.startswith(inicio):
            break
    for linea in archivo:

        if linea.startswith(palabra+' ='):
            aux = aux + 1
            reads = reads + float(linea.split('=')[1])
        elif linea.startswith(fin):
            break
    return reads / aux

def IPC(nombreArchivo, monton):
    f, t= plt.subplots(1)
    #for test in grupo:
    f.set_size_inches(15, 10)
    f.set_dpi(150)
    if not os.path.exists(nombreArchivo):
        os.mkdir(nombreArchivo)

    for bench in BENCHMARKS :
        t.set_title(bench)

        try:
            for clave in sorted(TESTS.keys()) :

                df = monton[bench][clave].sum

                dato = df['total_intervalo']/ df['ciclos_intervalo']
                dato.plot(ax=t, label=clave, legend=True)

                plt.legend()
            t.set_ylabel('IPC')
            t.set_ylim(bottom = 0)
            t.set_xlabel('Instrucciones ejecutadas')
            #t.set_xticks([])
            f.savefig(nombreArchivo+bench+'IPC')
        except IOError as e:
            print('fallo al crear grafica ipc para:')
            print(e)
        except Exception as e:
            print('Fallo grafica ipc en el test: '+bench+' '+clave)
            print(e)
        t.cla()
    plt.close(f)
    return

def IPCmultitest_worker(directorioResultados, bench):

    tamanyoGrupo = 500000
    f, t= plt.subplots(2,3)
    f.set_size_inches(30, 20)
    f.set_dpi(300)
    t[0][0].set_title('OPC')
    t[0][1].set_title('Latencia')
    t[1][0].set_title('OPC acumulado')
    t[1][1].set_title('cantidad de accesos por intervalo')
    t[0][2].set_title('Hit Ratio L1')

    aux = pd.DataFrame()
    #aux = None;
    for exp in nombre_resumen :


        try:
            for clave in ['con L1'] :

                df = dict_por_instrucciones[exp][bench][clave].df
                ind = pd.Index(dict_por_instrucciones[exp][bench][clave].df['total_global'], name='Operaciones')
                tamanyoGrupo = 10000 * np.ceil(df.shape[0]/100)

                df2 = df.set_index(ind).groupby( lambda x :  (x // tamanyoGrupo) * tamanyoGrupo).sum() #.astype(np.int32)

                if exp in nombre_resumen[3:]:
                     pd.DataFrame({exp:df2['total_global']/df2['ciclos_totales']}).plot(ax=t[1][2])
                     pd.DataFrame({exp:df2['mshr_size_L1']/tamanyoGrupo}).plot(ax=t[0][2])
                     continue

                #OPC
                dato = pd.DataFrame({exp:df2['total_intervalo']/ df2['ciclos_intervalo']})

                #if exists(opcOptimo)
                #opcOptimo.merge(dato)
                #dato = dato.set_index(ind).groupby( lambda x :  (x // tamanyoGrupo) * tamanyoGrupo).mean()
                dato.plot(ax=t[0][0])

                #latencia
                dato = pd.DataFrame({exp:df2['lat_loads_gpu']/ df2['num_loads_gpu']})
                #dato = dato.set_index(ind).groupby( lambda x :  (x // tamanyoGrupo) * tamanyoGrupo).mean()
                dato.plot(ax=t[0][1])

                #accesos
                dato = pd.DataFrame({exp:df2['accesos_gpu']})
                dato.plot(ax=t[1][1])

                #opc acumulado
                dato = pd.DataFrame({exp:df2['total_global']/df2['ciclos_totales']})
                dato.plot(ax=t[1][0])
                aux = aux.join(dato, how="right")


                #hitRatio
                #pd.DataFrame({exp:(df2['hits L1']/ df2['efectivos L1'])}).replace({np.inf:0}).plot(ax=t[0][2])

                #tamaño mshr
                #pd.DataFrame({exp:df2['mshr size L1']}).plot(ax=t[0][2])

                #plt.legend()

        except IOError as e:
            print('fallo al crear grafica opc para:')
            print(e)
        except Exception as e:
            print('Fallo grafica ipcmultitest en el test: '+bench+' '+clave)
            print(e)

    try:
        aux.max(1).plot(ax=t[1][2],style=['k--'])
        aux.min(1).plot(ax=t[1][2],style=['k--'])

        #pd.DataFrame(aux[nombre_resumen[3]]).plot(ax=t[1][2])
    except Exception as e:
        print('Fallo ipcmultitest grafica maximos y minimos en el test: '+exp)
        print(e)


    t[0][0].set_ylabel('OPC')
    t[0][0].set_ylim(bottom = 0)
    t[0][0].set_xlabel('Operaciones ejecutadas')
    #opcOptimo.plot(ax=t[0][2])
    #t[0].set_xticks([])
    #f[0].savefig(directorioResultados+bench+'_OPCmultitest')

    t[0][1].set_ylabel('Latencia (ciclos GPU)')
    t[0][1].set_ylim(bottom = 0)
    t[0][1].set_xlabel('Operaciones ejecutadas')
    #t[1].set_xticks([])

    t[1][0].set_ylabel('OPC')
    t[1][0].set_ylim(bottom = 0)
    t[1][0].set_xlabel('Operaciones ejecutadas')



    t[1][1].set_ylabel('accesos por intervalo')
    t[1][1].set_ylim(bottom = 0)
    t[1][1].set_xlabel('Operaciones ejecutadas')

    t[0][2].set_ylabel('hit Ratio(base 1)')
    t[0][2].set_ylim(bottom = 0)
    t[0][2].set_xlabel('Operaciones ejecutadas')

    t[1][2].set_ylabel('OPC')
    t[1][2].set_ylim(bottom = 0)
    t[1][2].set_xlabel('Operaciones ejecutadas')

    f.suptitle(bench+' (intervalo = '+str(tamanyoGrupo)+')', fontsize=25)
    f.savefig(directorioResultados+bench+'_OpcLatenciaMultitest-'+str(tamanyoGrupo)+'.eps',format='eps')

    #t[0].cla()
    #t[1].cla()
    plt.close(f)
    return

def IPCmultitestmultiprocess(directorioResultados):
    if not os.path.exists(directorioResultados):
        os.mkdir(directorioResultados)

    args = []
    for bench in BENCHMARKS :
        args.append((directorioResultados, bench))

    pool_ipc = multiprocessing.Pool()
    pool_ipc.starmap(IPCmultitest_worker,args)

    pool_ipc.close()
    pool_ipc.join()

    return

def benchXexp(directorioResultados, monton):

    f, t = plt.subplots(2,2)
    f.set_size_inches(30, 20)
    f.set_dpi(300)

    dir = directorioResultados+'/benchXexp/'
    if not os.path.exists(dir):
        os.mkdir(dir)

    #pprint.pprint(nombre_resumen)

    for bench in sorted_nicely(BENCHMARKS) :


        latencia = pd.DataFrame(index=nombre_resumen, columns=['latencia_total','latencia_retry'])
        evictionsL2 = pd.DataFrame(index=nombre_resumen, columns=['evictions_L2'])
        invalidations = pd.DataFrame(index=nombre_resumen, columns=['invalidations'])
        hitratio = pd.DataFrame(index=nombre_resumen, columns=['hit_ratio'])
        memEventsLoad = pd.DataFrame(index=nombre_resumen, columns=['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load'])
        gpuEventsLoad = pd.DataFrame(index=nombre_resumen, columns=['gpu_queue_load','gpu_lock_mshr_load','gpu_lock_dir_load','gpu_eviction_load','gpu_retry_load','gpu_miss_load','gpu_finish_load'])
        memEventsStore = pd.DataFrame(index=nombre_resumen, columns=['queue_nc_write','lock_mshr_nc_write','lock_dir_nc_write','eviction_nc_write','retry_nc_write','miss_nc_write','finish_nc_write'])
        df_dispatch = pd.DataFrame(index=nombre_resumen, columns=['cycles_simd_running','dispatch_stall_mem_access','dispatch_stall_barrier','dispatch_stall_instruction_infly','dispatch_stall_others'])
        df_instructions_infly = pd.DataFrame(index=nombre_resumen, columns=['dispatch_branch_instruction_infly', 'dispatch_scalar_instruction_infly','dispatch_simd_instruction_infly', 'dispatch_lds_instruction_infly', 'dispatch_mem_scalar_instruction_infly', 'dispatch_v_mem_instruction_infly'])

        for exp in nombre_resumen :
            try:
                df_aux = monton[exp][bench]['con L1'].df
                df_max = pd.DataFrame(monton[exp][bench]['con L1'].df.max(),columns=[exp]).transpose()

                lat_retry = df_aux[['cycles_load_action_retry','cycles_load_miss_retry','cycles_nc_store_writeback_retry','cycles_nc_store_action_retry','cycles_nc_store_miss_retry']].sum().sum() / df_aux['num_loads_mem'].sum()

                latencia.ix[exp] = pd.DataFrame({'latencia_total':(df_aux['lat_loads_mem'].sum() / df_aux['num_loads_mem'].sum()) - lat_retry, 'latencia_retry':lat_retry},index=[exp]).ix[exp]

                evictionsL2.ix[exp,'evictions_L2'] = df_aux['evictions_L2'].sum() / df_aux['num_loads_mem'].sum()

                memEventsLoad.ix[exp] = pd.DataFrame(df_aux[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum(0)/ df_aux['access_load'].sum(0),columns=[exp]).transpose().ix[exp]
                memEventsStore.ix[exp] = pd.DataFrame(df_aux[['queue_nc_write','lock_mshr_nc_write','lock_dir_nc_write','eviction_nc_write','retry_nc_write','miss_nc_write','finish_nc_write']].sum(0)/ df_aux['access_nc_write'].sum(0),columns=[exp]).transpose().ix[exp]
                gpuEventsLoad.ix[exp] = pd.DataFrame(df_aux[['gpu_queue_load','gpu_lock_mshr_load','gpu_lock_dir_load','gpu_eviction_load','gpu_retry_load','gpu_miss_load','gpu_finish_load']].sum(0)/ df_aux['gpu_access_load'].sum(0),columns=[exp]).transpose().ix[exp]

                #analysis stall in dispatch
                df_dispatch.ix[exp] = df_max.ix[exp,['cycles_simd_running','dispatch_stall_mem_access','dispatch_stall_barrier','dispatch_stall_instruction_infly','dispatch_stall_others']]
                df_instructions_infly.ix[exp] = df_max.ix[exp,['dispatch_branch_instruction_infly', 'dispatch_scalar_instruction_infly','dispatch_simd_instruction_infly', 'dispatch_lds_instruction_infly', 'dispatch_mem_scalar_instruction_infly', 'dispatch_v_mem_instruction_infly']]



                invalidations.ix[exp,'invalidations'] = df_aux['invalidations'].sum() / df_aux['num_loads_mem'].sum()
                #pprint.pprint(contenedor_de_datos+'/'+exp+'_conL1/'+bench+'-mem-report')
                fi = open(contenedor_de_datos+'/'+exp+'_conL1/'+bench+'-mem-report')
                hitratio.ix[exp,'hit ratio'] =  buscar_y_acumular(fi,'[ l2-0 ]','[ l1-cu00 ]','HitRatio')
                fi.close()



            except IOError as e:
                print('fallo en benchXexp para:')
                print(e)
            except Exception as e:
                print('Fallo en benchXexp en el test: '+bench)
                print(e)

        latencia.plot(ax=t[0][0], kind='bar',stacked=True,title='latencia con retry')


        evictionsL2.plot(ax=t[0][1], kind='bar',stacked=True,title='evictions en L2')

        invalidations.plot(ax=t[1][0], kind='bar',stacked=True,title='invalidations')
        hitratio.plot(ax=t[1][1], kind='bar',stacked=True,title='HitRatio')


        f.savefig(dir+bench+'.eps',format='eps', bbox_inches='tight')

        f_mem, t_mem = plt.subplots(1,2)
        f_mem.set_size_inches(30, 20)
        f_mem.set_dpi(300)

        memEventsLoad.plot(ax=t_mem[0], kind='bar',stacked=True,title='memEventsLoad')

        gpuEventsLoad.plot(ax=t_mem[1], kind='bar',stacked=True,title='gpuEventsLoad')
        #memEventsStore.plot(ax=t_mem[1], kind='bar',stacked=True,title='memEventsWrite')
        #.legend(loc='upper left', bbox_to_anchor=(0.6, 1.4))

        t_mem[0].set_ylabel('Latencia (ciclos)')
        t_mem[1].set_ylabel('Latencia (ciclos)')

        max_ylim = max([t_mem[0].get_ylim()[1],t_mem[1].get_ylim()[1]])
        t_mem[0].set_ylim([0,max_ylim])
        t_mem[1].set_ylim([0,max_ylim])
        t_mem[0].set_ylabel("Latency Cycles")

        f_mem.suptitle(bench, fontsize=25)

        f_mem.savefig(dir+bench+'_memEvents.eps',format='eps', bbox_inches='tight')

        fig_paper , tab_paper = plt.subplots()
        fig_paper.set_size_inches(30, 20)
        fig_paper.set_dpi(300)
        memEventsLoad.plot(ax=tab_paper, kind='bar',stacked=True,title="Latency variations")
        fig_paper.savefig(dir+bench+'_latpaper.eps',format='eps', bbox_inches='tight')

        plt.close(fig_paper)
        plt.close(f_mem)
        t[0][0].cla()
        t[0][1].cla()
        t[1][0].cla()
        t[1][1].cla()

        df_dispatch[['dispatch_stall_mem_access','dispatch_stall_barrier','dispatch_stall_instruction_infly','dispatch_stall_others']] = df_dispatch[['dispatch_stall_mem_access','dispatch_stall_barrier','dispatch_stall_instruction_infly','dispatch_stall_others']] / 10
        #df_dispatch = df_dispatch / 10
        df_dispatch = df_dispatch.div(df_dispatch.sum(1), axis=0)
        df_instructions_infly = df_instructions_infly.div(df_instructions_infly.sum(1), axis=0)

        f_dispatch, t_dispatch = plt.subplots(1,2)
        f_dispatch.set_size_inches(30, 20)
        f_dispatch.set_dpi(300)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        df_dispatch.columns = ['cycles_simd_running','others_stalls','barrier_stall','memory_fence_stall','instruction_infly_stall']
        df_dispatch.plot(ax=t_dispatch[0], kind='bar',stacked=True,title='cicles idle').legend(loc='upper right', bbox_to_anchor=(0.5, 1.15))
        #f.savefig(dir + exp +'_dispatch_stall.eps',format='eps')

        df_instructions_infly.plot(ax=t_dispatch[1], kind='bar',stacked=True,title='instructions in fly').legend(loc='upper left', bbox_to_anchor=(0.4, 1.20))
        t_dispatch[0].set_ylim(0,1)
        t_dispatch[1].set_ylim(0,1)
        f_dispatch.suptitle(bench +'_dispatch_stall', fontsize=25)
        f_dispatch.savefig(dir + bench +'_dispatch_stall.eps',format='eps', bbox_inches='tight')

        plt.close(f_dispatch)

    t[0][0].cla()
    t[0][1].cla()
    t[1][0].cla()
    t[1][1].cla()
    plt.close(f)



    return

def barras_opc(directorioResultados, monton):

    dir = directorioResultados+'/barras OPC/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    f, t = plt.subplots()
    f.set_size_inches(30, 20)
    f.set_dpi(300)
    opc = pd.DataFrame(index=benchmarks_amd, columns=nombre_resumen)
    ipc = pd.DataFrame(index=benchmarks_amd, columns=nombre_resumen)
    invalidations = pd.DataFrame(index=benchmarks_amd, columns=nombre_resumen)
    retries = pd.DataFrame(index=benchmarks_amd, columns=nombre_resumen)
    latencia = pd.DataFrame(index=benchmarks_amd, columns=nombre_resumen)
    coalesce = pd.DataFrame(index=benchmarks_amd, columns=nombre_resumen)
    for exp in sorted_nicely(monton.keys()) :

        for bench in BENCHMARKS :

            try:
                df_aux = monton[exp][bench]['con L1'].df
                #opc[exp][bench] = df_aux['total_global'].max() / float(df_aux['ciclos_totales'].max())
                opc[exp][bench] = df_aux[['i_scalar','i_simd','i_s_mem','i_v_mem','i_branch','i_lds']].sum(0).sum() / float(df_aux['ciclos_intervalo'].sum())
                print('bench = '+df_aux[['i_scalar','mi_simd','i_s_mem','mi_v_mem','i_branch','mi_lds']].sum(0).to_string()+ ' ; '+ str(df_aux[['i_scalar','mi_simd','i_s_mem','mi_v_mem','i_branch','mi_lds']].sum(0).sum())+' ; '+ str(df_aux['ciclos_intervalo'].sum()))

                ipc[exp][bench] = df_aux[['i_scalar','mi_simd','i_s_mem','mi_v_mem','i_branch','mi_lds']].sum(0).sum() / float(df_aux['ciclos_intervalo'].sum())
                invalidations[exp][bench] = df_aux['invalidations'].sum() / (df_aux['accesos_gpu'].sum() - df_aux['Coalesces_gpu'].sum() - df_aux['Coalesces_L1'].sum())
                retries[exp][bench] = df_aux['accesses_with_retries'].sum() / (df_aux['accesos_gpu'].sum() - df_aux['Coalesces_gpu'].sum() - df_aux['Coalesces_L1'].sum())
                latencia[exp][bench] = df_aux['lat_loads_mem'].sum() / df_aux['num_loads_mem'].sum()


                memEventsLoad = df_aux[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum(0).sum()/ df_aux['access_load'].sum(0).sum()
                memEventsStore = df_aux[['queue_nc_write','lock_mshr_nc_write','lock_dir_nc_write','eviction_nc_write','retry_nc_write','miss_nc_write','finish_nc_write']].sum(0).sum()/ df_aux['access_nc_write'].sum(0).sum()
                gpuEventsLoad = df_aux[['gpu_queue_load','gpu_lock_mshr_load','gpu_lock_dir_load','gpu_eviction_load','gpu_retry_load','gpu_miss_load','gpu_finish_load']].sum(0).sum()/ df_aux['gpu_access_load'].sum(0).sum()

                latencia[exp][bench] = memEventsLoad + memEventsStore

            except Exception as e:
                print('Fallo barras_opc(): '+exp+' '+bench)
                print(e)

        #latencia retry
    latencia.plot(ax=t, kind='bar')
    f.savefig(dir + 'latencias.eps',format='eps')
    t.cla()

    carpeta = '/nfs/gap/fracanma/benchmark/resultados/'

    t.set_ylabel("OPC")
    t.set_xlabel("Benchmarks")
    opc.plot(ax=t, kind='bar')
    f.savefig(dir + 'opc.eps',format='eps', bbox_inches='tight')
    f.savefig(carpeta + 'opc.eps',format='eps', bbox_inches='tight')
    t.cla()

    t.set_ylabel("IPC")
    t.set_xlabel("Benchmarks")
    ipc.plot(ax=t, kind='bar',rot=25, title='mshr performance')
    f.savefig(carpeta + 'ipc.eps',format='eps', bbox_inches='tight')
    t.cla()

    #opc.columns = ['4 mshr entries','8 mshr entries','16 mshr entries','mshr disabled']
    t.set_ylabel("OPC")
    t.set_xlabel("Benchmarks")
    opc.ix[['BlackScholes','DCT','MersenneTwister','QuasiRandomSequence']].plot(ax=t, kind='bar',rot=25, title='mshr performance')
    f.savefig(carpeta + 'opc_altos.eps',format='eps', bbox_inches='tight')
    t.cla()

    t.set_ylabel("OPC")
    t.set_xlabel("Benchmarks")
    opc.ix[['DwtHaar1D','FastWalshTransform','FloydWarshall','RecursiveGaussian','Reduction','ScanLargeArrays']].plot(ax=t, kind='bar',rot=25, title='mshr performance')
    f.savefig(carpeta + 'opc_bajos.eps',format='eps', bbox_inches='tight')
    t.cla()

    t.set_ylabel("IPC")
    t.set_xlabel("Benchmarks")
    ipc.ix[['DwtHaar1D','FastWalshTransform','FloydWarshall','RecursiveGaussian','Reduction','ScanLargeArrays']].plot(ax=t, kind='bar',rot=25, title='IPC')
    f.savefig(carpeta + 'ipc_bajos.eps',format='eps', bbox_inches='tight')
    t.cla()

    invalidations.plot(ax=t, kind='bar')
    f.savefig(dir + 'invalidations.eps',format='eps')
    t.cla()

    retries.plot(ax=t, kind='bar')
    f.savefig(dir + 'retries.eps',format='eps')
    t.cla()
    plt.close(f)

    return


def analisis_stall(directorioResultados, monton):

    dir = directorioResultados+'/barras OPC/'
    if not os.path.exists(dir):
        os.mkdir(dir)

    dir_latencias = directorioResultados+'/por_instrucciones/latencias/'
    if not os.path.exists(dir_latencias):
        os.mkdir(dir_latencias)

    df_exp = pd.DataFrame()

    for exp in sorted_nicely(monton.keys()) :
        df_fetch = pd.DataFrame(index=sorted_nicely(monton[exp].keys()),columns = ['no_stall','stall_mem_access','stall_barrier','stall_instruction_infly','stall_fetch_buffer_full','stall_no_wavefront','stall_others'])
        df_dispatch = pd.DataFrame(index=sorted_nicely(monton[exp].keys()),columns = ['cycles_simd_running','dispatch_stall_others','dispatch_stall_barrier','dispatch_stall_mem_access','dispatch_stall_instruction_infly'])
        df_instructions = pd.DataFrame(index=sorted_nicely(monton[exp].keys()),columns = ['i_scalar','mi_simd','i_branch','mi_lds','i_s_mem','mi_v_mem'])
        df_operations = pd.DataFrame(index=sorted_nicely(monton[exp].keys()),columns = ['i_scalar', 'i_simd', 'i_s_mem', 'i_v_mem', 'i_branch', 'i_lds',])
        df_instructions_infly = pd.DataFrame(index=sorted_nicely(monton[exp].keys()),columns = ['dispatch_branch_instruction infly', 'dispatch_scalar_instruction_infly','dispatch_simd_instruction_infly', 'dispatch_lds_instruction_infly', 'dispatch_mem_scalar_instruction_infly', 'dispatch_v_mem_instruction_infly'])

        for bench in BENCHMARKS:
            try:
                df_max = pd.DataFrame(monton[exp][bench]['con L1'].df.max(),columns=[bench]).transpose()
                df_sum = pd.DataFrame(monton[exp][bench]['con L1'].df.sum(),columns=[bench]).transpose()

                df_fetch.ix[bench] = df_max.ix[bench,['no_stall','stall mem_access','stall_barrier','stall_instruction_infly','stall_no_wavefront','stall_others','stall_fetch_buffer_full']]
                df_dispatch.ix[bench] = df_max.ix[bench,['cycles_simd_running','dispatch_stall_mem_access','dispatch_stall_barrier','dispatch_stall_instruction_infly','dispatch_stall_others']]
                df_instructions_infly.ix[bench] = df_max.ix[bench,['dispatch_branch_instruction_infly', 'dispatch_scalar_instruction_infly', 'dispatch_mem_scalar_instruction_infly', 'dispatch_simd_instruction_infly', 'dispatch_v_mem_instruction_infly', 'dispatch_lds_instruction_infly']]


                df_instructions.ix[bench] = df_sum.ix[bench,['i_scalar', 'mi_simd', 'i_s_mem', 'mi_v_mem', 'i_branch', 'mi_lds',]]

                df_operations.ix[bench] = df_sum.ix[bench,['i_scalar', 'i_simd', 'i_s_mem', 'i_v_mem', 'i_branch', 'i_lds',]]

                #comparacion latencias
                df_lat = monton[exp][bench]['con L1'].df

                latencia_retries = pd.DataFrame({'retry_lat':df_lat[['cycles_load_action_retry','cycles_load_miss_retry','cycles_nc_store_writeback_retry','cycles_nc_store_action_retry','cycles_nc_store_miss_retry']].sum(1) / df_lat['num_loads_mem']})
                                                             #df_lat[['counter load action retry','counter load miss retry','counter nc store writeback retry','counter nc store action retry','counter nc store miss retry']].sum(1)})
                latencia_total = pd.DataFrame({'latencia_total':df_lat['lat_loads_mem'] /  df_lat['num_loads_mem']})

                f2, t2 = plt.subplots()
                f2.set_size_inches(30, 20)
                f2.set_dpi(300)

                pd.rolling_mean(latencia_retries, 20).plot(ax=t2)
                pd.rolling_mean(latencia_total, 20).plot(ax=t2)
                f2.savefig(dir_latencias + bench +'_'+exp+'.eps',format='eps')

                t2.cla()
                plt.close(f2)

            except Exception as e:
                print('Fallo analisis stall(): '+exp+' '+bench)
                print(e)

        df_fetch = df_fetch.div(df_fetch.sum(1), axis=0)
        df_dispatch[['dispatch_stall_mem_access','dispatch_stall_barrier','dispatch_stall_instruction_infly','dispatch_stall_others']] = df_dispatch[['dispatch_stall_mem_access','dispatch_stall_barrier','dispatch_stall_instruction_infly','dispatch_stall_others']] / 10

        #prueba
        #df_dispatch_aux = pd.DataFrame(df_dispatch)
        df_exp = df_exp.append(pd.DataFrame(df_dispatch.values,index= pd.MultiIndex.from_product([[exp],df_dispatch.index]) ,columns=df_dispatch.columns))

        df_dispatch = df_dispatch.div(df_dispatch.sum(1), axis=0)
        df_instructions = df_instructions.div(df_instructions.sum(1), axis=0)
        df_operations = df_operations.div(df_operations.sum(1), axis=0)
        df_instructions_infly = df_instructions_infly.div(df_instructions_infly.sum(1), axis=0)

        f, t = plt.subplots()
        f.set_size_inches(30, 20)
        f.set_dpi(300)

        df_fetch.plot(ax=t, kind='bar',stacked=True)
        f.suptitle(exp +'_fetch_stall', fontsize=25)
        f.savefig(dir + exp +'_fetch_stall.eps',format='eps')
        t.cla()

        df_instructions.plot(ax=t, kind='bar',stacked=True)
        f.suptitle(exp +'_instructions', fontsize=25)
        f.savefig(dir + exp +'_instructions.eps',format='eps')
        t.cla()

        df_operations.plot(ax=t, kind='bar',stacked=True)
        f.suptitle(exp +'_operations', fontsize=25)
        f.savefig(dir + exp +'_operations.eps',format='eps')
        t.cla()

        plt.close(f)

        f, t = plt.subplots(1,2)
        f.set_size_inches(30, 20)
        f.set_dpi(300)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        df_dispatch.columns = ['cycles simd running','others stalls','barrier stall','memory fence stall','instruction infly stall']
        df_dispatch.plot(ax=t[0], kind='bar',stacked=True,title='cicles idle').legend(loc='upper left', bbox_to_anchor=(0.6, 1.15))
        #f.savefig(dir + exp +'_dispatch_stall.eps',format='eps')

        df_instructions_infly.plot(ax=t[1], kind='bar',stacked=True,title='instructions in fly').legend(loc='upper left', bbox_to_anchor=(0.6, 1.15))
        t[0].set_ylim(0,1)
        t[1].set_ylim(0,1)
        f.suptitle(exp +'_dispatch_stall', fontsize=25)
        f.savefig(dir + exp +'_dispatch_stall.eps',format='eps', bbox_inches='tight')

        plt.close(f)

    #grafica para comparar los distintos experimentos


    for i in BENCHMARKS:

        f, t = plt.subplots()
        f.set_size_inches(30, 20)
        f.set_dpi(300)

        df_exp.xs(i,level=1).plot(ax=t, kind='bar',stacked=True,title=i)
        f.savefig(dir + 'exp_'+i+'.eps',format='eps',bbox_inches='tight')
        t.cla()
        plt.close(f)




    return


'''def analisis_stall(directorioResultados, monton):

    dir = directorioResultados+'/barras OPC/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    f, t = plt.subplots()
    f.set_size_inches(30, 20)
    f.set_dpi(300)
    for exp in sorted(monton.keys()) :

        df_aux = pd.DataFrame(index=sorted(monton[exp].keys()), columns=['no stall','stall mem access','stall barrier','stall instruction infly','stall fetch buffer full','stall no wavefront','stall others'])

        for bench in sorted(monton[exp].keys()):
            try:
                df = pd.DataFrame(monton[exp][bench]['con L1'].df.max()).transpose()

                for col in ['no stall','stall mem access','stall barrier','stall instruction infly','stall fetch buffer full','stall no wavefront','stall others']:

                    df_aux.loc[bench,col] = (df[col+str(0)][0] + df[col+str(1)][0] + df[col+str(2)][0] + df[col+str(3)][0]+ df[col+str(4)][0]) / 5

            except Exception as e:
                print('Fallo analisis stall(): '+exp+' '+bench)
                print(e)

        df_aux = df_aux.div(df_aux.sum(1), axis=0)
        df_aux.plot(ax=t, kind='bar',stacked=True)

        f.savefig(dir + exp+'stall.eps',format='eps')

    t.cla()
    plt.close(f)

    return'''

def plot_axis(ax, df, legend, title, xlabel, ylabel):

    df.plot(ax=ax,legend=legend,title=title)

    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom = 0)
    ax.set_xlabel(xlabel)
    return

def OPCmultitest():
    return

def IPCmultitest(directorioResultados, monton):

    if not os.path.exists(directorioResultados):
        os.mkdir(directorioResultados)

    dirLatencias = directorioResultados+'/latencias/'
    if not os.path.exists(dirLatencias):
        os.mkdir(dirLatencias)

    tamanyoGrupo = 500000
    for bench in BENCHMARKS :

        f_lat, t_lat = plt.subplots(2,4)
        f_lat.set_size_inches(30, 20)
        f_lat.set_dpi(300)
        t_lat[0][0].set_title('lock mshr')
        t_lat[0][1].set_title('lock dir')
        t_lat[0][2].set_title('finish')

        f, t= plt.subplots(2,3)
        f.set_size_inches(30, 20)
        f.set_dpi(300)
        t[0][0].set_title('OPC')
        t[0][1].set_title('Latencia')
        t[1][0].set_title('tiempo vector mem buffer full')
        t[1][1].set_title('tiempo simd idle')
        t[0][2].set_title('Tamaño MSHR')

        aux = pd.DataFrame()
        #aux = None;
        for exp in nombre_resumen :


            try:
                for clave in ['con L1'] :

                    df = monton[exp][bench][clave].df
                    ind = pd.Index(df['ciclos_totales'], name='Operaciones')
                    tamanyoGrupo = 10000 * df.shape[0]

                    df2 = df.set_index(ind)
                    #.groupby( lambda x :  (x // tamanyoGrupo) * tamanyoGrupo).sum() #.astype(np.int32)
                    df_cumsum = df2.cumsum()

                    if exp in nombre_resumen[0:]:
                        pd.DataFrame({exp:df_cumsum['total_intervalo']/df_cumsum['ciclos_intervalo']}).plot(ax=t[1][2],legend=False)
                        pd.DataFrame({exp:df2['mshr_size_L1']}).plot(ax=t[0][2],legend=False)
                        pd.DataFrame({exp:df2['lat_loads_gpu']/ df2['num_loads_gpu']}).plot(ax=t[0][1],legend=False)
                        pd.DataFrame({exp:df2['total_intervalo']/ df2['ciclos_intervalo']}).plot(ax=t[0][0])
                        if 'simd_idle1' in df2.columns:
                            dato = pd.DataFrame({exp:((df2['simd_idle1'] + df2['simd_idle2'] + df2['simd_idle3'] + df2['simd_idle4']) / (4 * 5)) / df2['ciclos_intervalo']}).plot(ax=t[1][1],legend=False)
                            dato.plot(ax=t[1][1],legend=False)
                        #LATENCIAS
                        pd.DataFrame({exp:df2['lock_mshr_load']/df2['access_load']}).plot(ax=t_lat[0][0])
                        pd.DataFrame({exp:(df2['lock_dir_load'] + df2['eviction_load'])/df2['access_load']}).plot(ax=t_lat[0][1],legend=False)
                        pd.DataFrame({exp:df2['lock_mshr_load']/df2['access_load'] + (df2['lock_dir_load'] + df2['eviction_load'])/df2['access_load']}).plot(ax=t_lat[0][2],legend=False)
                        pd.DataFrame({exp:df2['finish_load']/df2['access_load']}).plot(ax=t_lat[0][3],legend=False)

                        pd.DataFrame({exp:df2['hits_L1']/df2['efectivos_L1']}).plot(ax=t_lat[1][0],legend=False,title='HitRatio L1')
                        pd.DataFrame({exp:df2['hits_L2']/df2['efectivos_L2']}).plot(ax=t_lat[1][1],legend=False,title='HitRatio L2')


                        pd.DataFrame({exp:df2['lat_loads_gpu']/df2['num_loads_gpu']}).plot(ax=t_lat[1][2],legend=False,title='Latencia GPU')
                        pd.DataFrame({exp:df2['Coalesces_L1']/df2['accesos_L1']}).plot(ax=t_lat[1][3],legend=False,title='coalesce GPU')
                        t_lat[1][0].set_ylim([0,1])
                        t_lat[1][1].set_ylim([0,1])
                        t_lat[1][2].set_ylim(bottom = 0)
                        t_lat[1][3].set_ylim([0,1])
                        '''pd.DataFrame({exp:df2['lock mshr nc write']/df2['access nc write']}).plot(ax=t_lat[1][0],legend=False)
                        pd.DataFrame({exp:df2['evicted dir nc write']/df2['access nc write']}).plot(ax=t_lat[1][1],legend=False)
                        pd.DataFrame({exp:df2['lock mshr nc write']/df2['access nc write'] + df2['evicted dir nc write']/df2['access nc write']}).plot(ax=t_lat[1][2],legend=False)
                        pd.DataFrame({exp:df2['finish nc write']/df2['access nc write']}).plot(ax=t_lat[1][3],legend=False)'''
                        #continue

                    #OPC
                    #dato = pd.DataFrame({exp:df2['total intervalo']/ df2['ciclos intervalo']})

                    #if exists(opcOptimo)
                    #opcOptimo.merge(dato)
                    #dato = dato.set_index(ind).groupby( lambda x :  (x // tamanyoGrupo) * tamanyoGrupo).mean()
                    #dato.plot(ax=t[0][0])

                    #latencia
                    #dato = pd.DataFrame({exp:df2['lat loads gpu']/ df2['num loads gpu']})
                    #dato.plot(ax=t[0][1])

                    #accesos
                    #dato = pd.DataFrame({exp:df2['efectivos L1']/df2['ciclos intervalo']}).plot(ax=t[1][1])
                    #dato.plot(ax=t[1][1])

                    #if 'simd idle1' in df2.columns:
                    #    dato = pd.DataFrame({exp:((df2['simd idle1'] + df2['simd idle2'] + df2['simd idle3'] + df2['simd idle4']) / (4 * 5)) / df2['ciclos intervalo']}).plot(ax=t[1][1])
                    #    dato.plot(ax=t[1][1])

                    #opc acumulado
                    dato = pd.DataFrame({exp:df_cumsum['total_intervalo']/df_cumsum['ciclos_intervalo']})
                    aux = aux.join(dato, how="right")

                    #v_mem_full
                    dato = pd.DataFrame({exp:(df2['v_mem_full'] / 5)/df2['ciclos_intervalo']})
                    dato.plot(ax=t[1][0],legend=False)


                    #hitRatio
                    #pd.DataFrame({exp:(df2['hits L1']/ df2['efectivos L1'])}).replace({np.inf:0}).plot(ax=t[0][2])

                    #tamaño mshr
                    #pd.DataFrame({exp:df2['mshr size L1']}).plot(ax=t[0][2])

                    #plt.legend()

                    #LATENCIAS
                    #pd.DataFrame({exp:df2['lock mshr load']/df2['access load']}).plot(ax=t_lat[0][0])
                    #pd.DataFrame({exp:df2['evicted dir load']/df2['access load']}).plot(ax=t_lat[0][1])
                    #pd.DataFrame({exp:df2['lock mshr load']/df2['access load'] + df2['evicted dir load']/df2['access load']}).plot(ax=t_lat[0][2])
                    #pd.DataFrame({exp:df2['finish load']/df2['access load']}).plot(ax=t_lat[0][3])
                    #pd.DataFrame({exp:df2['lock mshr nc write']/df2['access nc write']}).plot(ax=t_lat[1][0])
                    #pd.DataFrame({exp:df2['evicted dir nc write']/df2['access nc write']}).plot(ax=t_lat[1][1])
                    #pd.DataFrame({exp:df2['lock mshr nc write']/df2['access nc write'] + df2['evicted dir nc write']/df2['access nc write']}).plot(ax=t_lat[1][2])
                    #pd.DataFrame({exp:df2['finish nc write']/df2['access nc write']}).plot(ax=t_lat[1][3])

            except IOError as e:
                print('fallo al crear grafica opc para:')
                print(e)
            except Exception as e:
                print('Fallo grafica ipcmultitest en el test: '+bench+' '+clave)
                print(e)

        try:
            aux.max(1).plot(ax=t[1][2],style=['k--'],legend=False)
            aux.min(1).plot(ax=t[1][2],style=['k--'],legend=False)

            #pd.DataFrame(aux[nombre_resumen[3]]).plot(ax=t[1][2])
        except Exception as e:
            print('Fallo ipcmultitest grafica maximos y minimos en el test: '+exp)
            print(e)


        t[0][0].set_ylabel('OPC')
        t[0][0].set_ylim(bottom = 0)
        t[0][0].set_xlabel('Operaciones ejecutadas')
        #opcOptimo.plot(ax=t[0][2])
        #t[0].set_xticks([])
        #f[0].savefig(directorioResultados+bench+'_OPCmultitest')

        t[0][1].set_ylabel('Latencia (ciclos GPU)')
        t[0][1].set_ylim(bottom = 0)
        t[0][1].set_xlabel('Operaciones ejecutadas')
        #t[1].set_xticks([])

        t[1][0].set_ylabel('% tiempo vector mem buffer full')
        t[1][0].set_ylim(bottom = 0)
        t[1][0].set_xlabel('Operaciones ejecutadas')



        t[1][1].set_ylabel('% medio de ciclos simd idle')
        t[1][1].set_ylim(bottom = 0)
        t[1][1].set_xlabel('Operaciones ejecutadas')

        t[0][2].set_ylabel('Cantidad de entradas')
        t[0][2].set_ylim(bottom = 0)
        t[0][2].set_xlabel('Operaciones ejecutadas')

        t[1][2].set_ylabel('OPC')
        t[1][2].set_ylim(bottom = 0)
        t[1][2].set_xlabel('Operaciones ejecutadas')

        max_load = max([t_lat[0][0].get_ylim()[1], t_lat[0][1].get_ylim()[1], t_lat[0][2].get_ylim()[1]])
        max_write = max([t_lat[1][0].get_ylim()[1], t_lat[1][1].get_ylim()[1], t_lat[1][2].get_ylim()[1]])

        t_lat[0][0].set_ylim(bottom = 0, top = max_load)
        t_lat[0][1].set_ylim(bottom = 0, top = max_load)
        t_lat[0][2].set_ylim(bottom = 0, top = max_load)
        t_lat[0][3].set_ylim(bottom = 0, top = max_load)
        '''t_lat[1][0].set_ylim(bottom = 0, top = max_write)
        t_lat[1][1].set_ylim(bottom = 0, top = max_write)
        t_lat[1][2].set_ylim(bottom = 0, top = max_write)
        t_lat[1][3].set_ylim(bottom = 0, top = max_write)
        '''



        f.suptitle(bench+' (intervalo = '+str(tamanyoGrupo)+')', fontsize=25)
        f.savefig(directorioResultados+bench+'_OpcMultitest-'+str(tamanyoGrupo)+'.eps',format='eps')

        f_lat.suptitle(bench+' (intervalo = '+str(tamanyoGrupo)+')', fontsize=25)
        f_lat.savefig(dirLatencias+bench+'_Multitest-'+str(tamanyoGrupo)+'.eps',format='eps')
        #t[0].cla()
        #t[1].cla()
        plt.close(f)
        plt.close(f_lat)
    return



def IPCacumulado(nombreArchivo):
    f, t= plt.subplots(1)
    #for test in grupo:

    if not os.path.exists(directorio_salida+'/graficos/ipc'):
        os.mkdir(directorio_salida+'/graficos/ipc')

    for bench in BENCHMARKS :
        t.set_title(bench+' - IPC acumulado')
        try:
            for clave in sorted(TESTS.keys()) :
                dato = aux[bench][clave].df['total global']
                #pd.DataFrame({clave :np.loadtxt( contenedor_de_datos+'/'+TESTS[clave]+'/'+bench+'-_ipc',delimiter = '\n')})
                #dato = pd.DataFrame({clave : aux[bench][clave].df['ipc']})
                dato = dato.groupby( lambda x :  x // (np.ceil(dato.shape[0]/100))).mean()
                dato.plot(ax=t )

            t.set_ylabel('IPC')
            t.set_xlabel('Instrucciones ejecutadas')
            t.set_xticks([])
            #'''np.arange(np.ceil(dato.shape[0]/100))*50000''')
            f.savefig(nombreArchivo+bench+'-acumulado')
        except IOError as e:
            print('fallo al crear grafica ipc para:')
            print(e)
        except Exception as e:
            print('Fallo grafica ipc en el test: '+bench+' '+clave)
            print(e)
        t.cla()
    plt.close(f)
    return

'''def dibujar4tablas(nombreArchivo,monton):

    #for test in grupo:
    if not os.path.exists(nombreArchivo):
        os.mkdir(nombreArchivo)

    for bench in BENCHMARKS :

        f, t= plt.subplots(2,2)
        f.get_size_inches()
        f.set_size_inches(20, 14)
        f.set_dpi(50)

        try:
            for clave in TESTS.keys() :
                #dato = pd.DataFrame({clave : monton[bench][clave].df['Coalesces L1']})
                #dato = pd.DataFrame({clave : aux[bench][clave].df['ipc']})

                #dato = dato.groupby( lambda x :  x // resolucion).mean()
                #dato.plot(ax=t[0][0])

                #dato = pd.DataFrame({clave : monton[bench][clave].df['accesos L1']})
                #dato = pd.DataFrame({clave : aux[bench][clave].df['ipc']})
                #dato = dato.groupby( lambda x :  x // resolucion).mean()
                #dato.plot(ax=t[0][0],secondary_y=True)

                t[0][0].set_title('Coalesces / accesos efectivos')
                t[0][0].set_ylabel('(coalesces / accesos) en L1')

                efectivos = monton[bench][clave].df['accesos L2'] - monton[bench][clave].df['Coalesces L2']
                dato = pd.DataFrame({clave: (monton[bench][clave].df['Coalesces L2'] / efectivos)})
                resolucion = np.ceil(dato.shape[0]/100)

                dato = dato.groupby( lambda x :  x // resolucion).mean()
                dato.plot(ax=t[0][0],secondary_y=True)
                t[0][0].right_ax.set_ylabel('(coalesces / accesos) en L2')

                efectivos = monton[bench][clave].df['accesos L1'] - monton[bench][clave].df['Coalesces L1']
                dato = pd.DataFrame({clave: (monton[bench][clave].df['Coalesces L1'] / efectivos )})

                dato = dato.groupby( lambda x :  x // resolucion).mean()
                dato.plot(ax=t[0][0])

                dato = pd.DataFrame({clave : monton[bench][clave].df['MPKI L2']})
                #dato = pd.DataFrame({clave : aux[bench][clave].df['ipc']})
                dato = dato.groupby( lambda x :  x // resolucion).mean()
                dato.plot(ax=t[0][1],secondary_y=True)

                dato = pd.DataFrame({clave : monton[bench][clave].df['MPKI L1']})
                #dato = pd.DataFrame({clave : aux[bench][clave].df['ipc']})
                dato = dato.groupby( lambda x :  x // resolucion).mean()
                dato.plot(ax=t[0][1])

                dato = pd.DataFrame({clave : monton[bench][clave].df['HR L1']})
                #dato = pd.DataFrame({clave : aux[bench][clave].df['ipc']})
                dato = dato.groupby( lambda x :  x // resolucion).mean()
                dato.plot(ax=t[1][0])

                dato = pd.DataFrame({clave : monton[bench][clave].df['HR L2']})
                #dato = pd.DataFrame({clave : aux[bench][clave].df['ipc']})
                dato = dato.groupby( lambda x :  x // resolucion).mean()
                dato.plot(ax=t[1][1])

            f.set_label('LABEL DE LA FIGURA')
            #t[0][0].set_title('Coalesces y accesos L1')
            t[0][1].set_title('MPKI')
            t[1][0].set_title('HR L1')
            t[1][1].set_title('HR L2')

            #t[0][0].set_ylabel('cantidad de coalesces')
            #t[0][0].right_ax.set_ylabel('accesos')
            #t[0][0].set_xlabel('Instrucciones ejecutadas')

            t[0][1].set_ylabel('Misses / 1000 inst')
            t[0][1].right_ax.set_ylabel('MPKI L2')
            #t[0][1].set_xlabel('Instrucciones ejecutadas')

            t[1][0].set_ylabel('hit ratio % (base 1)')
            #t[1][0].set_xlabel('Instrucciones ejecutadas')

            t[1][1].set_ylabel('hit ratio % (base 1)')
            #t[1][1].set_xlabel('Instrucciones ejecutadas')
            #t[0][0].set_xticks(np.arange(np.ceil(dato.shape[0]/100))*50000)

            f.savefig(nombreArchivo+'/'+bench)



        except IOError as e:
            print('fallo al crear grafica ipc para:')
            print(e)
        except Exception as e:
            print('Fallo dibujar4tablas en el test: '+bench+' '+clave)
            print(e)

        plt.close(f)
    return
'''

def dibujar4tablas(nombreArchivo,monton,exp):

    #for test in grupo:
    if not os.path.exists(nombreArchivo):
        os.mkdir(nombreArchivo)

    TESTS = {'con L1':exp+'_conL1'}

    for bench in BENCHMARKS :

        f, t= plt.subplots(2,3)
        f.set_size_inches(30, 15)
        f.set_dpi(150)

        try:
            # grafica coalesces / efectivos para L1 y L2(izquierda)
            t[0][0].set_title('Coalesces / accesos efectivos')

            for clave in sorted(TESTS.keys()) :

                df = monton[bench][clave].media

                efectivos = df['efectivos L2']
                dato = pd.DataFrame({'L2 ('+ clave +')': (df['Coalesces L2'] / efectivos)})

                dato.plot(ax=t[0][0])

                efectivos = df['efectivos L1']
                dato = pd.DataFrame({'L1 ('+ clave +')': (df['Coalesces L1'] / efectivos )})

                dato.plot(ax=t[0][0])

                efectivos = pd.DataFrame({'as':df['accesos gpu'] - df['Coalesces gpu']})

                dato = pd.DataFrame({'gpu ('+ clave +')': (df['Coalesces gpu'] / efectivos['as'] )})

                dato.plot(ax=t[0][0])

            t[0][0].set_ylabel('(coalesces / accesos efectivos)')
            #t[0][0].right_ax.set_ylabel('(coalesces / accesos) en L2')
            t[0][0].set_xlabel('Instrucciones ejecutadas')

            # grafica MKPI L1 y L2(izquierda)
            t[0][1].set_title('MPKI')

            for clave in sorted(TESTS.keys()) :

                df = monton[bench][clave].media

                dato = pd.DataFrame({'L2 ('+ clave +')': df['misses L2']/10})
                dato.plot(ax=t[0][1])

                dato = pd.DataFrame({'L1 ('+ clave +')' : df['misses L1']/10})
                dato.plot(ax=t[0][1])

            t[0][1].set_ylabel('MPKI')
            #t[0][1].right_ax.set_ylabel('MPKI L2')
            t[0][1].set_xlabel('Instrucciones ejecutadas')

            #IPC
            t[1][1].set_title('IPC')
            for clave in sorted(TESTS.keys()) :

                df = monton[bench][clave].sum

            #    dato = pd.DataFrame({clave : df['total intervalo']/ df['ciclos intervalo']})
                dato = pd.DataFrame({clave : df['total intervalo']/ (df['ciclos intervalo'] )})
                dato.plot(ax=t[1][1])
            t[1][1].set_ylabel('IPC')
            t[1][1].set_ylim(bottom = 0)
            t[1][1].set_xlabel('Instrucciones ejecutadas')

            #Graficas HR L1 y HR L2
            t[1][0].set_title('HitRatio L1 y L2')
            #t[1][1].set_title('hits L2')
            for clave in sorted(TESTS.keys()) :

                df = monton[bench][clave].media

                efectivos = df['efectivos L1']
                dato = pd.DataFrame({'HR L1 ('+clave+')' : df['hits L1'] / efectivos})
                dato.plot(ax=t[1][0])

                efectivos = df['efectivos L2']
                dato = pd.DataFrame({'HR L2 ('+clave+')' : df['hits L2'] / efectivos})
                dato.plot(ax=t[1][0])

            t[1][0].set_ylabel('hit ratio % (base 1)')
            #t[1][1].set_ylabel('hit ratio % (base 1)')
            t[1][0].set_xlabel('Instrucciones ejecutadas')
            #t[1][1].set_xlabel('Instrucciones ejecutadas')

            #grafica latencias de red para L1 y L2(izqueirda)
            ''' t[0][2].set_title('Latencia red')
            for clave in sorted(TESTS.keys()) :

                df = monton[bench][clave].media

                paquetes = df['paquetes L2-MM']
                dato = pd.DataFrame({'red L2-MM ('+ clave+')': ((df['lat L2-MM'] / 8) / paquetes)})

                dato.plot(ax=t[0][2])

                paquetes = df['paquetes L1-L2']
                dato = pd.DataFrame({'red L1-L2 ('+ clave+')': ((df['lat L1-L2'] / 8) / paquetes)})

                dato.plot(ax=t[0][2])

            t[0][2].set_ylabel('ciclos de latencia (L1-L2)')
            #t[0][2].right_ax.set_ylabel('ciclos de latencia (L2-MM)')
            t[0][2].set_xlabel('Instrucciones ejecutadas')'''

            #grafica latencias de red para L1 y L2(izqueirda)
            t[1][2].set_title('Latencia load')
            for clave in sorted(TESTS.keys()) :

                df = monton[bench][clave].media

                latencia = df['lat loads gpu']
                peticiones = df['num loads gpu']
                dato = pd.DataFrame({'GPU loads ('+ clave+')': latencia / peticiones})

                dato.plot(ax=t[1][2])

                latencia = df['lat loads mem']
                peticiones = df['num loads mem']
                dato = pd.DataFrame({'MEM-SYSTEM loads ('+ clave+')': latencia / peticiones})

                dato.plot(ax=t[1][2])

            t[1][2].set_ylabel('ciclos de latencia (GPU)')
            #t[1][2].right_ax.set_ylabel('ciclos de latencia (MEM-SYSTEM)')
            t[1][2].set_xlabel('Instrucciones ejecutadas')


            #grafica latencias de red para L1 y L2(izqueirda)
            ''' t[0][3].set_title('load efectivas')
            for clave in sorted(TESTS.keys()) :

                df = monton[bench][clave].sum

                dato = pd.DataFrame({'loads/instruccion ('+ clave+')': df['efectivos L1']/df['total intervalo']})

                dato.plot(ax=t[0][3])

            t[0][3].set_ylabel('cantidad de loads')
            #t[1][2].right_ax.set_ylabel('ciclos de latencia (MEM-SYSTEM)')
            t[0][3].set_xlabel('Instrucciones ejecutadas')
            t[0][3].set_ylim(bottom = 0)'''

            #grafica latencias de red para L1 y L2(izqueirda)
            t[0][2].set_title('entradas bloqueadas')
            for clave in sorted(TESTS.keys()) :

                df = monton[bench][clave].media

                dato = pd.DataFrame({'L1 ('+ clave+')': df['entradas bloqueadas L1'], 'L2 ('+ clave+')': df['entradas bloqueadas L2']})

                dato.plot(ax=t[0][2])

            t[0][2].set_ylabel('numero de entradas bloqueadas')
            #t[1][2].right_ax.set_ylabel('ciclos de latencia (MEM-SYSTEM)')
            t[0][2].set_xlabel('Instrucciones ejecutadas')

            f.suptitle(EXPERIMENTO +' - '+ bench, fontsize=25)
            f.savefig(nombreArchivo+'/'+bench)



        except IOError as e:
            print('fallo al crear grafica ipc para:')
            print(e)
        except Exception as e:
            print('Fallo dibujar4tablas en el test: '+bench+' '+clave)
            print(e)
        t[0][0].cla()
        t[0][1].cla()
        t[0][2].cla()
        #t[0][3].cla()
        t[1][0].cla()
        t[1][1].cla()
        t[1][2].cla()
        #t[1][3].cla()
        plt.close(f)
    return

def dibujar4tablas2(nombreArchivo,monton):

    #for test in grupo:
    if not os.path.exists(nombreArchivo):
        os.mkdir(nombreArchivo)

    for bench in BENCHMARKS :

        f, t= plt.subplots(2,3)
        f.set_size_inches(20, 10)
        f.set_dpi(150)

        try:
            # nivel de paralelismo
            t[0][0].set_title('nivel de paralalismo (max 64)')

            df = monton[bench]['con L1'].sum

            simd = df['i simd'] / df['mi simd']
            v_mem = df['i v mem'] / df['mi v mem']

            dato = pd.DataFrame({'SIMD': simd})
            dato.plot(ax=t[0][0])
            dato = pd.DataFrame({'Memoria': v_mem})
            dato.plot(ax=t[0][0])

            t[0][0].set_ylim(bottom = 0)
            t[0][0].set_ylabel('(thread)')
            t[0][0].set_xlabel('Instrucciones ejecutadas')


            # reads y write
            t[0][1].set_title('read y write')


            df = monton[bench]['con L1'].media
            write = df['writesL1']
            load = df['loadsL1']

            dato = pd.DataFrame({'writes': write})
            dato.plot(ax=t[0][1])
            dato = pd.DataFrame({'loads': load})
            dato.plot(ax=t[0][1])


            t[0][1].set_ylim(bottom = 0)
            t[0][1].set_ylabel('cantidad de accesos efectivos')
            t[0][1].set_xlabel('Instrucciones ejecutadas')

            # Coalesce Gpu
            t[1][1].set_title('Coalesce GPU')

            df = monton[bench]['con L1'].sum

            dato = pd.DataFrame({'Coalesce': (df['Coalesces gpu'])})

            dato.plot(ax=t[0][2])
            dato = pd.DataFrame({'accesses': (df['accesos gpu'])})
            dato.plot(ax=t[0][2])

            t[1][1].set_ylim(bottom = 0)
            t[1][1].set_ylabel('(HR con coalesces)')
            t[1][1].set_xlabel('Instrucciones ejecutadas')

            f.suptitle(bench, fontsize=25)
            f.savefig(nombreArchivo+'/'+bench+'HRcoalesce')

        except IOError as e:
            print('fallo al crear grafica ipc para:')
            print(e)
        except Exception as e:
            print('Fallo dibujar4tablas2 en el test: '+bench)
            print(e)

        plt.close(f)
    return


def comparar_velocidad(nombreArchivo):
    f, t= plt.subplots(1)
    #for test in grupo:
    t.set_title('comparacion lat ')
    TESS = {'10_lat_conL1':'05-15_prueba_10_lat_conL1','100_lat_conL1':'05-15_prueba_100_lat_conL1'}

    for bench in BENCHMARKS :

        try:
            dato = pd.DataFrame({'10_lat_conL1' : np.loadtxt( contenedor_de_datos+'/'+TESS['10_lat_conL1']+'/'+bench+'-fran_t10000k',delimiter = '\n')})
            dato1 = pd.DataFrame({'100_lat_conL1' : np.loadtxt( contenedor_de_datos+'/'+TESS['100_lat_conL1']+'/'+bench+'-fran_t10000k',delimiter = '\n')})
            #dato = dato['10_lat_conL1'] - dato1['100_lat_conL1']
            dato = dato.join(dato1)
            dato.plot(ax=t)

            t.set_ylabel('ciclos')
            t.set_xlabel('loads')
            t.set_xticks([])
            f.savefig(nombreArchivo+bench)
        except IOError as e:
            print('fallo al crear grafica ipc para:')
            print(e)
        except Exception as e:
            print('Fallo grafica ipc en el test: '+bench)
            print(e)
        t.cla()
    plt.close(f)
    return

class contenedor:
        pass

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def loadDictCompletomultiprocesses2(sufijo, contenedor_de_datos, exp, bench, tip=None):

    TESTS = {'con L1':exp+'_conL1'}
    return loadworker(bench,sufijo,TESTS,contenedor_de_datos)


def loadDictCompletomultiprocesses(sufijo, contenedor_de_datos, exp, tip=None):
    directorio_salida = contenedor_de_datos+'/'+exp+'_resumen'

    #TESTS = {'sin L1':EXPERIMENTO+'_sinL1','con L1':EXPERIMENTO+'_conL1'}
    TESTS = {'con L1':exp+'_conL1'}
    DIR_GRAFICOS = directorio_salida+'/graficos'



    # crear directorios
    if not os.path.exists(directorio_salida):
        os.mkdir(directorio_salida)
    if not os.path.exists(directorio_salida+'/tablas'):
        os.mkdir(directorio_salida+'/tablas')
    if not os.path.exists(directorio_salida+'/graficos'):
        os.mkdir(directorio_salida+'/graficos')
    args = []
    for bench in BENCHMARKS :
        args.append((bench,sufijo,TESTS,contenedor_de_datos))

    pool = multiprocessing.Pool()
    resultados_load = pool.starmap(loadworker, args)
    print('lanzado 2')
    pool.close()
    pool.join()
    print(resultados_load)
    dict_general = dict(zip(BENCHMARKS,resultados_load))
    return dict_general

def loadworker(bench,sufijo, TESTS, contenedor_de_datos):
    dict_general = dict()

    print('cargando -> '+bench)
    tip = None
    for clave in sorted(TESTS.keys()) :
        archivo = contenedor_de_datos+'/'+TESTS[clave]+'/'+bench+sufijo

        dict_general[clave] = contenedor()
        try:
            f = open(contenedor_de_datos+'/'+TESTS[clave]+'/condor_log/'+bench+'.err')
            simend = buscar_string(f, 'SimEnd')
            f.close()
            if 'ContextsFinished' == simend :
                df = pd.read_csv(archivo,sep = ' ', header = 0)

                #tamanyoGrupo = np.ceil(df.shape[0]/100)
                #df = df.groupby( lambda x :  (x // tamanyoGrupo) * tamanyoGrupo).sum()

                dict_general[clave].df = df
        except Exception as e:
            print('Fallo al cargar el archivo -> '+archivo)
            print(e)
    return dict_general

def loadDictCompleto(sufijo, contenedor_de_datos, exp, tip=None):


    directorio_salida = contenedor_de_datos+'/'+exp+'_resumen'

    #TESTS = {'sin L1':EXPERIMENTO+'_sinL1','con L1':EXPERIMENTO+'_conL1'}
    TESTS = {'con L1':exp+'_conL1'}
    DIR_GRAFICOS = directorio_salida+'/graficos'



    # crear directorios
    if not os.path.exists(directorio_salida):
        os.mkdir(directorio_salida)
    if not os.path.exists(directorio_salida+'/tablas'):
        os.mkdir(directorio_salida+'/tablas')
    if not os.path.exists(directorio_salida+'/graficos'):
        os.mkdir(directorio_salida+'/graficos')

    dict_general = dict()

    for bench in BENCHMARKS :
        dict_general[bench] = dict()

        print('cargando -> '+bench)

        for clave in sorted(TESTS.keys()) :
            archivo = contenedor_de_datos+'/'+TESTS[clave]+'/'+bench+sufijo

            dict_general[bench][clave] = contenedor()
            try:
                f = open(contenedor_de_datos+'/'+TESTS[clave]+'/condor_log/'+bench+'.err')
                simend = buscar_string(f, 'SimEnd')
                f.close()
                if 'ContextsFinished' == simend :
                    if tip == None :
                        tip = leerNombreDatos(archivo)
                        print(tip.names)

                    #df = pd.read_table(archivo)
                    df = pd.DataFrame(np.loadtxt(archivo,comments='#', skiprows=1, delimiter = ' ', dtype=tip))
                    #df = df.groupby( lambda x :  x // (np.ceil(df.shape[0]/30))).mean()
                    #dict_general[bench][clave] = ajustarResolucionPorTamanyo(df, 50)
                    #dict_general[bench][clave] = ajustarResolucion(df, 100)
                    dict_general[bench][clave].df = df
                    #dict_general[bench][clave].lista = loadDatosSueltos(archivo, dict_general[bench][clave].df)
            except Exception as e:
                print('Fallo al cargar el archivo -> '+archivo)
                print(e)
    return dict_general

def loadDatosSueltos(dict_c, dict_i):


    col= ['benchmark','test','estado','acceso gpu','coalesces gpu','Loads L1','coalesces L1','Loads L2','coalesces L2','ciclos','Macro IPC','IPC','latencia mem CU','hrL1','mpkiL1','evictions L2','evictions Sharers L2'''','latencia red L1-L2','latencia red L2-MM','blk compartidos','replicas''']
    df2 = pd.DataFrame(columns=col)
    for bench in BENCHMARKS :
        tabla = []
        for clave in sorted(TESTS.keys()) :
            lista = []
            finishStatus = 'NO_FINISHED'
            try:
                # test
                lista.append(bench)
                lista.append(clave)

                f = open(contenedor_de_datos+'/'+TESTS[clave]+'/condor_log/'+bench+'.err')
                finishStatus = buscar_string(f, 'SimEnd')
                if finishStatus == '' :
                    finishStatus = 'NO_FINISHED'
                lista.append(finishStatus)
                f.close()

                df = dict_i[bench][clave].df.astype('float')

                #acceso gpu
                lista.append(df['accesos gpu'].sum())

                #coalesced gpu
                lista.append(df['Coalesces gpu'].sum())


                # Loads L1
                f = open(contenedor_de_datos+'/'+TESTS[clave]+'/'+bench+'-mem-report')
                #lista.append(buscar_y_acumular(f,'[ l1','fffffff','Loads'))
                lista.append(df['accesos L1'].sum())
                f.close()

                #coalesced L1
                lista.append(df['Coalesces L1'].sum())

                #Loads desde L1 a L2
                lista.append(df['accesos L2'].sum())

                #coalesced L2
                lista.append(df['Coalesces L2'].sum())



                # ciclos
                f = open(contenedor_de_datos+'/'+TESTS[clave]+'/'+bench+'-si-report')
                ciclo_temp  = buscar_valor(f,'Cycles')
                lista.append(ciclo_temp)
                f.close()

                # Macro IPC
                f = open(contenedor_de_datos+'/'+TESTS[clave]+'/'+bench+'-si-report')
                instruciones = buscar_valor(f,'Instructions')
                lista.append(instruciones/ float(ciclo_temp))
                f.close()

                # IPC
                lista.append(df['total intervalo'].sum()/ float(ciclo_temp))

                # latencia CU mem
                lista.append(df['lat loads gpu'].sum()/ df['num loads gpu'].sum())

                # HR
                lista.append(df['hits L1'].sum() / df['efectivos L1'].sum())


                # MPKI
                lista.append(df['misses L1'].mean()/10)


                # Evictions en L2
                f = open(contenedor_de_datos+'/'+TESTS[clave]+'/'+bench+'-mem-report')
                lista.append(buscar_y_acumular(f,'[ l2','[ l1','Evictions'))
                f.close()

                # Evitions with sharers
                f = open(contenedor_de_datos+'/'+TESTS[clave]+'/'+bench+'-mem-report')
                lista.append(buscar_y_acumular(f,'[ l2','[ l1','EvictionsWithSharers'))
                f.close()

                # latencias red L1-L2 L2-MM
                '''paq = dict_c[bench][clave].df['paquetes L1-L2'].sum()

                if paq > 0 :
                    lista.append((dict_c[bench][clave].df['lat L1-L2'].sum()/8)/ paq)
                else:
                    lista.append(0)


                paq = dict_c[bench][clave].df['paquetes L2-MM'].sum()
                if paq > 0 :
                    lista.append((dict_c[bench][clave].df['lat L2-MM'].sum()/8)/ paq)
                else:
                    lista.append(0)


                #bloques compartidos y replicas

                lista.append(dict_c[bench][clave].df['blk compartidos'].mean())
                lista.append(dict_c[bench][clave].df['blk replicas'].mean())
                '''


                '''

                lista.append(stats.nanmean(df['latencia'].values))


                f = open(DIRECTORIO_RAIZ+'/'+j+'/'+i+'-si-report')
                instruciones = buscar_valor(f,'Instructions')
                lista.append(instruciones/ float(ciclo_temp))
                f.close()

                f = open(DIRECTORIO_RAIZ+'/'+j+'/'+i+'-mem-report')
                lista.append(buscar_y_acumular(f,'[ l2','[ l1','Evictions'))
                f.close()

                f = open(DIRECTORIO_RAIZ+'/'+j+'/'+i+'-mem-report')
                lista.append(buscar_y_acumular(f,'[ l2','[ l1','EvictionsWithSharers'))
                f.close()

                f = open(DIRECTORIO_RAIZ+'/'+j+'/'+i+'-mem-report')
                lista.append(buscar_y_acumular(f,'[ l2','[ l1','EvictionsSharersInvalidation'))
                f.close()
                if str(j).find('sinL1') == -1 :

                    lista.append(stats.nanmean(df['hr_loads_L2'].values))

                    lista.append(stats.nanmean(df['hr_loads_L1'].values))

                else:

                    lista.append(stats.nanmean(df['hr_loads_L1'].values))

                    lista.append(0)

                f = open(DIRECTORIO_RAIZ+'/'+j+'/'+i+'-mem-report')
                lista.append(buscar_y_acumular(f,'[ l1','fffffff','Loads'))
                f.close()

                f = open(DIRECTORIO_RAIZ+'/'+j+'/'+i+'-mem-report')
                lista.append(buscar_y_acumular(f,'[ l1','fffffff','CoalescedRead'))
                f.close()

                f = open(DIRECTORIO_RAIZ+'/'+j+'/'+i+'-mem-report')
                lista.append(buscar_y_acumular(f,'[ l2','[ l1','Loads'))
                f.close()

                if str(j).find('sinL1') == -1 :
                    lista.append(df.sum(0)['Coalesces L2'])
                else:
                    f = open(DIRECTORIO_RAIZ+'/'+j+'/'+i+'-mem-report')
                    lista.append(buscar_y_acumular(f,'[ l2','[ l1','CoalescedRead'))
                    f.close()

                   '''
            except Exception as e:
                print('problemas de lectura en el test: '+bench+' --> '+clave)
                print(e)
                lista = [bench,clave,finishStatus,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

            tabla.append(lista)

        df = pd.DataFrame(tabla, columns=col)

        df2 = df2.append(df)
    df2.set_index(['benchmark','test'],inplace=True)

    df2.to_excel(directorio_salida+'/'+EXPERIMENTO+'.xlsx',engine='xlsxwriter')

def ajustarResolucion(df, resolucion):

    dato = contenedor()
    dato.df = df
    try:
        tamanyoGrupo = np.ceil(df.shape[0]/resolucion)
        dato.sum = df.groupby( lambda x :  x // tamanyoGrupo).sum()
        dato.media = df.groupby( lambda x :  x // tamanyoGrupo).mean()
        #dato.media = df.groupby(funcionmagica(x,df)).mean()
        #return dato
    except Exception as e:
        print('exception en ajustarResolucion()')
        print(e)

    return dato

def ajustarResolucionPorTamanyo(df, tamanyoGrupo):

    dato = contenedor()
    dato.df = df
    try:
        dato.sum = df.groupby( lambda x :  x // tamanyoGrupo).sum()
        dato.media = df.groupby( lambda x :  x // tamanyoGrupo).mean()

    except Exception as e:
        print('exception en ajustarResolucioPorTamanyo()')
        print(e)

    return dato

def funcionmagica(x,df):
    tamanyoGrupo = np.ceil(df.shape[0]/resolucion)
    resolucion = 100
    if x > (df.shape[0] - (((tamanyoGrupo * resolucion) - df.shape[0]) * (tamanyoGrupo - 1))):
        return x // (tamanyGrupo - 1)
    else:
        return x // tamanyoGrupo

def leerNombreDatos(nombreArchivo):
    f = open(nombreArchivo)
    linea = f.readline()
    f.close()
    lista = []
    for i in linea.split(' '):
        #lista.append((i.strip(' \n\t\r').replace('_',' '),np.int64))
        lista.append((i.strip(' \n\t\r').replace('_',' '),np.float))

    return np.dtype(lista)

def tablas_access_list(nombreArchivo):


    if not os.path.exists(nombreArchivo):
        os.mkdir(nombreArchivo)

    for bench in BENCHMARKS :

        f, t= plt.subplots(1,3)
        f.set_size_inches(15, 8)
        f.set_dpi(100)
       # t[0][0].set_title('write access list')
       # t[0][1].set_title('access list')
        try:
            for prueba in ['08-05_vi8192_conL1','08-05_nmoesi8192_conL1']:

                archivo = contenedor_de_datos+'/'+prueba+'/'+bench+'-_ipc'
                file = open(contenedor_de_datos+'/'+prueba+'/condor_log/'+bench+'.err')
                simend = buscar_string(file, 'SimEnd')
                file.close()
                if 'ContextsFinished' == simend :
                    tip = leerNombreDatos(archivo)
                    print(tip.names)

                    df = pd.DataFrame(np.loadtxt(archivo,comments='#', skiprows=1, delimiter = ' ', dtype=tip))
                    #df = ajustarResolucionPorTamanyo(df, 10).df

                    dato = pd.DataFrame({prueba : df['write_access_list_count']})
                    dato.plot(ax=t[0])

                    dato = pd.DataFrame({prueba : df['access_list_count']})
                    dato.plot(ax=t[1])

                    dato = pd.DataFrame({prueba : df['nc_write_access_list_count']})
                    dato.plot(ax=t[2])



            t[0].set_ylim(bottom = 0)
            t[0].set_ylabel('write en cola')
            t[0].set_xlabel('Instrucciones ejecutadas')

            t[1].set_ylim(bottom = 0)
            t[1].set_ylabel('accesos en cola')
            t[1].set_xlabel('Instrucciones ejecutadas')

            t[2].set_ylim(bottom = 0)
            t[2].set_ylabel('NC write en cola')
            t[2].set_xlabel('Instrucciones ejecutadas')


            f.suptitle(bench, fontsize=25)
            f.savefig(nombreArchivo+'/'+bench+'acces_list')

        except IOError as e:
            print('Fallo tablas_access_list en el test:')
            print(e)
        except Exception as e:
            print('Fallo tablas_access_list en el test: '+bench)
            print(e)
        t[0].cla()
        t[1].cla()
        plt.close(f)
    return

def mshr():

    start = time.time()

    nombre_resumen = ['4 mshr entries','8 mshr entries','16 mshr entries','mshr disables']

    contenedor_de_datos = '/nfs/gap/fracanma/benchmark/resultados'


    BENCHMARKS =['BlackScholes','DCT','DwtHaar1D','FastWalshTransform','FloydWarshall','MatrixMultiplication','MatrixTranspose','MersenneTwister','QuasiRandomSequence','RecursiveGaussian','Reduction','ScanLargeArrays']


    directorio_salida = contenedor_de_datos+'/'+'graficos mshr'
    DIR_GRAFICOS = directorio_salida

    args = []
    for EXPERIMENTO in nombre_resumen:
        args.append(('_ipc', contenedor_de_datos,EXPERIMENTO))

    #prueba

    pool = multiprocessing.Pool(processes=4)
    resultados_temp = []
    pool_result = []
    for EXPERIMENTO in nombre_resumen:
        args = []


        directorio_salida = contenedor_de_datos+'/'+EXPERIMENTO+'_resumen'
        DIR_GRAFICOS = directorio_salida+'/graficos'

        # crear directorios
        if not os.path.exists(directorio_salida):
            os.mkdir(directorio_salida)
        if not os.path.exists(directorio_salida+'/tablas'):
            os.mkdir(directorio_salida+'/tablas')
        if not os.path.exists(directorio_salida+'/graficos'):
            os.mkdir(directorio_salida+'/graficos')

        for bench in BENCHMARKS:
            args.append(('-_ipc', contenedor_de_datos,EXPERIMENTO,bench))

        pool_result.append(pool.starmap_async(loadDictCompletomultiprocesses2, args))

    for r in pool_result:
        aux = r.get()
        resultados_temp.append( dict(zip(BENCHMARKS,aux)))

    dict_por_instrucciones = dict(zip(nombre_resumen, resultados_temp ))

    print('SE CARGARON TODOS LOS DATOS!!!')
    print('TIEMPO LEYENDO DATOS : ',time.time() - start)



    print('TIEMPO DE EXECUCION : ',time.time() - start)

    return

def no_blocking():
    return

def grayify_cmap(cmap):
    """Return a grayscale version of the colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors_i = np.linspace(0, 1., 100)
    colors=cmap(colors_i)

    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    N=100
    rgb=np.zeros((3,N,3))
    for n in range(3):
        rgb[n,:,0]=np.linspace(0,1,N)
        rgb[n,:,1]=luminance
        rgb[n,:,2]=luminance
    k=['red', 'green', 'blue']
    data=dict(zip(k,rgb))
    my_cmap = mpl.colors.LinearSegmentedColormap("grayify",data)
    return my_cmap

def grafica_ipc_opc(monton):

    #cmap = sb.color_palette("Greys_r", 4)
    #sb.set_palette(cmap, n_colors=4)

    f, t = plt.subplots()
    f.set_size_inches(15,10)
    f.set_dpi(300)
    #opc = pd.DataFrame(index=BENCHMARKS , columns=['execution time','opc','ipc'])
    opc = pd.DataFrame(index=BENCHMARKS , columns=['vector inst', 'scalar inst'])
    opi = pd.DataFrame(index=BENCHMARKS , columns=['mshr_disable'])
    ipc = pd.DataFrame(index=BENCHMARKS , columns=['vector inst', 'scalar inst'])
    exec_time = pd.DataFrame(index=BENCHMARKS , columns=['mshr_disable'])
    df_instructions = pd.DataFrame(index=BENCHMARKS,columns = ['i_scalar','mi_simd','i_branch','mi_lds','i_s_mem','mi_v_mem'])


    speedup= pd.DataFrame(index=['IPC','OPC'])


    for bench in BENCHMARKS :


        try:
            #df_aux = monton[bench][bench]['con L1'].df
            '''opc[exp][bench] = df_aux[['i_scalar','i_simd','i_s_mem','i_v_mem','i_branch','i_lds']].sum(0).sum() / float(df_aux['ciclos_intervalo'].sum())

            ipc[exp][bench] = df_aux[['i_scalar','mi_simd','i_s_mem','mi_v_mem','i_branch','mi_lds']].sum(0).sum() / float(df_aux['ciclos_intervalo'].sum())
            invalidations[exp][bench] = df_aux['invalidations'].sum() / (df_aux['accesos_gpu'].sum() - df_aux['Coalesces_gpu'].sum() - df_aux['Coalesces_L1'].sum())
            retries[exp][bench] = df_aux['accesses_with_retries'].sum() / (df_aux['accesos_gpu'].sum() - df_aux['Coalesces_gpu'].sum() - df_aux['Coalesces_L1'].sum())
            latencia[exp][bench] = df_aux['lat_loads_mem'].sum() / df_aux['num_loads_mem'].sum()


            memEventsLoad = df_aux[['queue_load','lock_mshr_load','lock_dir_load','eviction_load','retry_load','miss_load','finish_load']].sum(0).sum()/ df_aux['access_load'].sum(0).sum()
            memEventsStore = df_aux[['queue_nc_write','lock_mshr_nc_write','lock_dir_nc_write','eviction_nc_write','retry_nc_write','miss_nc_write','finish_nc_write']].sum(0).sum()/ df_aux['access_nc_write'].sum(0).sum()
            gpuEventsLoad = df_aux[['gpu_queue_load','gpu_lock_mshr_load','gpu_lock_dir_load','gpu_eviction_load','gpu_retry_load','gpu_miss_load','gpu_finish_load']].sum(0).sum()/ df_aux['gpu_access_load'].sum(0).sum()

            latencia[exp][bench] = memEventsLoad + memEventsStore
    '''

            baseline = monton['mshr_disable'][bench]['con L1'].df

            #variacion = monton['4 mshr entries'][bench]['con L1'].df

            #opc['execution time'][bench] = variacion['ciclos_intervalo'].sum() / float(baseline['ciclos_intervalo'].sum())
            #opc['opc'][bench] = (variacion[['i_scalar','i_simd','i_s_mem','i_v_mem','i_branch','i_lds']].sum(0).sum() / float(variacion['ciclos_intervalo'].sum())) / (baseline[['i_scalar','i_simd','i_s_mem','i_v_mem','i_branch','i_lds']].sum(0).sum() / baseline['ciclos_intervalo'].sum())
            #opc['ipc'][bench] = (variacion[['i_scalar','mi_simd','i_s_mem','mi_v_mem','i_branch','mi_lds']].sum(0).sum() / float(variacion['ciclos_intervalo'].sum())) / (baseline[['i_scalar','mi_simd','i_s_mem','mi_v_mem','i_branch','mi_lds']].sum(0).sum() / baseline['ciclos_intervalo'].sum())

            #print('variacion opc = '+str(variacion[['i_scalar','i_simd','i_s_mem','i_v_mem','i_branch','i_lds']].sum(0).sum() / float(variacion['ciclos_intervalo'].sum())))
            #print('baseline opc = '+str(baseline[['i_scalar','i_simd','i_s_mem','i_v_mem','i_branch','i_lds']].sum(0).sum() / baseline['ciclos_intervalo'].sum()))

            #print('variacion ipc = '+str(variacion[['i_scalar','mi_simd','i_s_mem','mi_v_mem','i_branch','mi_lds']].sum(0).sum() / float(variacion['ciclos_intervalo'].sum())))
            #print('baseline ipc = '+str(baseline[['i_scalar','mi_simd','i_s_mem','mi_v_mem','i_branch','mi_lds']].sum(0).sum() / baseline['ciclos_intervalo'].sum()))

            #exec_time['mshr_disable'][bench] = baseline['ciclos_intervalo'].sum()
            #exec_time['4 mshr entries'][bench] = variacion['ciclos_intervalo'].sum()

            opc['vector inst'][bench] = baseline[['i_simd','i_v_mem','i_lds']].sum(0).sum() / float(baseline['ciclos_intervalo'].sum())
            opc['scalar inst'][bench] = baseline[['i_scalar','i_s_mem','i_branch']].sum(0).sum() / float(baseline['ciclos_intervalo'].sum())

            ipc['vector inst'][bench] = baseline[['mi_simd','mi_v_mem','mi_lds']].sum(0).sum() / float(baseline['ciclos_intervalo'].sum())
            ipc['scalar inst'][bench] = baseline[['i_scalar','i_s_mem','i_branch']].sum(0).sum() / float(baseline['ciclos_intervalo'].sum())

            #ipc['mshr_disable'][bench] = baseline[['i_scalar','mi_simd','i_s_mem','mi_v_mem','i_branch','mi_lds']].sum(0).sum() / float(baseline['ciclos_intervalo'].sum())

            opi['mshr_disable'][bench] = baseline[['i_simd','i_v_mem','i_lds']].sum(0).sum() / baseline[['mi_simd','mi_v_mem','mi_lds']].sum(0).sum()

            df_sum = pd.DataFrame(baseline.sum(),columns=[bench]).transpose()
            df_instructions.ix[bench] = df_sum.ix[bench,['i_scalar', 'mi_simd', 'i_s_mem', 'mi_v_mem', 'i_branch', 'mi_lds',]]



        except Exception as e:
            print('Fallo grafica_ipc_opc(): '+bench)
            print(e)

    #opc.plot(ax=t, kind='bar',rot=25, title='mshr performance')
    df_instructions = df_instructions.div(df_instructions.sum(1), axis=0)
    df_instructions.plot(ax=t,rot=90, kind='bar',stacked=True)
    t.legend(loc='upper left', bbox_to_anchor=(1, 1))
    t.set_xticklabels(ticks)
    t.set_ylabel('Instructions type distribution')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + 'porcentage_de_inst.eps',format='eps', bbox_inches='tight')
    t.cla()



    opc.plot(ax=t, kind='bar',stacked=True, rot=90)
    t.set_xticklabels(ticks)
    t.set_ylabel('OPC')
    #t.set_xticks([])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + 'opc.eps',format='eps', bbox_inches='tight')
    t.cla()
    #t[0].set_xlabel('OPC')
    opi.plot(ax=t, kind='bar', rot=90,legend=False)
    t.set_ylabel('OPI')
    t.set_xticklabels(ticks)
    #t.set_xticks([])
    f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + 'opi.eps',format='eps', bbox_inches='tight')
    t.cla()
    #t[0].set_xlabel('OPI')
    ipc.plot(ax=t, kind='bar',stacked=True , rot=90)
    t.set_ylabel('IPC')
    t.set_xticklabels(ticks)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + 'ipc.eps',format='eps', bbox_inches='tight')
    #t[0].set_xlabel('IPC')
    t.cla()
    plt.close(f)

    graficas_coalesce(monton)



def graficas_no_blocking_store(monton):

    f, t = plt.subplots()
    f.set_size_inches(15, 10)
    f.set_dpi(300)

    vmb_blocking_time = pd.DataFrame(index=BENCHMARKS , columns=['NBS disable','NBS enable']) #,'store NBS disable'])
    opc = pd.DataFrame(index=BENCHMARKS , columns=['NBS disable','NBS enable'])
    #vmb_inst_rate = pd.DataFrame(index=BENCHMARKS , columns=['blocking_access_NBS_disable','blocking_store_NBS_disable'])
    for bench in BENCHMARKS :
        try:
            #vmb
            coalesce_gpu_enable = monton['no_blocking_store_enable'][bench]['con L1'].df
            coalesce_gpu_disable = monton['coalesce_gpu_disable'][bench]['con L1'].df

            CU_used = 10
            blocking_store_enable = 0
            blocking_store_disable = 0
            blocking_enable = 0
            blocking_disable = 0
            for i in [0,1,2,3,4,5,6,7,8,9] :

                #print('CU '+str(i)+' = '+str(coalesce_gpu_enable['vmb_blocked_store_CU'+str(i)].sum()))

                #if coalesce_gpu_disable['vmb_blocked_store_CU'+str(i)].sum() == 0 :
                #    CU_used = i
                #    break

                blocking_store_enable +=  coalesce_gpu_enable['vmb_blocked_store_CU'+str(i)].sum()
                blocking_enable +=  coalesce_gpu_enable['vmb_blocked_store_CU'+str(i)].sum() + coalesce_gpu_enable['vmb_blocked_load_CU'+str(i)].sum()
                blocking_disable +=  coalesce_gpu_disable['vmb_blocked_store_CU'+str(i)].sum() + coalesce_gpu_disable['vmb_blocked_load_CU'+str(i)].sum()
                # + coalesce_gpu_enable['vmb_blocked_load_CU'+str(i)].sum()
                blocking_store_disable +=  coalesce_gpu_disable['vmb_blocked_store_CU'+str(i)].sum()
                # + coalesce_gpu_disable['vmb_blocked_load_CU'+str(i)].sum()


            #vmb_blocking_time['no_blocking_store_enable'][bench] = ((blocking_store_enable / CU_used) / coalesce_gpu_enable['ciclos_intervalo'].sum()) * 100

            #vmb_blocking_time['store NBS disable'][bench] = ((blocking_store_disable / CU_used) / coalesce_gpu_disable['ciclos_intervalo'].sum()) * 100
            #vmb_blocking_time['store NBS disable'][bench] = ((blocking_store_disable / 10) / coalesce_gpu_disable['ciclos_intervalo'].sum()) * 100

            vmb_blocking_time['NBS enable'][bench] = ((blocking_enable / 10) / coalesce_gpu_disable['ciclos_intervalo'].sum()) * 100

            vmb_blocking_time['NBS disable'][bench] = ((blocking_disable / 10) / coalesce_gpu_disable['ciclos_intervalo'].sum()) * 100

            opc['NBS disable'][bench] = coalesce_gpu_disable[['i_simd','i_v_mem','i_lds','i_scalar','i_s_mem','i_branch']].sum(0).sum() / float(coalesce_gpu_disable['ciclos_intervalo'].sum())
            opc['NBS enable'][bench] = coalesce_gpu_enable[['i_simd','i_v_mem','i_lds','i_scalar','i_s_mem','i_branch']].sum(0).sum() / float(coalesce_gpu_enable['ciclos_intervalo'].sum())
            #vmb_inst_rate.ix[bench]['no_blocking_store_enable'] = coalesce_gpu_enable['vmb_inst_counter']
            #vmb_inst_rate.ix[bench]['no_blocking_store_disable'] = coalesce_gpu_disable['vmb_inst_counter']
            (coalesce_gpu_enable['vmb_inst_counter'] / coalesce_gpu_enable['ciclos_intervalo']).plot(ax=t, legend=True, rot=45, label='no blocking store enable', title='Instruction rate at VMB ('+ bench +')')
            (coalesce_gpu_disable['vmb_inst_counter'] / coalesce_gpu_enable['ciclos_intervalo']).plot(ax=t, legend=True, rot=45, label='no blocking store disable',title='Instruction rate at VMB ('+ bench +')')
            t.set_xticks([])
            t.set_xlabel('Executed instructions')
            t.set_ylabel('instructions / cycles')
            f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + bench + 'inst_rate.eps', format='eps', bbox_inches='tight')
            t.cla()


        except Exception as e:
            print('Fallo graficas_no_blocking_store(): '+bench)
            print(e)

    #vmb_blocking_time.columns = ['no blocking store enable','no blocking store disable']
    vmb_blocking_time.plot(ax=t, kind='bar', rot=75)
    t.set_ylabel('% Time VMB blocked')
    t.set_xticklabels(ticks)
    t.set_yticklabels(['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%'])
    f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + 'vmb.eps',format='eps', bbox_inches='tight')
    t.cla()

    opc.plot(ax=t, kind='bar', rot=75)
    t.set_ylabel('OPC')
    t.set_xticklabels(ticks)
    t.legend(loc='upper center', bbox_to_anchor=(0.4, 1))
    f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + 'vmb_opc.eps',format='eps', bbox_inches='tight')
    t.cla()

    (opc['NBS enable']/opc['NBS disable']).plot(ax=t, kind='bar', rot=75, legend=False)
    t.set_ylabel('Speedup')
    t.set_xticklabels(ticks)
    t.set_ylim(bottom = 1)
    f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + 'vmb_opc_speedup.eps',format='eps', bbox_inches='tight')
    t.cla()

    vmb_blocking_time.ix[['FastWalshTransform','MatrixMultiplication','MatrixTranspose','MersenneTwister']].plot(ax=t, kind='bar',rot=45, title='Cycles VMB is blocked')
    t.set_ylabel('Average time VMB was blocked by store(% over total cycles)')
    f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + 'vmb_altos.eps',format='eps', bbox_inches='tight')
    t.cla()

    vmb_blocking_time.ix[['BlackScholes','DCT','DwtHaar1D','FloydWarshall','QuasiRandomSequence','RecursiveGaussian','Reduction','ScanLargeArrays']].plot(ax=t, kind='bar',rot=65, title='Cycles VMB is blocked')
    t.set_ylabel('Average time VMB was blocked by store(% over total cycles)')
    f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + 'vmb_bajos.eps',format='eps', bbox_inches='tight')
    t.cla()

    t.cla()
    plt.close(f)

def graficas_coalesce(monton):

    f, t = plt.subplots()
    f.set_size_inches(15, 10)
    f.set_dpi(300)

    latency = pd.DataFrame(index=nombre_resumen, columns=['memory system latency','GPU latency'])
    opc = pd.DataFrame(index=benchmarks_amd, columns=nombre_resumen)
    coalesce = pd.DataFrame(index=benchmarks_amd, columns=['coalesce'])
    opc_coalesce = pd.DataFrame(index=benchmarks_amd, columns=['speedup'])


    for bench in BENCHMARKS :
        try:
            for exp in nombre_resumen :

                df_aux = monton[exp][bench]['con L1'].df
                latency.ix[exp]['memory system latency'] = df_aux['lat_loads_mem'].sum(0)/ df_aux['num_loads_mem'].sum(0)
                latency.ix[exp]['GPU latency'] = df_aux['lat_loads_gpu'].sum(0)/ df_aux['num_loads_gpu'].sum(0)

                '''
                opc_coalesce_enable = monton['coalesce_gpu_enable'][bench]['con L1'].df[['i_scalar','i_simd','i_s_mem','i_v_mem','i_branch','i_lds']].sum(0).sum() / float(monton['coalesce_gpu_enable'][bench]['con L1'].df['ciclos_intervalo'].sum())
                opc_coalesce_disable = monton['coalesce_gpu_disable'][bench]['con L1'].df[['i_scalar','i_simd','i_s_mem','i_v_mem','i_branch','i_lds']].sum(0).sum() / float(monton['coalesce_gpu_disable'][bench]['con L1'].df['ciclos_intervalo'].sum())
                opc_coalesce.ix[bench]['speedup'] =  opc_coalesce_enable / opc_coalesce_disable
                '''


                opc[exp][bench] = df_aux[['i_scalar','i_simd','i_s_mem','i_v_mem','i_branch','i_lds']].sum(0).sum() / float(df_aux['ciclos_intervalo'].sum())


            coalesce.ix[bench]['coalesce'] = (monton['coalesce_gpu_enable'][bench]['con L1'].df['Coalesces_gpu'].sum(0) / monton['coalesce_gpu_disable'][bench]['con L1'].df['Coalesces_L1'].sum(0))-1
            opc_coalesce_enable = monton['coalesce_gpu_enable'][bench]['con L1'].df[['i_scalar','i_simd','i_s_mem','i_v_mem','i_branch','i_lds']].sum(0).sum() / float(monton['coalesce_gpu_enable'][bench]['con L1'].df['ciclos_intervalo'].sum())
            opc_coalesce_disable = monton['coalesce_gpu_disable'][bench]['con L1'].df[['i_scalar','i_simd','i_s_mem','i_v_mem','i_branch','i_lds']].sum(0).sum() / float(monton['coalesce_gpu_disable'][bench]['con L1'].df['ciclos_intervalo'].sum())
            opc_coalesce.ix[bench]['speedup'] =  ((opc_coalesce_enable / opc_coalesce_disable) - 1)*100

        except Exception as e:
            print('Fallo graficas_coalesce(): '+bench)
            print(e)


        latency.plot(ax=t, kind='bar',rot=25)
        t.set_ylabel('Latency (cycles)')
        t.set_xticklabels(['4-MSHR','8-MSHR','16-MSHR','NO-MSHR'])
        f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' +bench+ ' latency.eps',format='eps', bbox_inches='tight')
        t.cla()

    '''
    opc.columns = ['4-MSHR','8-MSHR','16-MSHR','NO-MSHR']
    t.set_ylabel("OPC")
    opc.plot(ax=t, kind='bar',)
    t.set_xticklabels(ticks)
    f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + 'opc_mshr.eps',format='eps', bbox_inches='tight')
    t.cla()

    t.set_ylabel("OPC")

    opc.ix[['BlackScholes','DCT','MersenneTwister','QuasiRandomSequence']].plot(ax=t, kind='bar',rot=25)
    t.legend(loc='upper center', bbox_to_anchor=(0.4, 1))
    t.set_xticklabels(['BlackS','DCT','Mersenne','QuasiRand'])
    f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + 'opc_mshr_altos.eps',format='eps', bbox_inches='tight')
    t.cla()


    t.set_ylabel("OPC")
    opc.ix[['DwtHaar1D','FastWalshTransform','FloydWarshall','MatrixTranspose','RecursiveGaussian','Reduction','ScanLargeArrays']].plot(ax=t, kind='bar',rot=45)
    t.legend(loc='upper center', bbox_to_anchor=(0.6, 1))
    t.set_xticklabels(['Dwt','FastWalsh','Floyd','MatrixT','Gaussian','Reduction','Scan'])
    f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + 'opc_mshr_bajos.eps',format='eps', bbox_inches='tight')
    t.cla()
    '''

    coalesce.plot(ax=t, kind='bar',rot=75, legend=False)
    t.set_ylabel("Relative # coalesced accesses")
    t.set_xticklabels(ticks)
    t.set_yticklabels(['-10%','0%','10%','20%','30%','40%','50%','60%','70%','80%'])
    f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + 'coalesces.eps',format='eps', bbox_inches='tight')
    t.cla()

    opc_coalesce.plot(ax=t, kind='bar',rot=75, legend=False)
    t.set_ylabel("Relative OPC")
    t.set_xticklabels(ticks)
    t.set_yticklabels(['-50%','0%','50%','100%','150%','200%','250%'])
    f.savefig('/nfs/gap/fracanma/benchmark/resultados/graficos/' + 'opc_coalesces.eps',format='eps', bbox_inches='tight')
    t.cla()



    plt.close(f)


    return


if __name__ == '__main__':
    # Create the queue and thread pool.

    start = time.time()


    '''font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

    mpl.rc('font', **font)'''

    '''sb.set(font_scale=3)
    sb.set_style("whitegrid")
    cmap = sb.color_palette("Greys_r", 2)
    sb.set_palette(cmap, n_colors=2)
    mpl.rcParams.update({'font.size': 16})'''



    '''
    tipos_instrucciones = np.dtype([('mshr L1',np.int64),('mshr L2',np.int64),('entradas bloqueadas L1',np.int64),('entradas bloqueadas L2',np.int64),('Coalesces L1',np.int64),('Coalesces L2',np.int64),('accesos L1',np.int64),('accesos L2',np.int64),('efectivos L1',np.int64),('efectivos L2',np.int64),('misses L1',np.int64),('misses L2',np.int64),('hits L1',np.int64),('hits L2',np.int64),('Cmisses L1',np.int64),('Cmisses L2',np.int64),('Chits L1',np.int64),('Chits L2',np.int64),('lat L1-L2',np.int64),('paquetes L1-L2',np.int64),('lat L2-MM',np.int64),('paquetes L2-MM',np.int64),('lat loads gpu',np.int64),('num loads gpu',np.int64),('lat loads mem',np.int64),('num loads mem',np.int64),('i_scalar',np.int64),('i_simd',np.int64),('mi_simd',np.int64),('i_s_mem',np.int64),('i_v_mem',np.int64),('mi_v_mem',np.int64),('i_branch',np.int64),('i_lds',np.int64),('mi_lds',np.int64),('total intervalo',np.int64),('total global',np.int64),('ciclos intervalo',np.int64),('ciclos totales',np.int64)])

    tipos_ciclos = np.dtype([('entradas bloqueadas L1(borrar)',np.int64),('entradas bloqueadas L2(borrar)',np.int64),('lat loads',np.int64),('num loads',np.int64),('Coalesces L1',np.int64),('accesos L1',np.int64),('hits L1',np.int64),('invalidations L1',np.int64),('Coalesces L2',np.int64),('accesos L2',np.int64),('hits L2',np.int64),('invalidations L2',np.int64),('busy in L1-L2',np.int64),('busy out L1-L2',np.int64),('busy in L2-MM',np.int64),('busy out L2-MM',np.int64),('lat L1-L2',np.int64),('paquetes L1-L2',np.int64),('lat L2-MM',np.int64),('paquetes L2-MM',np.int64),('blk compartidos',np.int64),('blk replicas',np.int64),('entradas bloqueadas L1',np.int64),('entradas bloqueadas L2',np.int64),('ciclos intervalo',np.int64),('ciclos totales',np.int64)])
    '''

    # tipos_instrucciones = np.dtype([('access_list_count',np.int64),('mshr L1',np.int64),('mshr L2',np.int64),('entradas bloqueadas L1',np.int64),('entradas bloqueadas L2',np.int64),('Coalesces gpu',np.int64),('Coalesces L1',np.int64),('Coalesces L2',np.int64),('accesos gpu',np.int64),('accesos L1',np.int64),('accesos L2',np.int64),('efectivos L1',np.int64),('efectivos L2',np.int64),('misses L1',np.int64),('misses L2',np.int64),('hits L1',np.int64),('hits L2',np.int64),('Cmisses L1',np.int64),('Cmisses L2',np.int64),('Chits L1',np.int64),('Chits L2',np.int64),('lat L1-L2',np.int64),('paquetes L1-L2',np.int64),('lat L2-MM',np.int64),('paquetes L2-MM',np.int64),('lat loads gpu',np.int64),('num loads gpu',np.int64),('lat loads mem',np.int64),('num loads mem',np.int64),('i_scalar',np.int64),('i_simd',np.int64),('mi_simd',np.int64),('i_s_mem',np.int64),('i_v_mem',np.int64),('mi_v_mem',np.int64),('i_branch',np.int64),('i_lds',np.int64),('mi_lds',np.int64),('total intervalo',np.int64),('total global',np.int64),('ciclos intervalo',np.int64),('ciclos totales',np.int64)])


    #nombre_resumen = ['04-16_mshr32_solo_no_blocking_store_10CU','no_blocking_write_mshr_disabled']
    #nombre_resumen = ['04-16_mshr32_solo_coalesce_gpu_10CU','04-16_mshr32_coalesce_gpu_mixto_10CU','coalesce_mshr_disabled']
    nombre_resumen = ['07-03_nmoesi_mshr16_test_statistics2','07-03_nmoesi_mshr32_test_statistics2','07-03_nmoesi_mshr256_test_statistics2']
    nombre_resumen = ['07-29_nmoesi_mshr16_test','07-29_nmoesi_mshr32_test','07-29_nmoesi_mshr256_test']
    #,'06-22_nmoesi_mshr16_mshr_estatico_recursosextra','06-22_nmoesi_mshr32_mshr_estatico_recursosextra','06-22_nmoesi_mshr256_mshr_estatico_recursosextra']

    # variables de los test

    #coalesce
    #nombre_resumen = ['coalesce_gpu_enable','coalesce_gpu_disable']

    #no_blocking
    #nombre_resumen = ['no_blocking_store_enable','coalesce_gpu_disable']

    #mshr
    #nombre_resumen = ['4 mshr entries','8 mshr entries','16 mshr entries','mshr_disable']



    #nombre_resumen = ['04-15_mshr4_cc','04-15_mshr8_cc','04-15_mshr16_cc','04-15_mshr32_cc','04-15_mshr4_no_blocking_store_enable','04-15_mshr8_no_blocking_store_enable','04-15_mshr16_no_blocking_store_enable','04-15_mshr32_no_blocking_store_enable','04-15_mshr4_coalesce_gpu_enable','04-15_mshr8_coalesce_gpu_enable','04-15_mshr16_coalesce_gpu_enable','04-15_mshr32_coalesce_gpu_enable']


    # no_blocking_write
    #nombre_resumen = ['04-15_mshr4_no_blocking_store_enable_5CU','04-15_mshr8_no_blocking_store_enable_5CU','04-15_mshr16_no_blocking_store_enable_5CU','04-15_mshr32_no_blocking_store_enable_5CU','04-15_mshr4_coalesce_gpu_enable_5CU','04-15_mshr8_coalesce_gpu_enable_5CU','04-15_mshr16_coalesce_gpu_enable_5CU','04-15_mshr32_coalesce_gpu_enable_5CU']

    # coalesce_gpu
    #nombre_resumen = ['04-15_mshr4_coalesce_gpu_enable_5CU','04-15_mshr8_coalesce_gpu_enable_5CU','04-15_mshr16_coalesce_gpu_enable_5CU','04-15_mshr32_coalesce_gpu_enable_5CU','04-15_mshr4_mshr_enable_5CU','04-15_mshr8_mshr_enable_5CU','04-15_mshr16_mshr_enable_5CU','04-15_mshr32_mshr_enable_5CU']

    # mshr
    #nombre_resumen = ['04-15_mshr4_mshr_enable_5CU','04-15_mshr8_mshr_enable_5CU','04-15_mshr16_mshr_enable_5CU','04-15_mshr32_mshr_enable_5CU','04-15_mshr16_mshr_disable_5CU']

    # no_blocking_write
    #nombre_resumen = ['04-15_mshr4_no_blocking_store_enable','04-15_mshr8_no_blocking_store_enable','04-15_mshr16_no_blocking_store_enable','04-15_mshr32_no_blocking_store_enable','04-15_mshr4_coalesce_gpu_enable','04-15_mshr8_coalesce_gpu_enable','04-15_mshr16_coalesce_gpu_enable','04-15_mshr32_coalesce_gpu_enable']

    # coalesce_gpu
    #nombre_resumen = ['04-15_mshr4_coalesce_gpu_enable','04-15_mshr8_coalesce_gpu_enable','04-15_mshr16_coalesce_gpu_enable','04-15_mshr32_coalesce_gpu_enable','04-15_mshr4_mshr_enable','04-15_mshr8_mshr_enable','04-15_mshr16_mshr_enable','04-15_mshr32_mshr_enable']



    #nombre_resumen = ['04-13_mshr4_con_mshr','04-13_mshr8_con_mshr','04-13_mshr16_con_mshr','04-13_mshr32_con_mshr','04-13_mshr4_sin_mshr','04-13_mshr8_sin_mshr','04-13_mshr16_sin_mshr','04-13_mshr32_sin_mshr']

    #nombre_resumen = ['04-07_mshr4_witness','04-07_mshr8_witness','04-07_mshr16_witness','04-07_mshr32_witness','03-27_mshr64_m2s_sin_mod']
    #nombre_resumen = ['03-27_mshr4_m2s_sin_mod','03-27_mshr8_m2s_sin_mod','03-27_mshr16_m2s_sin_mod','03-27_mshr32_m2s_sin_mod','03-27_mshr64_m2s_sin_mod','03-27_mshr128_m2s_sin_mod','03-27_mshr256_m2s_sin_mod','04-01_mshr4_m2s_mshr2','04-01_mshr8_m2s_mshr2','04-01_mshr16_m2s_mshr2','04-01_mshr32_m2s_mshr2','04-01_mshr64_m2s_mshr2','04-01_mshr128_m2s_mshr2','04-01_mshr256_m2s_mshr2','04-01_mshr4_m2s_witness','04-01_mshr8_m2s_witness','04-01_mshr16_m2s_witness','04-01_mshr32_m2s_witness']

    #nombre_resumen = ['02-23_mshr4','02-23_mshr8','02-23_mshr16','02-23_mshr32','02-23_mshr64','02-23_mshr128','02-23_mshr256','03-23_mshr4_base1_cc','03-23_mshr8_base1_cc','03-23_mshr16_base1_cc','03-23_mshr32_base1_cc','03-23_mshr64_base1_cc','03-23_mshr128_base1_cc','03-23_mshr256_base1_cc','03-24_mshr4_cc_test','03-24_mshr8_cc_test','03-24_mshr16_cc_test','03-24_mshr32_cc_test','03-24_mshr64_cc_test','03-24_mshr128_cc_test','03-24_mshr256_cc_test','03-27_mshr4_m2s_sin_mod','03-27_mshr8_m2s_sin_mod','03-27_mshr16_m2s_sin_mod','03-27_mshr32_m2s_sin_mod','03-27_mshr64_m2s_sin_mod','03-27_mshr128_m2s_sin_mod','03-27_mshr256_m2s_sin_mod']

    #nombre_resumen = ['02-23_mshr4','02-23_mshr8','02-23_mshr16','02-23_mshr32','02-23_mshr64','02-23_mshr128','02-23_mshr256','03-24_mshr4_cc_test','03-24_mshr8_cc_test','03-24_mshr16_cc_test','03-24_mshr32_cc_test','03-24_mshr64_cc_test','03-24_mshr128_cc_test','03-24_mshr256_cc_test','02-23_mshr4_CBy_efectivos','02-23_mshr8_CBy_efectivos','02-23_mshr16_CBy_efectivos','02-23_mshr32_CBy_efectivos','02-23_mshr64_CBy_efectivos','02-23_mshr128_CBy_efectivos','02-23_mshr256_CBy_efectivos']

    #nombre_resumen = ['02-10_mshr32','02-19_prueba_dir_32','02-19_prueba_dir_32_2']

    #nombre_resumen = ['02-06_mshr16','02-06_mshr32','02-06_mshr64','02-06_mshr128','02-06_mshr256']

    #nombre_resumen = ['01-28_mshr16_128L2','01-28_mshr32_128L2' ,'01-28_mshr64_128L2','01-28_mshr128_128L2','01-28_mshr256_128L2','01-28_mshr16_512L2','01-28_mshr32_512L2' ,'01-28_mshr64_512L2','01-28_mshr128_512L2','01-28_mshr256_512L2']


    #nombre_resumen = ['01-22_mshr_16','01-22_mshr_32','01-22_mshr_64','01-22_mshr_128','01-22_mshr_256','01-23_mshr_1024']

    #nombre_resumen = ['01-12_prueba_scalar_256','01-12_prueba_local_memory','01-12_mshr_32','01-06_test32']
    #nombre_resumen = ['01-12_control_mshr_basico','01-12_control_mshr_reset','01-12_control_mshr_intevalor_dinamico','01-12_control_mshr_intevalor_dinamico_reset','01-12_prueba_scalar_256','01-12_prueba_local_memory','01-12_mshr_32','01-06_test32']

    #nombre_resumen = ['12-04_5CU_mshr_16','12-04_5CU_mshr_32','12-04_5CU_mshr_256','12-03_pruebamshronmiss','12-04_5CU_mshr_holdonmiss','12-07_5CU_mshr_32_simd_idle']
    #,'11-25_mshr1_latdim_reset','11-25_mshr1_reset','11-25_mshr1_latdim']
    #nombre_resumen = ['11-17_mshr-colas-16','11-17_mshr-colas-32','11-17_mshr-colas-256','11-18_mshr1-latdim-256','11-18_mshr1-latdim-16','11-18_mshr1-latdim-16']
    #nombre_resumen = ['10-06_mshr16-32VIEJO','10-06_mshr32-64VIEJO','10-06_mshr1024-8192VIEJO','10-22_mshr_opc_500000','10-22_mshr_opc2','10-22_mshr_opc2_500000']
    #nombre_resumen = ['10-06_mshr16-32VIEJO','10-06_mshr32-64VIEJO','10-06_mshr1024-8192VIEJO','11-09_mshr1-32-64_latdim','11-09_mshr1-32-64_reset','11-09_mshr1-32-64_reset-latdim']
    #nombre_resumen = ['10-06_mshr16-32VIEJO','10-06_mshr32-64VIEJO','10-06_mshr1024-8192VIEJO','11-06_mshr1-32-64_latDim','11-06_mshr1-32-64_reset','11-07_mshr1-32-64_reset-latdim']

    #nombre_resumen =['10-01_mshrL2-8','10-01_mshrL2-32','10-01_mshrL2-128','10-01_mshrL2-512','10-01_mshrL2-2048']
    #nombre_resumen =['09-26_test']
    contenedor_de_datos = '/nfs/gap/fracanma/benchmark/resultados'

    dict_por_instrucciones = {}


    #ticks = ['BlackS','DCT','Dwt','FastWalsh','Floyd','MatrixT','Mersenne','QuasiRand','Gaussian','Reduction','Scan']
    #benchmarks_amd =['BlackScholes','DCT','DwtHaar1D','FastWalshTransform','FloydWarshall','MatrixTranspose','MersenneTwister','QuasiRandomSequence','RecursiveGaussian','Reduction','ScanLargeArrays']

    benchmarks_amd = ['BinarySearch','BinomialOption','BlackScholes','DCT','DwtHaar1D','FastWalshTransform','FloydWarshall','MatrixMultiplication','MatrixTranspose','MersenneTwister','QuasiRandomSequence','RadixSort','RecursiveGaussian','Reduction','ScanLargeArrays','SimpleConvolution']

    #benchmarks_amd = ['BinomialOption','DwtHaar1D','FastWalshTransform','FloydWarshall','MatrixMultiplication','MatrixTranspose']
    #benchmarks_amd = ['BlackScholes','FastWalshTransform','FloydWarshall','MatrixMultiplication','MersenneTwister']
    benchmarks_rodinia = ['backprop','bfs','b+tree','gaussian','kmeans','lud','streamcluster']

    BENCHMARKS = benchmarks_amd
    ticks = BENCHMARKS
    directorio_salida = contenedor_de_datos+'/'+nombre_resumen[0]+'_resumen'
    DIR_GRAFICOS = directorio_salida+'/graficos'

    pool = multiprocessing.Pool(processes=4)
    resultados_temp = []
    pool_result = []
    for EXPERIMENTO in nombre_resumen:
        args = []


        directorio_salida = contenedor_de_datos+'/'+EXPERIMENTO+'_resumen'
        DIR_GRAFICOS = directorio_salida+'/graficos'

        # crear directorios
        if not os.path.exists(directorio_salida):
            os.mkdir(directorio_salida)
        if not os.path.exists(directorio_salida+'/tablas'):
            os.mkdir(directorio_salida+'/tablas')
        if not os.path.exists(directorio_salida+'/graficos'):
            os.mkdir(directorio_salida+'/graficos')

        for bench in BENCHMARKS:
            args.append(('_ipc', contenedor_de_datos,EXPERIMENTO,bench))

        pool_result.append(pool.starmap_async(loadDictCompletomultiprocesses2, args))

    for r in pool_result:
        aux = r.get()
        resultados_temp.append( dict(zip(BENCHMARKS,aux)))


        # leer nombre de las columnas


    #dict_por_instrucciones[EXPERIMENTO] = loadDictCompleto('-_ipc',directorio_salida, TESTS)
    #dict_por_ciclos = loadDictCompleto('-fran_general')
    dict_por_ciclos = 1
    #loadDatosSueltos(dict_por_ciclos, dict_por_instrucciones[EXPERIMENTO])

    #pool = MyPool()
    #resultados_temp = pool.starmap(loadDictCompletomultiprocesses, args)

    #pool = multiprocessing.Pool()
    #resultados_temp = pool.starmap(loadDictCompleto, args)


    print('SE CARGARON TODOS LOS DATOS!!!')
    print('TIEMPO LEYENDO DATOS : ',time.time() - start)
    #pool.close()
    #pool.join()
    dict_por_instrucciones = dict(zip(nombre_resumen, resultados_temp ))


        #IPC(DIR_GRAFICOS+'/por_instrucciones/',dict_por_instrucciones)
        #dibujar4tablas(DIR_GRAFICOS+'/por_instrucciones/',dict_por_instrucciones[exp],exp)

        #dibujar4tablas2(DIR_GRAFICOS+'/por_instrucciones/',dict_por_instrucciones)
        #tablas_access_list(DIR_GRAFICOS+'/por_instrucciones/')
        #comparar_velocidad('/nfs/gap/fracanma/benchmark/resultados/prueba_resumen/grafico')fracanma/benchmark/resultados/prueba_resumen/grafico')
        #gc.collect()
    #grafica_ipc_opc(dict_por_instrucciones)
    #graficas_no_blocking_store(dict_por_instrucciones)
    #graficas_coalesce(dict_por_instrucciones)
    IPCmultitest(DIR_GRAFICOS+'/por_instrucciones/', dict_por_instrucciones)
    barras_opc(DIR_GRAFICOS, dict_por_instrucciones)
    analisis_stall(DIR_GRAFICOS, dict_por_instrucciones)
    benchXexp(DIR_GRAFICOS, dict_por_instrucciones)
    '''
    pool.starmap_async(IPCmultitest,[(DIR_GRAFICOS+'/por_instrucciones/', dict_por_instrucciones)])
    pool.starmap_async(barras_opc,[(DIR_GRAFICOS, dict_por_instrucciones)])
    pool.starmap_async(analisis_stall,[(DIR_GRAFICOS, dict_por_instrucciones)])
    pool.starmap_async(benchXexp,[(DIR_GRAFICOS, dict_por_instrucciones)])
    '''
    pool.close()
    pool.join()
    print('TIEMPO DE EXECUCION : ',time.time() - start)
