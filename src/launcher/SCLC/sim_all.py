from src.launcher.SCLC import init_sim, sbm_sim

'''
This script is capable of launching all simulations for our DSLW 2022 paper. Nevertheless, launching everything sequentially will take a huge amount of time. Instead we are using an HTCondor cluster to parallelize the simulations
'''

if __name__ == '__main__':
    init_sim.make_result_dirs()
    for i in range(100):
        init_sim.run(i,0)
    for i in range(100):
        init_sim.run(i,3)
    for i in range(100):
        init_sim.run(i,6)
    init_sim.combine_results()
    init_sim.plot()

    sbm_sim.make_result_dirs()
    for pid in range(9900):
        sbm_sim.run(pid)
    sbm_sim.combine_results()
    sbm_sim.plot()
