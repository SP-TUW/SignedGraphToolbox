from src.launcher.SCLC import init_sim, sbm_sim


if __name__ == '__main__':
    init_sim.make_result_dirs()
    for i in range(2):
        init_sim.run(i,0)
    for i in range(2):
        init_sim.run(i,1)
    for i in range(2):
        init_sim.run(i,2)
    init_sim.combine_results()
    init_sim.plot()

    sbm_sim.make_result_dirs()
    for pid in range(2):
        sbm_sim.run(pid)
    sbm_sim.combine_results()
    sbm_sim.plot()
