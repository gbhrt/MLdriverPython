#import cProfile
#cProfile.run('2 + 2')
import actions_for_given_path as act

import cProfile, pstats, io
import time
#pr = cProfile.Profile()
#pr.enable()
## ... do something ...
#2+2
#print("aaa")
#pr.disable()
#s = io.StringIO()
#sortby = 'cumulative'
#ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.dump_stats("test_profile.profile")
##ps.print_stats()
##print(s.getvalue())


def test(Agent,env):
    pr = cProfile.Profile()
    
    env_state = env.reset(seed = 1111)
    state = Agent.get_state(env_state)
    acc,steer = 0.0,0.0
    acc,steer,planningData = Agent.comp_action(state,acc,steer)#act.comp_MB_action(Agent.nets,state,acc,steer,Agent.trainHP)
    acc,steer,planningData = Agent.comp_action(state,acc,steer)#act.comp_MB_action(Agent.nets,state,acc,steer,Agent.trainHP)
    pr.enable()
    acc,steer,planningData = Agent.comp_action(state,acc,steer)#act.comp_MB_action(Agent.nets,state,acc,steer,Agent.trainHP)
    print(acc,steer)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.dump_stats("action_prof.profile")

    #Agent.start_training()
    #time.sleep(5.0)
    #Agent.stop_training()
    #pr.disable()
    #s = io.StringIO()
    #sortby = 'cumulative'
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.dump_stats("action_prof.profile")

    env.close()