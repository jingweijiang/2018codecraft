import random
#random.seed(0)
def single_server_summation(single_server, flavor_specification):
    summation = [0] * len(flavor_specification[1])
    for v in single_server:
        summation[0] += flavor_specification[v][0]
        summation[1] += flavor_specification[v][1]
    return summation

def go_through_server_limitation(summation, server_limitations):
    result = [0] * len(server_limitations)
    for index, limitation in enumerate(server_limitations):
        for _index, sub_summation in enumerate(summation):
            result[index] *= sub_summation * 100 / limitation[_index]
    return result

def allocate_server(individual, server_limitations, flavor_specifications):
    def v0_a_v1_inplace(v0, v1):
        #v0 as inplace
        for index in range(len(v0)):
            v0[index] = v0[index] + v1[index]
            
    def v0_m_v1_inplace(v0, v1):
#        print("v0:", v0)
#        print("v1:", v1)
        for index in range(len(v0)):
            v0[index] = v0[index] - v1[index]

            
    def sum_check_result(results):
        return sum(results)
    
    def check_limitation(checked_limitation, target_limitation):
        for index in range(len(checked_limitation)):
            if checked_limitation[index] > target_limitation[index]:
#                print("which dimension boom:", index)
                return True
        return False
    
    def check_limitations(checked_limitations, target_limitations):
        check_results = []
        for index, checked_limitation in enumerate(checked_limitations):
#            print("checked_limitations:", checked_limitation)
#            print("target_limitations:", target_limitations[index])
            check_results.append(check_limitation(checked_limitation, target_limitations[index]))
        return check_results    
    
    def add_flavor_limitation_and_end_point(checked_results, limitations, flavor_limitation, end_points, end_point):
#        print("limitations:", limitations)
#        print("end_points:", end_points)
        for index, result in enumerate(checked_results):
            if (result == False):
                v0_a_v1_inplace(limitations[index], flavor_limitation)
                end_points[index] = end_point  
                
    def evaluate_single(summation, server_limitation):
        result = 0
        for index in range(len(summation)):
            result += (summation[index] * 100) / server_limitation[index]
#            print("result:", result)
        return result
    
    def evaluate_all(summations, server_limitations, end_points, flavor_specifications, individual):
#        print("summations:", summations)
#        print("server_limitations:", server_limitations)
#        print("end_points:", end_points)
#        print("flavor_specifications:", flavor_specifications)
        results = []
        for index in range(len(summations)):
            v0_m_v1_inplace(summations[index], flavor_specifications[individual[end_points[index]]])
            results.append(evaluate_single(summations[index], server_limitations[index]))
#        print("after minus:", summations)
#        print("results:", results)
        return results.index(max(results)), end_points[results.index(max(results))]

    def evaluate_last(summations, server_limitations, flavor_specifications, checked_results):
        results = []
        for index, check in enumerate(checked_results):
            if check == True:
                results.append(0)
            else:
                results.append(evaluate_single(summations[index], server_limitations[index]))
        return results.index(max(results))
    
    limitation_temp = [[0] * len(server_limitations[0]) for i in range(len(server_limitations))]
    index_point_temp = [0] * len(server_limitations)
    checked_results = [False] * len(server_limitations)
    allo_result = []
    server_result = []
    start_point = 0
    last_point = len(individual) 
    index = start_point
    while(True):
        
        while(sum_check_result(checked_results) != len(checked_results)):
#            print("########################################################################")
#            print("individual:", individual[index:])
#            print("checked_results:", checked_results)
            flavor_limitation = flavor_specifications[individual[index]]
            add_flavor_limitation_and_end_point(checked_results, limitation_temp, flavor_limitation, index_point_temp, index)
#            print("after add, limitation_temp:", limitation_temp)
#            print("after add:", index_point_temp)
            checked_results = check_limitations(limitation_temp, server_limitations)
#            print("checked_results:", checked_results)
#            print("allo_result:", allo_result)
#            print("server_result:", server_result)
#            input()
            index += 1
#            print("after individual:", individual[index:])
            if(index == last_point):
                break
        if(sum_check_result(checked_results) == len(checked_results)):
#            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            server_choosed, end_point = evaluate_all(limitation_temp, server_limitations, index_point_temp, flavor_specifications, individual)
#            print("server_choosed:", server_choosed)
#            print("end_point:", end_point)
#            input()
            allo_result.append(individual[start_point:end_point])
            server_result.append(server_choosed)
            limitation_temp = [[lim for lim in flavor_specifications[individual[end_point]]] for i in range(len(server_limitations))]
            index_point_temp = [end_point] * len(server_limitations)
            start_point = end_point
            index = end_point + 1
#            print("start_point:", start_point)
#            print("index:", index)
            checked_results = check_limitations(limitation_temp, server_limitations)
        if(index == last_point):
            checked_results = check_limitations(limitation_temp, server_limitations)
            server_choosed = evaluate_last(limitation_temp, server_limitations, flavor_specifications, checked_results)
            allo_result.append(individual[start_point:])
            server_result.append(server_choosed)
            break
    return allo_result, server_result












flavor_specification = {1:[1, 1024], 2:[1, 2048], 3:[1, 4096], 4:[2, 2048], 5:[2, 4096], 6:[2, 8192],
                        7:[4, 4096], 8:[4, 8192], 9:[4, 16384], 10:[8, 8192], 11:[8, 16384],
                        12:[8, 32768], 13:[16, 16384], 14:[16, 32768], 15:[16, 65536], 16:[32, 32768], 17:[32, 65536], 18:[32, 131072]}
server_cfg=[[56,131072], [84, 262144], [112, 196608]]
individual = [random.randint(1, 18) for i in range(20)]
#individual = [3 for i in range(1000 )]
allo_result, server_result = allocate_server(individual, server_cfg, flavor_specification)
print("individual:", individual)
print("#########################################################################")
for index, allo in enumerate(allo_result):
#    print(allo, single_server_summation(allo, flavor_specification), server_cfg[server_result[index]])
    print(single_server_summation(allo, flavor_specification), server_cfg[server_result[index]])