import time
import random
import copy
def genetic_alg_boxing(prediction, server_limitation, flavor_specification, population_scale, start_time, duration_time, elite_scale):

    def v0_a_v1_inplace(v0, v1):
        #v0 as inplace
        for index in range(len(v0)):
            v0[index] = v0[index] + v1[index]
    
    def explane_prediction(prediction):
        encode_prediction = []
        for key, value in prediction.items():
            for i in range(value):
                encode_prediction.append(key)#because index0 represent flavor1
        return encode_prediction
    
    def summation_f(prediction, flavor_specification):
        result = [0] * len(flavor_specification[1])
        for pre in prediction:
            v0_a_v1_inplace(result, flavor_specification[pre])
        return result
    
    def initial_population(encode_prediction, population_scale):
        initial_population = []
        for i in range(population_scale):
            random.shuffle(encode_prediction)
            initial_population.append(copy.copy(encode_prediction))    
        return initial_population

    def evaluation_function(summations, servers, server_limitation):
        server_summation = summation_f(servers, server_limitation)
#        print("summations:", summations)
#        print("server_summation:", server_summation)
        result = 0
        for index, summation in enumerate(summations):
            result += 10000 * summation / server_summation[index] 
#        print("result:", result)
        return result
    
    
    def evaluate_individual(population, server_limitation, flavor_specification, summation):

        evaluation = []
        for individual in population:
#            print(allocate_server(individual, server_limitation, flavor_specification))
#            print(evaluation_function(individual, allocate_server(individual, server_limitation, flavor_specification), flavor_specification, server_limitation, optimization_target))
            evaluation.append(
                    (evaluation_function(summation, allocate_server(individual, server_limitation, flavor_specification)[1], server_limitation)
                    , individual))
        return evaluation    
    
    
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

    def Propagate_descendants(evaluation, elite_scale):
        def mutate(individual):
            
            individual = copy.copy(individual)
            individual_size = len(individual)
            mutation_point = random.randint(0, individual_size-1) # minus one because endpoint included
            left_len = mutation_point
            right_len = individual_size - 1 - mutation_point
        #            candidate_len = left_len if left_len <= right_len else right_len
        #    print("mutation_point:", mutation_point)
            direction = "left" if left_len < right_len else "right" if left_len > right_len else "center"
        #    print("direction:", direction)
        #    print("left_len:", left_len)
        #    print("right_len:", right_len)
            if direction == "left":
                
                exchange_len = random.randint(1, left_len+1)
        #        print("exchange_len:", exchange_len)
                left_end_point = mutation_point + 1
                left_start_point = left_end_point-exchange_len
                
                left_slice = slice(left_start_point, left_end_point)
                
                right_start_point = random.randint(mutation_point+1, individual_size-exchange_len)
                right_end_point = right_start_point+exchange_len
                right_slice = slice(right_start_point, right_end_point)
        #                right_slice = slice()
            elif direction == "right":
                exchange_len = random.randint(1, right_len+1)
        #        print("exchange_len:", exchange_len)        
                right_start_point = mutation_point
                right_end_point = mutation_point + exchange_len
                right_slice = slice(right_start_point, right_end_point)
                
                left_end_point = random.randint(exchange_len, mutation_point)
                left_start_point = left_end_point - exchange_len
                left_slice = slice(left_start_point, left_end_point)
            elif direction == "center":
                return individual
            
        #    print("left_slice:", left_slice)
        #    print("right_slice:", right_slice)
            temp = individual[left_slice]
            individual[left_slice] = individual[right_slice]
        #    print("individual:", individual)
            individual[right_slice] = temp
        #    print("individual:", individual)
            return individual
                
        
        origin_scale = len(evaluation)
        del(evaluation[elite_scale:])
#        print("len(evaluation):", len(evaluation))
#        elite_group = evaluation[0:elite_scale]
        mutated_group = []
        for i in range(elite_scale, origin_scale):
            individual = random.sample(evaluation, 1)[0][1]
#            mutated_individual = mutate(individual)
#            elite_group.append(evaluate_individual())
            mutated_group.append(mutate(individual))
#        print("mutated_group:", mutated_group)
#        evaluate_individual(mutated_group, )
        return mutated_group    
     
    def single_server_summation(single_server, flavor_specification):
        summation = [0] * len(flavor_specification[1])
        for v in single_server:
            summation[0] += flavor_specification[v][0]
            summation[1] += flavor_specification[v][1]
        return summation    
    
    
    encode_prediction = explane_prediction(prediction)
    summation = summation_f(encode_prediction, flavor_specification)
    population = initial_population(encode_prediction, population_scale)
    evaluation = evaluate_individual(population, server_limitation, flavor_specification, summation)
    evaluation.sort(reverse = True)
    while(time.clock()-start_time < duration_time):
        descendants = Propagate_descendants(evaluation, elite_scale)
        evaluation.extend(evaluate_individual(descendants, server_limitation, flavor_specification, summation))
        evaluation.sort(reverse = True)
        print("the best individual you get:")
        print(evaluation[0][0]/float(200), len(allocate_server(evaluation[0][1], server_limitation, flavor_specification)[1]))
    print("################## finally result ########################")
    allo_result, server_result = allocate_server(evaluation[0][1], server_limitation, flavor_specification)
    print("one server with virtual machine       summation of virtual machine     server specification")
    for index, allo in enumerate(allo_result):
        print(allo, single_server_summation(allo, flavor_specification), server_limitation[server_result[index]])









def test():
    flavor_specification = {1:[1, 1024], 2:[1, 2048], 3:[1, 4096], 4:[2, 2048], 5:[2, 4096], 6:[2, 8192],
                            7:[4, 4096], 8:[4, 8192], 9:[4, 16384], 10:[8, 8192], 11:[8, 16384],
                            12:[8, 32768], 13:[16, 16384], 14:[16, 32768], 15:[16, 65536], 16:[32, 32768], 17:[32, 65536], 18:[32, 131072]}
    server_limitation=[[56,131072], [84, 262144], [112, 196608]]
    prediction =  {1:0,2:0,3:0,4:9,5:38,6:114,7:238,8:35,9:0,10:188,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0}
    start_time = time.clock()
    duration_time = 50
    population_scale = 100
    res=genetic_alg_boxing(prediction, server_limitation, flavor_specification, population_scale, start_time, duration_time, elite_scale=population_scale/5)
    
if __name__ == '__main__':
    test()