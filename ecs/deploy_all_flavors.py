# coding=utf-8
import time
import random
import copy
FEED_COUNT = 5
SECOND_FEED_COUNT = 800
random.seed(0)
def genetic_alg_boxing(prediction, server_limitation, flavor_specification, population_scale, start_time, duration_time, elite_scale):

    def v0_a_v1_inplace(v0, v1):
        #v0 as inplace
        for index in range(len(v0)):
            v0[index] = v0[index] + v1[index]
    
    def v0_m_v1(v0, v1):
        result = []
        for index in range(len(v0)):
            result.append(v0[index]-v1[index])
        return result
    
    def v0_a_v1(v0, v1):
        result = []
        for index in range(len(v0)):
            result.append(v0[index]+v1[index])
        return result

    def v0_ge_v1(v0, v1):
        result = 1
        for index in range(len(v0)):
            result *= (v0[index] >= v1[index])
        return result    
    
    def v0_s_v1(v0, v1):
        result = 1
        for index in range(len(v0)):
            result *= (v0[index] < v1[index])
        return result
    
    def explane_prediction(prediction):
        encode_prediction = []
        for key, value in prediction.items():
            for i in range(value):
                encode_prediction.append(key)#because index0 - flavor1
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
#        print("summa:", summations)
#        print("server_summ:", server_summation)
        result = 0
        for index, summation in enumerate(summations):
            result += 1000000 * summation / server_summation[index] 
#        print("result:", result)
        return result
    
    
    def evaluate_individual(population, server_limitation, flavor_specification, summation):

        evaluation = []
        for individual in population:
            #评估
#            print(evaluation_function(individual, allocate_server(individual, server_limitation, flavor_specification), flavor_specification, server_limitation, optimization_target))
            evaluation.append(
                    (evaluation_function(summation, allocate_server(individual, server_limitation, flavor_specification)[1], server_limitation)
                    , individual))
        return evaluation    
    def evaluate_single_score(summation, server_limitation):
        result = 0
        for index in range(len(summation)):
            result += (summation[index] * 100) / server_limitation[index]
#            print("result:", result)
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
                    

        
        def evaluate_all(summations, server_limitations, end_points, flavor_specifications, individual):
    #        print("summations:", summations)

    #        print("flavor_specifications:", flavor_specifications)
            results = []
            for index in range(len(summations)):
                v0_m_v1_inplace(summations[index], flavor_specifications[individual[end_points[index]]])
                results.append(evaluate_single_score(summations[index], server_limitations[index]))
    #        print("after minus:", summations)
    #        print("results:", results)
            return results.index(max(results)), end_points[results.index(max(results))]
    
        def evaluate_last(summations, server_limitations, flavor_specifications, checked_results):
            results = []
            for index, check in enumerate(checked_results):
                if check == True:
                    results.append(0)
                else:
                    results.append(evaluate_single_score(summations[index], server_limitations[index]))
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

        #    print("mutation_point:", mutation_point)
            direction = "left" if left_len < right_len else "right" if left_len > right_len else "center"

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
    
    def print_two_dimension_data(two_dimension_data):
        print("######################################")
        for i in range(len(two_dimension_data)):
            print(i, ": ", two_dimension_data[i])
        print("######################################")
              
    def filter_best_candidates(candidates):
        best_score = candidates[0][0]
        candidate_group = []
        for candidate in candidates:
            if candidate[0] == best_score:
                candidate_group.append([best_score, allocate_server(candidate[1], server_limitation, flavor_specification)])
        return candidate_group    

    def evaluate_single(summation, server_limitation):
        result = 1
        surplus = []
        for index in range(len(summation)):
            temp_result =  1 - float(summation[index]) / server_limitation[index]
            surplus.append(server_limitation[index]-summation[index])
#            print("############################################################")
#            print("surplus:", surplus)
#            print("temp_result:", temp_result)
            result *= temp_result
#            print("result:", result)
#            print("############################################################")
        return result, surplus

    def choose_good_candidate(candidates, flavor_specification, server_limitation):
        '''#######################################################################
        reuslt : [allocated_virtual_machine, box_server_index]
        surpluses : [[cpu_idleness, memory_idleness], [cpu_idleness, memory_idleness].......] 
        ###########################################################################'''
        index_list = []
        surplueses = []
        idlenesseses = []
        for index_servers, servers in enumerate(candidates):
            surpluses = []
            idlenesses = []
            material = zip(servers[1][0], servers[1][1])
            idleness = 0
            for index_server, server in enumerate(material):
                idleness_temp, surplus = evaluate_single(summation_f(server[0], flavor_specification), server_limitation[server[1]])
                idleness += idleness_temp
                idlenesses.append([idleness_temp, index_server])
                surpluses.append([surplus, index_server])
            index_list.append([idleness, servers[1], index_servers])
            surplueses.append(surpluses)
            idlenesseses.append(idlenesses)
        index_list.sort(reverse=True)
#        print("index_list:", index_list)
        result = index_list[0][1]
        return result, surplueses[index_list[0][2]], idlenesseses[index_list[0][2]]
    
    def feed_flavor_balance_create(prediction):
        feed_flavor_balance={}
        for key in prediction.keys():
            feed_flavor_balance[key]=int(round(prediction[key])) 
        return feed_flavor_balance







    def generate_recommend(servers, flavor_material, flavor_specification, server_limitation):
#        print("servers:", servers)
#        print("flavor_material:", flavor_material)
#        print("flavor_specification:", flavor_specification)
#        print("server_limitation:", server_limitation)
        result = []
        for flavor in flavor_material:
            servers.append(flavor)
            summation = single_server_summation(servers, flavor_specification)
            score = float(evaluate_single_score(summation, server_limitation))/2
            result.append([score, flavor])
            del(servers[-1])
#        print("result:", result)
        result.sort(reverse = True)
        result_correct = []
        for _res in result:
            if(_res[0] <= 100):
                result_correct.append(_res)
        if(len(result_correct) == 0):
            return []
        else:
            result_correct = zip(*result_correct)
            return result_correct[-1]
            
            
            
    def feed_servers(surpluses, feed_flavor_balance, flavor_specification, server_limitation):
#        print("surpluses:", surpluses)
#        print("feed_flavor_balance:", feed_flavor_balance)
#        print("flavor_specification:", flavor_specification)
#        input()
        ###############################child function############################################################
        def feed_single_server(surplus, server_index, servers, server_kind, flavor_specification, feed_flavor_balance):
#            print("surplus:", surplus)
#            print("server_index:", server_index)
#            print("servers:", servers)
#            print("flavor_specification:", flavor_specification)
#            print("feed_flavor_balance:", feed_flavor_balance)
            ###############################gradchild function############################################################

            def v0_g_v1_all(vector0, vector1):
                matrix = zip(vector0, vector1)
                result = [v0 >= v1 for (v0, v1) in matrix]
                judge = sum(result) / len(result)
                if judge == 1:
                    return True
                else:
                    return False
            ###############################gradchild function############################################################
            flavor_material = feed_flavor_balance.keys()
            flavor_material.sort(reverse = True)
            flavor_material = generate_recommend(servers[server_index], flavor_material, flavor_specification, server_limitation[server_kind[server_index]])
#            print("flavor_material:", flavor_material)
            for flavor in flavor_material:
                    if(v0_g_v1_all(surplus, flavor_specification[flavor]) and (feed_flavor_balance[flavor] > 0)):
#                            print("flavor:", flavor)
                        feed_flavor_balance[flavor] -= 1
#                            print("server_index:", server_index)
#                            print("servers[server_index]:", servers[server_index])
                        servers[server_index].append(flavor)
#                            print("servers[server_index]:", servers[server_index])
                        break
        ###############################child function############################################################                
                        
            
        
#            print("surpluses[-1]:", surpluses[-1])
        for surplus in surpluses[0]:
#                print("############################################################################################")
#                print("surplus:", surplus)
            feed_single_server(surplus[0], surplus[1], surpluses[-1][0], surpluses[-1][-1], flavor_specification, feed_flavor_balance)
#            print("feed_flavor:", feed_flavor_balance)
#            print("surpluses[-1]:", surpluses[-1])           
        return surpluses[-1]

    def joint(lists):
        result = []
        for _list in lists:
            result.extend(_list)
        return result

    def servers2surplus(servers, flavor_specification, server_limitation):
        servers = zip(*servers)
        surpluses = []
        for index, server in enumerate(servers):
#            print("server:")
#            print(server)
            _, surplus = evaluate_single(summation_f(server[0], flavor_specification), server_limitation[server[1]])    
            surpluses.append([surplus, index])
        return surpluses
    
    def Roulette_Wheel_Selection(scores, objects):
#        print("scores:", scores)
#        print("objects:", objects)
        _sum = sum(scores) + 0.00000001
        p = [float(_obj) / _sum for _obj in scores]
#        print("p:", p)
        point = random.random()
#        print("point:", point)
        for index, _p in enumerate(p):
#            print("sum(p[:index]):", sum(p[:index]))
            if point < sum(p[:index+1]):
                return objects[index]
        return None
    def choose_dimension_biggest(surplurse):
#        print("surplurse:", surplurse)
        first_dimension_object = []
        second_dimension_object = []
#        temp = []
        for surplus in surplurse:
            first_dimension_object.append([surplus[0][0], surplus[-1]])
            second_dimension_object.append([surplus[0][1], surplus[-1]])
        first_dimension_object.sort()
        second_dimension_object.sort()
        first_dimension_object = zip(*first_dimension_object)
        second_dimension_object = zip(*second_dimension_object)
        first_dimension = Roulette_Wheel_Selection(first_dimension_object[0], first_dimension_object[1])
        second_dimension = Roulette_Wheel_Selection(second_dimension_object[0], second_dimension_object[1])
        if first_dimension == None or second_dimension == None:
            return None, None
#            temp.append([surplus[0][-1], surplus[-1]])
#        temp.sort(reverse=True)
#        second_dimension = surplurse[temp[0][-1]]
#        surplurse.sort(reverse=True)
#        first_dimension = surplurse[0]
        return surplurse[first_dimension], surplurse[second_dimension]
    
    def choose_virtual2exchange(first_surplus, first_servers, second_surplus, second_servers, flavor_specification):
        def recommend_virtual(largest, dimension):
            if dimension == 0:
                if largest >= 32:
                    return [16, 17, 18, 13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3]
                if largest >= 16:
                    return [13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3]
                if largest >= 8:
                    return [10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3]
                if largest >= 4:
                    return [7, 8, 9, 4, 5, 6, 1, 2, 3]
                if largest >= 2:
                    return [4, 5, 6, 1, 2, 3]
                if largest >= 1:
                    return [1, 2, 3]
                else:
                    return []
            if dimension == 1:
                if largest >= 131072:
                    return [18, 15, 17, 12, 14, 16, 9, 11, 13, 6, 8, 10, 3, 5, 7, 2, 4, 1]
                if largest >= 65536:
                    return [15, 17, 12, 14, 16, 9, 11, 13, 6, 8, 10, 3, 5, 7, 2, 4, 1]
                if largest >= 32768:
                    return [12, 14, 16, 9, 11, 13, 6, 8, 10, 3, 5, 7, 2, 4, 1]
                if largest >= 16384:
                    return [9, 11, 13, 6, 8, 10, 3, 5, 7, 2, 4, 1]
                if largest >= 8192:
                    return [6, 8, 10, 3, 5, 7, 2, 4, 1]
                if largest >= 4096:
                    return [3, 5, 7, 2, 4, 1]
                if largest >= 2048:
                    return [2, 4, 1]
                if largest >= 1024:
                    return [1]
                else:
                    return []
        def confirm_virtual(virtual, servers):
            for _v in virtual:
                if _v in servers:
                    return _v
            return None
        first_largest = first_surplus[0][0]
        second_largest = second_surplus[0][1]
        recommend_extract2one = recommend_virtual(first_largest, 0)
        recommend_extract2two = recommend_virtual(second_largest, 1)
#        print("first_largest:", first_largest)
#        print("second_largest:", second_largest)
#        print("recommend_extract2two:", recommend_extract2two)
#        print("recommend_extract2one:", recommend_extract2one)

        while(True):
#            input()
            virtual_extract2one = confirm_virtual(recommend_extract2one, second_servers)
            virtual_extract2two = confirm_virtual(recommend_extract2two, first_servers)
#            print("virtual_extract2one:", virtual_extract2one)
#            print("virtual_extract2two:", virtual_extract2two)
            if(virtual_extract2one == None or virtual_extract2two == None):
                return None, None

            if(virtual_extract2one == virtual_extract2two):
                if(len(recommend_extract2one) > len(recommend_extract2two)):
                    del(recommend_extract2one[recommend_extract2one.index(virtual_extract2one)])
#                    print("first del to 1")
#                    print(recommend_extract2one)
                    if(len(recommend_extract2one) == 0):
                        return None, None
                    else:
                        continue
                else:
                    del(recommend_extract2two[recommend_extract2two.index(virtual_extract2two)])
#                    print("first del to 2")
#                    print(recommend_extract2two)
                    if(len(recommend_extract2two) == 0):
                        return None, None
                    else:
                        continue  


#            print("second_servers:", second_servers)
#            print("first_servers:", first_servers)

#            input()
            first_surplus_extracted = v0_a_v1(first_surplus[0], flavor_specification[virtual_extract2two])
            second_surplus_extracted = v0_a_v1(second_surplus[0], flavor_specification[virtual_extract2one])
#            print("first_surplus_extracted:", first_surplus_extracted)
#            print("flavor_specification[virtual_extract2one]:", flavor_specification[virtual_extract2one])
#            print("v0_ge_v1:", v0_ge_v1(first_surplus_extracted, flavor_specification[virtual_extract2one]))
#            print("v0_s_v1:", v0_s_v1(first_surplus_extracted, flavor_specification[virtual_extract2one]))
#            print("second_surplus_extracted:", second_surplus_extracted)
#            print("flavor_specification[virtual_extract2two]:", flavor_specification[virtual_extract2two])
#            print("v0_ge_v1:", v0_ge_v1(second_surplus_extracted, flavor_specification[virtual_extract2two]))
#            print("v0_s_v1:", v0_s_v1(second_surplus_extracted, flavor_specification[virtual_extract2two]))
            if (v0_ge_v1(first_surplus_extracted, flavor_specification[virtual_extract2one]) and v0_ge_v1(second_surplus_extracted, flavor_specification[virtual_extract2two])):
                return virtual_extract2two, virtual_extract2one
            elif(v0_ge_v1(first_surplus_extracted, flavor_specification[virtual_extract2one]) == False):
                del(recommend_extract2one[recommend_extract2one.index(virtual_extract2one)])
#                print("second del to 1")
#                print(recommend_extract2one)
                if(len(recommend_extract2one) == 0):
                    return None, None
            elif(v0_ge_v1(second_surplus_extracted, flavor_specification[virtual_extract2two]) == False):
                del(recommend_extract2two[recommend_extract2two.index(virtual_extract2two)])
#                print("second del to 2")
#                print(recommend_extract2two)
                if(len(recommend_extract2two) == 0):
                    return None, None
        
            
        
    
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
        print(evaluation[0][0]/float(20000), len(allocate_server(evaluation[0][1], server_limitation, flavor_specification)[1]))
    
  
    
    ######################cheat#########################################
#    print("filter_best_candidates:")
    candidates = filter_best_candidates(evaluation) #
    candidate, surpluses, idlenesses = choose_good_candidate(candidates, flavor_specification, server_limitation)
#    print("candidate:")
#    print_two_dimension_data(candidate)
#    print("surpluses:")
#    print_two_dimension_data(surpluses)
#    print("idlenesses:")
#    print_two_dimension_data(idlenesses)    
    
    result = joint(candidate[0])
    
    summation = summation_f(result, flavor_specification)
    
    allo_result, server_result = allocate_server(result, server_limitation, flavor_specification)
    print("one server with virtual machine       summation of virtual machine     server specification")
    for index, allo in enumerate(allo_result):
        print(allo, single_server_summation(allo, flavor_specification), server_limitation[server_result[index]])      
    
    
    
    feed_flavor_balance = feed_flavor_balance_create(prediction)
    candidate = feed_servers([surpluses, candidate], feed_flavor_balance, flavor_specification, server_limitation)
    
    
    
#    print("candidate:")
#    print_two_dimension_data(candidate)      
    
    
    
    cheat_result = joint(candidate[0])
    summation = summation_f(cheat_result, flavor_specification)
    print("the best individual you cheat:")
    print(evaluation_function(summation, allocate_server(cheat_result, server_limitation, flavor_specification)[1], server_limitation)/float(20000), len(allocate_server(cheat_result, server_limitation, flavor_specification)[1])) 


    for i in range(FEED_COUNT):
        surpluses = servers2surplus(candidate, flavor_specification, server_limitation)
        candidate = feed_servers([surpluses, candidate], feed_flavor_balance, flavor_specification, server_limitation)
        cheat_result = joint(candidate[0])
        summation = summation_f(cheat_result, flavor_specification)
        print("the best individual you cheat:")
        print(evaluation_function(summation, allocate_server(cheat_result, server_limitation, flavor_specification)[1], server_limitation)/float(20000), len(allocate_server(cheat_result, server_limitation, flavor_specification)[1]))
    allo_result, server_result = allocate_server(cheat_result, server_limitation, flavor_specification)
    print("one server with virtual machine       summation of virtual machine     server specification")
    for index, allo in enumerate(allo_result):
        print(allo, single_server_summation(allo, flavor_specification), server_limitation[server_result[index]])   
#    input()
    
    
    
    
    
    ###############################second optimization#########################################
    for i in range(SECOND_FEED_COUNT):
        surpluses = servers2surplus(candidate, flavor_specification, server_limitation)
        first_dimension_biggset_surplus, second_dimension_biggest_surplus = choose_dimension_biggest(surpluses)
        if (first_dimension_biggset_surplus == None or second_dimension_biggest_surplus == None):
            break
#        print("first_dimension_biggset_surplus:", first_dimension_biggset_surplus)
#        print("second_dimension_biggest_surplus:", second_dimension_biggest_surplus)
#        print("origin first servers:", allo_result[first_dimension_biggset_surplus[-1]])
#        print("origin second servers:", allo_result[second_dimension_biggest_surplus[-1]])
        virtual_extract2two, virtual_extract2one = choose_virtual2exchange(first_dimension_biggset_surplus, allo_result[first_dimension_biggset_surplus[-1]], 
                                      second_dimension_biggest_surplus, allo_result[second_dimension_biggest_surplus[-1]], 
                                      flavor_specification)
        if(virtual_extract2two == None or virtual_extract2one == None):
            continue
        del(allo_result[first_dimension_biggset_surplus[-1]][allo_result[first_dimension_biggset_surplus[-1]].index(virtual_extract2two)])
        allo_result[first_dimension_biggset_surplus[-1]].append(virtual_extract2one)
        del(allo_result[second_dimension_biggest_surplus[-1]][allo_result[second_dimension_biggest_surplus[-1]].index(virtual_extract2one)])
        allo_result[second_dimension_biggest_surplus[-1]].append(virtual_extract2two)
#        print("first servers:", allo_result[first_dimension_biggset_surplus[-1]])
#        print("second servers:", allo_result[second_dimension_biggest_surplus[-1]])
        candidate = [allo_result, server_result]
    #    surpluses = servers2surplus(candidate, flavor_specification, server_limitation)
        surpluses = servers2surplus(candidate, flavor_specification, server_limitation)
        candidate = feed_servers([surpluses, candidate], feed_flavor_balance, flavor_specification, server_limitation)
        cheat_result = joint(candidate[0])
        summation = summation_f(cheat_result, flavor_specification)
        print("##################################################################################")
        print("the best individual you cheat:")
        print(evaluation_function(summation, allocate_server(cheat_result, server_limitation, flavor_specification)[1], server_limitation)/float(20000), len(allocate_server(cheat_result, server_limitation, flavor_specification)[1]))
    allo_result, server_result = allocate_server(cheat_result, server_limitation, flavor_specification)
    print("one server with virtual machine       summation of virtual machine     server specification")
    for index, allo in enumerate(allo_result):
        print(allo, single_server_summation(allo, flavor_specification), server_limitation[server_result[index]]) 
    
    
    
    
    
    
    print("################## finally result ########################")
    print("one server with virtual machine       summation of virtual machine     server specification")
    for index, allo in enumerate(candidate[0]):
        print(index+1, ":", single_server_summation(allo, flavor_specification), server_limitation[candidate[1][index]], surpluses[index])
        
#    allo_result, server_result = allocate_server(evaluation[0][1], server_limitation, flavor_specification)
#    print("one server with virtual machine       summation of virtual machine     server specification")
#    for index, allo in enumerate(allo_result):
#        print(allo, single_server_summation(allo, flavor_specification), server_limitation[server_result[index]])
#        
#    allo_result, server_result = allocate_server(evaluation[1][1], server_limitation, flavor_specification)
#    print("one server with virtual machine       summation of virtual machine     server specification")
#    for index, allo in enumerate(allo_result):
#        print(allo, single_server_summation(allo, flavor_specification), server_limitation[server_result[index]])
#    
#    allo_result, server_result = allocate_server(evaluation[2][1], server_limitation, flavor_specification)
#    print("one server with virtual machine       summation of virtual machine     server specification")
#    for index, allo in enumerate(allo_result):
#        print(allo, single_server_summation(allo, flavor_specification), server_limitation[server_result[index]])
    return allo_result,server_result

def print_two_dimension_data(two_dimension_data):
    print("######################################")
    for i in range(len(two_dimension_data)):
        print(i, ": ", two_dimension_data[i])
    print("######################################")
          
def gen_predict_res(flavor_names,res_alloc):
    flavor_prediction={}
    for single_server_alloc in res_alloc:
        set_single_server=set(single_server_alloc)
        for flavor in set_single_server:
            if flavor_prediction.has_key('flavor'+str(flavor)):
                flavor_prediction['flavor'+str(flavor)]+=single_server_alloc.count(flavor)
            else:
                flavor_prediction['flavor'+str(flavor)]=single_server_alloc.count(flavor)
    print ('flavor_prediction',flavor_prediction)
    # flavor_prediction=res_alloc
    res=''
    sum_of_flavors=0
    for name in flavor_names:
        if flavor_prediction.has_key(name)==False:
            flavor_prediction[name]=0
        sum_of_flavors+=flavor_prediction[name]
        res+=name+' '+str(flavor_prediction[name])+'\n'
    res=str(sum_of_flavors)+'\n'+res
    return res



def deploy_flavors(flavor_names,flavor_prediction,flavor_type,server_cfg,serverNames):
    print ('is working...')
    print ('flavor_names:',flavor_names)
    print ('flavor_prediction',flavor_prediction)
    flavor_names_int=[int(i[6:]) for i in flavor_names]
    flavor_names_int.sort()
    flavor_names=['flavor'+str(i) for i in flavor_names_int]
    
    print ('flavor_type:',flavor_type)
    print ('server_cfg:',server_cfg)
    print ('serverNames',serverNames)

    prediction={}
    flavor_specification={}
    for name in flavor_names:
        prediction[int(name[6:])] = flavor_prediction[name]
        flavor_specification[int(name[6:])] = flavor_type[name]
    server_limitation=[[server[0],server[1]*1024] for server in server_cfg]
    print ('server_limitation:', server_limitation)
    start_time = time.clock()
    duration_time = 45
    population_scale = 200
    print ('flavor_names:',flavor_names)
    print ('prediction',prediction)
    print ('flavor_specification:',flavor_specification)
    print ('server_limitation:',server_limitation)
    print ('serverNames',serverNames)
    flavor_specification = {1: [1, 1024], 2: [1, 2048], 3: [1, 4096], 4: [2, 2048], 5: [2, 4096], 6: [2, 8192],
                            7: [4, 4096], 8: [4, 8192], 9: [4, 16384], 10: [8, 8192], 11: [8, 16384],
                            12: [8, 32768], 13: [16, 16384], 14: [16, 32768], 15: [16, 65536], 16: [32, 32768],
                            17: [32, 65536], 18: [32, 131072]}
    res_alloc,res_server = genetic_alg_boxing(prediction, server_limitation, flavor_specification, population_scale, start_time,duration_time, elite_scale=population_scale / 5)

    res=''

    server_count=[res_server.count(0),res_server.count(1),res_server.count(2)]
    for index,count in enumerate(server_count):
        if count==0:
            continue
        res+=serverNames[index]+' '+str(count)+'\n'
        server_i_count=1
        for index_alloc,alloc in enumerate(res_alloc):
            if res_server[index_alloc]!=index:
                continue
            res+=serverNames[index]+'-'+str(server_i_count)+' '
            flavor_set = set(alloc)
            for single in flavor_set:
                res+='flavor'+str(single)+' '+str(alloc.count(single))+' '
            res+='\n'
            server_i_count+=1
        res+='\n'

    # print("res_alloc",res_alloc)
    res_predict=gen_predict_res(flavor_names,res_alloc)
    res=res_predict+'\n'+res
    return res

def example():
    flavor_predict_names = [ 'flavor2', 'flavor3', 'flavor4', 'flavor5', 'flavor6', 'flavor7', 'flavor8',                         'flavor9', 'flavor10', 'flavor11','flavor12', 'flavor13','flavor14','flavor15','flavor16','flavor17','flavor18']
    # flavor_predict_num=[312,122,434,125,123,214,552,612,32,239,112,432,400,340]
    # flavor_prediction={1: 32, 2: 34, 3: 23, 4: 13, 5: 50, 6: 15, 7: 23, 8: 20, 9: 40, 10: 60, 12: 34, 13: 27, 14: 23, 15: 43}
    flavor_prediction = { 'flavor2': 44, 'flavor3': 12, 'flavor4': 26, 'flavor5': 68, 'flavor6': 2,
                         'flavor7': 7, 'flavor8': 86, 'flavor9': 22, 'flavor10': 3,'flavor11': 36, 'flavor12': 7, 'flavor13': 9,
                         'flavor14': 8, 'flavor15': 0,'flavor16':4,'flavor17':5,'flavor18':0}
    flavor_type = {'flavor1': [1, 1024], 'flavor2': [1, 2048], 'flavor3': [1, 4096], 'flavor4': [2, 2048],
                   'flavor5': [2, 4096], 'flavor6': [2, 8192], 'flavor7': [4, 4096], 'flavor8': [4, 8192],
                   'flavor9': [4, 16384], 'flavor10': [8, 8192], 'flavor11': [8, 16384], 'flavor12': [8, 32768],
                   'flavor13': [16, 16384], 'flavor14': [16, 32768], 'flavor15': [16, 65536],'flavor16':[32, 32768], 'flavor17':[32, 65536], 'flavor18':[32, 131072]}
    server_cfg = [[56, 128], [84, 256], [112, 192]]

    server_names=['General','Large-Memory','High-Performance']
    res = deploy_flavors(flavor_predict_names, flavor_prediction, flavor_type, server_cfg,server_names)
    # for index,alloc in enumerate(res):
    #     print ('result to deal with:',alloc,'server:',server_cfg[server_result[index]])
    print res

def test():
    flavor_specification = {1:[1, 1024], 2:[1, 2048], 3:[1, 4096], 4:[2, 2048], 5:[2, 4096], 6:[2, 8192],
                            7:[4, 4096], 8:[4, 8192], 9:[4, 16384], 10:[8, 8192], 11:[8, 16384],
                            12:[8, 32768], 13:[16, 16384], 14:[16, 32768], 15:[16, 65536], 16:[32, 32768], 17:[32, 65536], 18:[32, 131072]}
    #服务器配置
    server_limitation=[[56,131072], [84, 262144], [112, 196608]]
    prediction =  {2:44,3:12,4:26,5:68,6:2,7:8,8:86,9:22,10:3,11:36,12:7,13:9,14:8,15:0,16:4,17:5,18:0}
    #定义运行时间
    start_time = time.clock()
    duration_time = 10
    #总群大小
    population_scale = 100
    res=genetic_alg_boxing(prediction, server_limitation, flavor_specification, population_scale, start_time, duration_time, elite_scale=population_scale/5)
    
if __name__ == '__main__':
    print(example())
