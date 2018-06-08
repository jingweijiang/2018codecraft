# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 13:09:29 2018

@author: Administrator
"""
import random
import math
import copy
import time

def print_two_dimension_data(two_dimension_data):
    print("######################################")
    for i in range(len(two_dimension_data)):
        print(i, ": ", two_dimension_data[i])
    print("######################################")
          
          
def fitting_curve(datas, limitation4b, decay, population_scale, elite_scale, duration_time, predict_info):
    def initial_population():
        return [[random.uniform(limitation4a[0], limitation4a[1]), random.uniform(limitation4b[0], limitation4b[1])] for i in range(population_scale)]
    
    def initial_decay(decay):
        interval = (1-decay)/_len
        return [decay + interval*i for i in range(_len)]
    
    def model_out(i, individual):
        return individual[0] * math.exp(individual[1] * i)
    
    def evaluate_individual(individual):
        def generate_prediction():
            
            model_prediction = []
            for i in range(_len):
#                print("individual[0]:", individual[0])
#                print("individual[1] * i:", individual[1] * i)
                model_prediction.append(model_out(i, individual))
            return model_prediction
        
        abs_subs = lambda (x0, x1, x2) : abs(x0 - x1) * x2
#        print("individual:", individual)
        prediction = generate_prediction()
        return [sum(map(abs_subs, zip(datas, prediction, decay_list))), individual]
        
        
        
    def evaluate_individuals(population):
        evaluations = []
        for pop in population:
            evaluations.append(evaluate_individual(pop))
        return evaluations
    
    def Propagate_descendants(evaluation):
        def mutate(individual0, individual1):
            
            individual = copy.copy(individual0)
#            print("individual:", individual)
#            print("individual1:", individual1)
            mutation_point = random.randint(0, 1) # minus one because endpoint included
#            print("mutation_point:", mutation_point)
            if (random.random() < 0.8):
#                print("first")
                individual[mutation_point] += random.uniform(-0.1, 0.1)
            else:
##                print("second")
                individual[mutation_point] = individual1[mutation_point]
#            print("individual:", individual)
#            print("########################################")
            return individual
                
        
        del(evaluation[elite_scale:])
#        print("len(evaluation):", len(evaluation))
#        elite_group = evaluation[0:elite_scale]
        mutated_group = []
        for i in range(elite_scale, population_scale):
            individual0 = random.sample(evaluation, 1)[0][1]
            individual1 = random.sample(evaluation, 1)[0][1]
#            mutated_individual = mutate(individual)
#            elite_group.append(evaluate_individual())
            mutated_group.append(mutate(individual0, individual1))
#        print("mutated_group:", mutated_group)
#        evaluate_individual(mutated_group, )
        return mutated_group  
    
    
    limitation4a = [datas[0]-10.0, datas[0]+10.0]
    _len = len(datas)
    decay_list = initial_decay(decay)
#    print("decay_list:", decay_list)
    population = initial_population()
#    print("population:", population)
    evaluation = evaluate_individuals(population)
    evaluation.sort()
#    print("evaluation:")
#    print_two_dimension_data(evaluation)
#    input()
    start_time = time.clock()
    while(time.clock()-start_time < duration_time):
        descendants = Propagate_descendants(evaluation)
#        print("descendants:")
#        print_two_dimension_data(descendants)        
        evaluation.extend(evaluate_individuals(descendants))
        evaluation.sort()
#        print("evaluation:")
#        print_two_dimension_data(evaluation)   
        print("the best individual you get:")
        print(evaluation[0])
#        input()
    predict_start_point = len(datas)
    return sum([model_out(index+predict_start_point, evaluation[0][1])*coe  for (index, coe) in predict_info])

#datas, limitation4b, decay, population_scale, elite_scale, duration_time, predict_info
print(fitting_curve([math.e, pow(math.e, 2), pow(math.e, 3), pow(math.e, 4), pow(math.e, 5), pow(math.e, 6), pow(math.e, 7)], [0.0, 3.0], 0.5, 40000, 100, 2, [(0, 1), (1, 0.2)]))