#from _numpy import _numpy as np
import random
import math
import time
import copy
eps = 0.000000001
def print_two_dimension_data(two_dimension_data):
    print("######################################")
    for i in range(len(two_dimension_data)):
        print(two_dimension_data[i])
    print("######################################")

def get_col_data(two_dimension_data, col):
#    print("two_dimension_data:", two_dimension_data)
    result = []
#    print("len(two_dimension_data):", len(two_dimension_data))
    for i in range(len(two_dimension_data)):
#        print("i:", i)
#        print("col:", col)
        result.append(two_dimension_data[i][col])
#        print("result:", result)
    return result

def smooth4two_dimension_data(two_dimension_data):
    def smooth(col_data, alpha = 0.8):
        smoothed = [0]
        col_data = [0] + col_data
#        print("col_data:", col_data)
        for day, data in enumerate(col_data):
#            day = index + 1
            if day == 0:
                pass
            else:
#                print("alpha * col_data[day - 1]:", alpha * col_data[day - 1])
#                print("(1 - alpha) * col_data[day]:", (1 - alpha) * col_data[day])
#                print("(1 - pow(alpha, day):", (1 - pow(alpha, day)))
                smoothed.append((alpha * smoothed[day - 1] + (1 - alpha) * col_data[day]) * (1 - pow(alpha, day)))
        del(smoothed[0])
        return smoothed
        
#    print("two_dimension_data[0]:", two_dimension_data[0])
    result = []
    row = len(two_dimension_data)
    col = len(two_dimension_data[0])
    
    for j in range(col):
        col_data = get_col_data(two_dimension_data, j)
#        print("col_data:", col_data)
#        print("smooth:", smooth(col_data))
        col_data = smooth(col_data)
#        result = []
        result.append(col_data)
    return zip(*result)

def divide_week_data(two_dimension_data, days):
    def weeked_data(col_data):
#        print("col_data:", col_data)
        result = []
#        for i in range(-1, -(len(col_data)), -7):
#            print("i+1:", i+1)
#            print("i-6:", i-6)
#            print("col_data[i-6 : i+1]:", col_data[i-6 : i] + [col_data[i]])
#            result.append(sum(col_data[i-6 : i] + [col_data[i]]))
#        print("result:", result)
        for i in range(len(col_data)%days, len(col_data)/days*days + len(col_data)%days, days):
#            print("i:", i)
#            print("i+7:", i+7)
#            print("col_data[i : i+days]:",col_data[i : i+days])
            result.append(sum(col_data[i : i+days])+1)
#        print("result:", result)            
        return result
    
    
    
    result = []
    row = len(two_dimension_data)
    col = len(two_dimension_data[0])
    for j in range(col):
        col_data = get_col_data(two_dimension_data, j)
#        print("col_data:", col_data)
#        print("smooth:", smooth(col_data))
        col_data = weeked_data(col_data)
#        result = []
        result.append(col_data)
    return zip(*result)

def generate_instance(two_dimension_data):
    def generate(col_data):
        temp = []
        result = []
        for i in range(0, len(col_data)-3):
            temp = (col_data[i:i+4])
            result.append([temp[0:3], [temp[3]]])
        target = [col_data[:], None]
        return result, target
    
    result = []
    targets = []
    row = len(two_dimension_data)
    col = len(two_dimension_data[0])
    for j in range(col):
        col_data = get_col_data(two_dimension_data, j)
#        print("col_data:", col_data)
#        print("smooth:", smooth(col_data))
        col_data, target_data = generate(col_data)
#        result = []
        result.extend(col_data)
        targets.append(target_data)
    return result, targets

def error_sum_of_square(vector0, vector1):
    result = []
    for i in range(len(vector0)):
        result.append(pow(vector0[i]-vector1[i], 2))
    return pow(sum(result), 0.5)

def compensate(target_vector0, neigh_vector1):
    mean_target = float(sum(target_vector0[0])) / len(target_vector0[0])
    mean_neigh = float(sum(neigh_vector1[0])) / len(neigh_vector1[0])
    return neigh_vector1[1][0] + mean_target - mean_neigh

def secret_algorithm(target_vector, vector_of_list):
    distance = []
    for index, vector in enumerate(vector_of_list):
        distance.append([error_sum_of_square(target_vector[0], vector[0]), compensate(target_vector, vector), vector[0]])
    distance.sort()
    del(distance[6:])
    print_two_dimension_data(distance)
#    compensation = []
    target_vector[1] = sum([1.0 / (distance[i][0] if abs(distance[i][0]) > eps else eps) * distance[i][1] for i in range(len(distance))]) / sum( [1.0 / (distance[i][0] if abs(distance[i][0]) > eps else eps) for i in range(len(distance))])
    return target_vector[1]

def predict(target_of_list, vector_of_list, days):
    print("days:", days)
#    input()
    if(days == 7):
        result = []
        for target in target_of_list:
            result.append(secret_algorithm(target, vector_of_list))
    elif(days>7 or days <=14):
        result0 = []
        result1 = []
        for target in target_of_list:
            result0.append(secret_algorithm(target, vector_of_list))
        print("result0:", result0)
        ##update the target data
        for index, target in enumerate(target_of_list):
            del(target[0][0])
            target[0].append(result0[index])
            result1.append(secret_algorithm(target, vector_of_list))
        print("target_of_list")
        print_two_dimension_data(target_of_list)
        print("result1:", result1)

        len_result = len(result0)
        result = []
        coe = float(days - 7) / 7 
        for index in range(len_result):
            result.append(result0[index] + coe * result1[index])
        
        
    return result

def mean_algorithm(target_vector, coe, predict_days):
    return int(round(sum(target_vector[0])/len(target_vector[0]) * coe * float(predict_days)/7))

def predict_mean(target_of_list, coe, predict_days):
    result = []
    for target in target_of_list:
        result.append(mean_algorithm(target, coe, predict_days))
    return result


def fitting_curve(datas, limitation4b, decay, population_scale, elite_scale, duration_time, predict_info):
    def initial_population():
        return [[random.uniform(limitation4a[0], limitation4a[1]), random.uniform(limitation4b[0], limitation4b[1])] for i in range(population_scale)]
    
    def initial_decay(decay):
        interval = (decay[1]-decay[0])/(_len-1)
        return [decay[0] + interval*i for i in range(_len)]
    
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
        
        abs_subs = lambda (x0, x1, x2) : pow((x0 - x1), 2) * x2
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
            if (random.random() < 0.2):
#                print("first")
                individual[mutation_point] += random.uniform(-0.05, 0.05)
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
    print("decay_list:", decay_list)
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
        print("##########################################")
        print("the best individual you get:")
        print(evaluation[0])
#        input()
    predict_start_point = len(datas)
    print("orgin_datas:", datas)
    print("model result:", [int(model_out(i, evaluation[0][1])) for i in range(_len+3)])
    print("predict_info:", predict_info)
    return int(sum([model_out(index+predict_start_point, evaluation[0][1])*coe  for (index, coe) in predict_info]))
        
        
def model_out(k, b, x):
    return [k * xi + b for xi in x]

def grad_update(x_data, y_data, decay, k_origin, b_origin, alpha, episode):
    train_data = zip(x_data, y_data, decay)
    for i in range(episode):
        k_new = k_origin + alpha * sum([((yi - (k_origin * xi + b_origin)) * xi) * decayi for (xi, yi, decayi) in train_data])
        b_new = b_origin + alpha * sum([(yi - (k_origin * xi + b_origin)) * decayi for (xi, yi, decayi) in train_data])
        k_origin = k_new
        b_origin = b_new
#        print("##########################################################")
#        print("sum([(yi - (k_origin * xi + b_origin) * xi) * decayi for (xi, yi, decayi) in train_data]):", sum([((yi - (k_origin * xi + b_origin)) * xi) * decayi for (xi, yi, decayi) in train_data]))
#        print()
    print("a:", math.exp(b_origin))
    print("b:", k_origin)
    print("Loss:", sum([pow(yi - (k_origin * xi + b_origin), 2) * decayi for (xi, yi, decayi) in train_data]))
    return k_origin, b_origin    

def API4Grad_update(data, decay_coe):
    ln_data = [math.log(_data) for _data in data]
    len_data = len(ln_data)
    decay_list = [decay_coe[0] + float((decay_coe[1] - decay_coe[0])) / (len_data-1) * i for i in range(len_data)]
    return ln_data, decay_list

def fitting_flavor_use_exp(datas, data4avg, decay_coe, alpha, episode, predict_info, predict_days,avg_times, normal_times, extra_times):
    y_data, decay_list = API4Grad_update(datas, decay_coe)
    x_data = range(len(y_data))
    x_data = [_x+0.5 for _x in x_data]
#    print("x_data:", x_data)
#    input()
    k, b = grad_update(x_data, y_data, decay_list, 0.1, 1, alpha, episode)
    predict_len = len(x_data) + predict_info[-1][0] + 1
    y_model_out = model_out(k, b, [i + 0.5 for i in range(predict_len)])
    if (k <= 0):
        print("y_data:", [round((_y_data), 2) for _y_data in data4avg])
#        print("y_model_out:", [round(math.exp(_y_model_out), 2) for _y_model_out in y_model_out])
#        print("sum(y_data):", sum(y_data))
#        print("len(y_data):", len(y_data))
        print("b smaller than 0!!!!!!!!!!!!!!!!!!!:")
        if(predict_days <= 7):
            return int(round(sum(data4avg)*avg_times/len(data4avg) *predict_days))
        else:
            print("extra_times prediction")
            return int(round(sum(data4avg)*extra_times/len(data4avg) *predict_days))
#        return int(round(sum([(sum([math.exp(_y_data) for _y_data in y_data]) / len(y_data)-1) * avg_times * coe for (i, coe) in predict_info])))
    else:
        print("decay_list:", decay_list)
        print("y_data:", [(round(math.exp(_y_data), 2)) for _y_data in y_data])
        print("y_model_out:", [round(math.exp(_y_model_out), 2) for _y_model_out in y_model_out])
        print("predict_info:", predict_info)
        print("[(math.exp(y_model_out[i + len(x_data)])) * coe for (i, coe) in predict_info]:", [(math.exp(y_model_out[i + len(x_data)])) * coe for (i, coe) in predict_info])
#        print("math.exp(y_model_out[i + len(x_data)])+math.exp(y_model_out[i-1 + len(x_data)]))/2 :", [(math.exp(y_model_out[i + len(x_data)])+math.exp(y_model_out[i-1 + len(x_data)]))/2 for (i, coe) in predict_info])
#        print("math.exp(y_model_out[i + len(x_data)])+math.exp(y_model_out[i-1 + len(x_data)]))/2*coe :", [(math.exp(y_model_out[i + len(x_data)])+math.exp(y_model_out[i-1 + len(x_data)]))/2*coe for (i, coe) in predict_info])  
#        print("sum([(math.exp(y_model_out[i + len(x_data)])+math.exp(y_model_out[i-1 + len(x_data)]))/2 * coe for (i, coe) in predict_info]):", sum([(math.exp(y_model_out[i + len(x_data)])+math.exp(y_model_out[i-1 + len(x_data)]))/2 * coe for (i, coe) in predict_info]))
#        return int(round(sum([(math.exp(y_model_out[i + len(x_data)])+math.exp(y_model_out[i-1 + len(x_data)]))/2 * coe for (i, coe) in predict_info]))*normal_times)
        return int(round(sum([(math.exp(y_model_out[i + len(x_data)])) * coe for (i, coe) in predict_info]))*normal_times)
def predict_exp(datass, data4avg, decay, alpha, episode, predict_info, predict_days,avg_times, normal_times, extra_times):
    predict_result = []
    for index, datas in enumerate(datass):
        print("\n\n")
        print("flavor"+str(index+1)+"prediction:")
        predict_result.append(fitting_flavor_use_exp(datas[0],data4avg[index], decay, alpha, episode, predict_info,predict_days, avg_times, normal_times, extra_times))
        print("predict result:", predict_result[-1])
    return predict_result

def test():
    data = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23], [24, 25, 26], [27, 28, 29], [30, 31, 32], [33, 34, 35], [36, 37, 38], [39, 40, 41], [42, 43, 44]]
    print_two_dimension_data(data)
#    print()
#    smoothed_data = smooth4two_dimension_data(data)
#    print_two_dimension_data(smoothed_data)
#    print_two_dimension_data(data)
#    print("divide_week_data:", )
#    print(divide_week_data(data))
    instance_data, target_data = generate_instance(data)
    print_two_dimension_data(instance_data)
    print_two_dimension_data(target_data)
#    for target in target_data:
#        target[1] = secret_algorithm(target, instance_data)
#    print(KNN([[0, 3, 4], None], instance_data))
#    print_two_dimension_data(target_data)
    print(predict(target_data, instance_data))
    
if __name__ == "__main__":
    test()      