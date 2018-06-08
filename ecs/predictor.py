'''#//
#//                       _oo0oo_
#//                      o8888888o
#//                      88" . "88
#//                      (| -_- |)
#//                      0\  =  /0
#//                    ___/`---'\___
#//                  .' \|     |// '.
#//                 / \|||  :  |||// \
#//                / _||||| -:- |||||- \
#//               |   | \  -  /// |     |
#//               | \_|  ''\---/''  |_/ |
#//               \  .-\__  '-'  ___/-. /
#//             ___'. .'  /--.--\  `. .'___
#//          ."" '<  `.___\_<|>_/___.' >' "".
#//         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#//         \  \ `_.   \_ __\ /__ _/   .-` /  /
#//     =====`-.____`.___ \_____/___.-`___.-'=====
#//                       `=---='
#//
#//
#//     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#//              Buddha bless, Never bug
#//
#//
#//'''

#import _numpy as np
#import numpy as np
import datetime
import sys
import copy
import time
import random 
import secret_algorithm as sa
import _numpy as _np
random.seed(0)
handle_data_index = {"year":0, "month":1, "day":2, "day4week":3}
train_len_coe = 0.7
BATCH = 10
#RAW_DATA_SIZE = 500
TRAIN_DAY_SIZE = 80
TRAIN_DATA_SIZE = 40
target_index = {}
COE = 0.95
DAYS = 7
TIMES = 2.61
AFTER_TWO_TIMES = 2.61
NORMAL_TIMES = 1.06
import deploy_all_flavors
def cal_day4week(year, month, day):
        if(month == 1 or 2):
            c = (year-1)//100
            y = (year-1)%100
            m = month + 12
        else:    
            c = year//100
            y = year%100
            m = month
        day4week = (c//4) - 2 * c + y + (y//4) + (26*(m+1)//10) + day - 1
        while(day4week > 6 or day4week < 0):day4week = day4week%7
        return day4week
    
    
def get_raw_data(values):
        uuid = values[0]
        flavorName = int(values[1][6:])
        date, time = values[2].split()
        year, month, day = date.split('-')
        hour, minute, second = time.split(':')
        year = int(year)
        month = int(month)
        day = int(day)
        hour = int(hour)
        minute = int(minute)
        second = int(second)
        return uuid, flavorName, year, month, day, hour, minute, second

def deal_input(input_lines):
    print("input_lines:", input_lines)
    server_name = []
    limitation_count = int(input_lines[0])
#    print("limitation_count:", limitation_count)
    limitation = []
    for i in range(limitation_count):
        _name, _cpu, _memory, __ = input_lines[i+1].split(' ')
        limitation.append([int(_cpu), int(_memory)])
        server_name.append(_name)
    print("limitation:", limitation)
    flavor_count = int(input_lines[limitation_count + 2])
    print("flavor_count:", flavor_count)
    flavors = []
    cpus = []
    memorys = []
    for flavor_info in input_lines[limitation_count+3:limitation_count+3+flavor_count]:
        _flavor, _cpu, _memory = flavor_info.split(' ') 
        _flavor = int(_flavor[6:])
        _cpu = int(_cpu)
        _memory = int(_memory)
        flavors.append(_flavor)
        cpus.append(_cpu)
        memorys.append(_memory)
    print("flavors:", flavors)
    print("cpus:", cpus)
    print("memorys:", memorys)
    start_time = deal_time(input_lines[limitation_count+3+flavor_count+1])
    end_time = deal_time(input_lines[limitation_count+3+flavor_count+2])
    print("start_time:", start_time)
    print("end_time:", end_time)
    predict_days = (datetime.date(end_time[0], end_time[1], end_time[2]) - datetime.date(start_time[0], start_time[1], start_time[2])).days+1
    
#    predict_days = _lib.cal_time_sub(end_time, start_time)-1#-1 because the end_time is 00:00
#    print("predict_days:", predict_days)
    return server_name, limitation, flavor_count, flavors, cpus, memorys, predict_days, datetime.date(start_time[0], start_time[1], start_time[2])

def deal_time(time):

    date, time = time.split()
    year, month, day = date.split('-')
    hour, minute, second = time.split(':')#    lstm_param = lstm.LstmParam(mem_cell_ct=in, x_dim)
#    lstm_net = lstm.LstmNetwork(lstm_param)    
    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    second = int(second)
    _time = (year, month, day, hour, minute, second)
    return _time

def deal_raw_datas(datas):
    if (datas == None):
        return None
    raw_datas = []
    for index, item in enumerate(datas):
        values = item.split("\t")
        uuid, flavorName, year, month, day, hour, minute, second = get_raw_data(values)
        day4week = cal_day4week(year, month, day)
        raw_datas.append([uuid, flavorName, year, month, day, hour, minute, second, day4week])
    return raw_datas

def cal_max_flavor(flavors):
    _max_flavor = 0
    for flavor in flavors:
        if flavor > _max_flavor:
            _max_flavor = flavor
#    print("flavors:", len(flavors))
#    print("_max:", _max_flavor)
    return _max_flavor

def print_two_dimension_data(two_dimension_data):
    print("######################################")
    for i in range(len(two_dimension_data)):
        print(i, ": ", two_dimension_data[i])
    print("######################################")



def del_abnormal_J(datas, limitation, windows, fix):
    def slide_mean(data, windows):
        mean = []
        for index, _data in enumerate(data):
#            print("datas[:windows]:", data[:windows])
#            print("int(sum(datas[:windows])/len(datas[:windows])):", int(sum(data[:windows])/len(data[:windows])))
            if(index < windows):
                mean.append(int(sum(data[:windows])/len(data[:windows])))
            else:
                mean.append(int(float(sum(data[index-windows+1:index+1]))/len(data[index-windows+1:index+1])))
#            if(_data==10):
#                print("10 mean:", mean[-1])
        return mean

    def slide_std(data, mean, windows):
        n_variance = sum([pow(data[index] - mean[index], 2)  for index in range(windows)])
        initial_std = pow(n_variance/(windows), 0.5)
#        print("initial_std:", initial_std)
#        input()
        std = []
        for index, _data in enumerate(data):
            if(index < windows):
                std.append(initial_std)
            else:
                n_variance -= pow((data[index-windows]-mean[index-windows]), 2)
                n_variance += pow((data[index]-mean[index]), 2)
                _std = pow(n_variance/(windows), 0.5)
                std.append(_std)
#            if(_data==10):
#                print("10 std:", std[-1])
        return std
    
    
    datas = [list(_data) for _data in zip(*datas)]
    means = []
    stds = []
    for data in datas:
        means.append(slide_mean(data, windows))
    for index, data in enumerate(datas):
        stds.append(slide_std(data, means[index], windows))
    print("means:")
    print_two_dimension_data(means)
    print("stds:")
    print_two_dimension_data(stds)    
#    input()
    for i, data in enumerate(datas):
        for j, _data in enumerate(data):
            if(abs(_data-means[i][j])>3):
                up_limitation = means[i][j] + limitation * stds[i][j]
                if(_data > up_limitation):
                    datas[i][j] = means[i][j] + fix * stds[i][j]
                if(_data == 10):
                    print("10 mean:", means[i][j])
                    print("10 std:", stds[i][j])
                    print("10 up_limitation:", up_limitation)
                    print("10 fix:", datas[i][j])
                
    return zip(*datas)        
                
def del_abnormal(datas, limitation, windows, fix):
    datas = [list(_data) for _data in zip(*datas)]
    for data in datas:
        for index, _data in enumerate(data):
#            print("\n")
#            print("index:", index)
            if(index < windows):
                mean = float(sum(data[:windows])) / len(data[:windows])
                std = pow(sum([pow(_data - mean, 2) for _data in data[:windows]]) / len(data[:windows]), 0.5)


            else:
                if(abs(_data - data[index-1]) < 3):
                    continue
                mean = float(sum(data[index-windows+1:index+1])) / len(data[index-windows+1:index+1])
                std = pow(sum([pow(_data - mean, 2) for _data in data[index-windows+1:index+1]]) / len(data[index-windows+1:index+1]), 0.5)
#                print("mean:", mean)
#                print("std:", std)            
            up_limitation = mean + limitation * std
            down_limitation = mean - limitation * std    
#            print("up_limitation:", up_limitation)
#            print("down_limitation:", down_limitation)
#            input()
#            if(_data == 12):
#                print("mean:", mean)
#                print("std:", std)
#                print("up_limitation:", up_limitation)
#                input()
            if (_data > up_limitation):
#                print("origin)
                data[index] = mean + fix * std
#            elif (_data < down_limitation):
#                data[index] = mean - fix * std                  
                
            
#        mean = float(sum(data)) / len(data)
#        std = pow(sum([pow(_data - mean, 2) for _data in data]) / len(data), 0.5)
#        up_limitation = mean + limitation * std
#        down_limitation = mean - limitation * std
#        print("up_limitation:", up_limitation)
#        print("down_limitation:", down_limitation)
#        for index in range(len(data)):
#            if (data[index] > up_limitation):
#                data[index] = up_limitation
#            elif (data[index] < down_limitation):
#                data[index] = down_limitation
    return zip(*datas)



def deal_handleDatas(raw_datas, output_target, predict_days, answer_array):
    
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
    
    def deal_missing_value(datas):
        def fill_with_front(datas, row):
            col_count = len(datas[0])
            for _index in range(col_count):
                datas[row][_index] = datas[row-1][_index]
        
        def fill_with_back(datas, row):
            col_count = len(datas[0])
            for _index in range(col_count):
                datas[row][_index] = datas[row+1][_index]        
        
        def fill_with_average(datas, row):
            col_count = len(datas[0])
            for _index in range(col_count):
                
                datas[row][_index] = int(round(float(datas[row-1][_index] + datas[row+1][_index]) / 2))
        
        def judge_fill(row, rows, len_list):
            if row == 0:
                return 2
            if row == len_list - 1:
                return 0
            if row+1 not in rows and row-1 not in rows:
                return 1
            if row+1 not in rows:
                return 2
            if row-1 not in rows:
                return 0
        rows_missing = []
        for index, data in enumerate(datas):
                if sum(data) == 0:
                    rows_missing.append(index)
        func_list = [fill_with_front, fill_with_average, fill_with_back]
        print("row_missing:", rows_missing)
        for row in rows_missing:
            func_index = judge_fill(row, rows_missing, len(datas))
            print("row:", row, "func_index:", func_index)
            func_list[func_index](datas, row)
            rows_missing[rows_missing.index(row)]=-1
            print("rows_missing:", rows_missing)
    flavor_count = len(output_target)
    input_sizes = (datetime.date(raw_datas[-1][2], raw_datas[-1][3], raw_datas[-1][4]) - datetime.date(raw_datas[0][2], raw_datas[0][3], raw_datas[0][4])).days+1
#    print("input_sizes:", input_sizes)
    #deal the last line time as 23:59
#    print("predict_days:", predict_days)
    day_datas = [[0]*flavor_count for i in range(input_sizes)]
#    print("day_datas:", day_datas)
#    input()
#    print("deal, doned_datas:", doned_datas.shape)
#    index_dealed = 0
#    year4start =  raw_datas[0][2]
#    month4start = raw_datas[0][3]
#    day4start = raw_datas[0][4]
    _index = 0
#    input_index = {} # key flavor, value index:
    output_target.sort()
    for key in output_target:
        target_index[key] = _index
        _index += 1
#    input_index.
#    print("target_index:", target_index)
        
    for item in raw_datas:
##        print("item:", item)
        index_dealed = (datetime.date(item[2], item[3], item[4]) - datetime.date(raw_datas[0][2], raw_datas[0][3], raw_datas[0][4])).days
#        print("index_dealed:", index_dealed)
#        doned_datas[index_dealed][handle_data_index["year"]] = item[2]%2000#only need the last two number
#        doned_datas[index_dealed][handle_data_index["month"]] = item[3]
#        doned_datas[index_dealed][handle_data_index["day"]] = item[4]
#        doned_datas[index_dealed][handle_data_index["day4week"]] = item[8]       
#        print("item[1]-1:", item[1]-1)
        if(item[1] in output_target):
            day_datas[index_dealed][target_index[item[1]]] = day_datas[index_dealed][target_index[item[1]]]+1 

    print("day_datas:")
    print_two_dimension_data(zip(*day_datas))
#    input()
#    smoothed_data = day_datas
#    print("smoothed_data:", smoothed_data)
#    print("smoothed_data:", len(smoothed_data))
#    print("smoothed_data:", len(smoothed_data[0]))
#    input()
#    print_two_dimension_data(day_datas)
#    deal_missing_value(day_datas)
#    print("deal_missing_datas:")
#    print_two_dimension_data(day_datas)    
#    print_two_dimension_data(day_datas)
#    print("day_datas:", day_datas)
#    input()

#    input()
#    smoothed_data = sa.smooth4two_dimension_data(day_datas)
    
    
#    day_datas_np = _np.array(day_datas)
#    mean = day_datas_np.mean(axis=0)
#    std = day_datas_np.std(axis=0)
##    up_limit = mean + std * 2
##    down_limit = mean - std * 2
##    abnormal_data = _np.clip(day_datas_np, down_limit, up_limit)
##    print("up_limit:", up_limit)
##    print("down_limit:", down_limit)
#    day_datas_np -= mean
#    day_datas_np /= std
#    
#    print("day_datas_np:", day_datas_np)
#    input()
    
    smoothed_data = del_abnormal_J(day_datas, 2.1, 7, 0.65)
    print("smoothed_data:")
    print_two_dimension_data(zip(*smoothed_data))
#    input()
    week_data = sa.divide_week_data(smoothed_data, DAYS)
#    _week_data = sa.divide_week_data(day_datas)
    print("week_data:")
    print_two_dimension_data(week_data)
#    input()
#    print("week_data:", len(week_data))
#    print("week_data:", len(week_data[0]))
#    input()
    instance_data, target_data = sa.generate_instance(week_data)
#    print("instance_data:", instance_data)
#    print("instance_data:", len(instance_data))
#    print("instance_data:", len(instance_data[0]))
    print("target_data:", target_data)
#    input()
#    print("target_data:", len(target_data))
#    print("target_data:", len(target_data[0]))   
#    input()
    if(answer_array != None):
        import numpy as np
#        print("answer_array:", answer_array)
        answer_sizes = (datetime.date(answer_array[-1][2], answer_array[-1][3], answer_array[-1][4]) - datetime.date(answer_array[0][2], answer_array[0][3], answer_array[0][4])).days+1
        answer_day_datas = [[0]*flavor_count for i in range(answer_sizes)]
        for item in answer_array:
    ##        print("item:", item)
            index_dealed = (datetime.date(item[2], item[3], item[4]) - datetime.date(answer_array[0][2], answer_array[0][3], answer_array[0][4])).days
    #        print("index_dealed:", index_dealed)
    #        doned_datas[index_dealed][handle_data_index["year"]] = item[2]%2000#only need the last two number
    #        doned_datas[index_dealed][handle_data_index["month"]] = item[3]
    #        doned_datas[index_dealed][handle_data_index["day"]] = item[4]
    #        doned_datas[index_dealed][handle_data_index["day4week"]] = item[8]       
    #        print("item[1]-1:", item[1]-1)
            if(item[1] in output_target):
                answer_day_datas[index_dealed][target_index[item[1]]] = answer_day_datas[index_dealed][target_index[item[1]]]+1    
#                print("answer_day_datas:", answer_day_datas)
        
#        input()
        answer_datas = np.array(answer_day_datas)
        answer_datas = np.sum(answer_datas, axis = 0)
        answer_array = list(answer_datas)
        print("answer_array:", answer_array)
    return instance_data, target_data, answer_array, datetime.date(raw_datas[-1][2], raw_datas[-1][3], raw_datas[-1][4]), zip(*smoothed_data)

def genenrate_int(low_limit, high_limit):
    assert (type(low_limit) == int and type(high_limit) == int and low_limit < high_limit)
    return int(np.random.randint(low_limit, high_limit))

def gene_train_test_start_end(row, train_len, test_len):
    start_ends = []
    for i in range(BATCH):
        train_start = genenrate_int(0, row-train_len-test_len)
        train_end = train_start+train_len
        test_start = train_end
        test_end = test_start + test_len
        start_ends.append([train_start, train_end, test_start, test_end])
    return start_ends


def gene_interval(row, pred):
    
    train_time_step_max = 1
    while(train_time_step_max + pred < row*train_len_coe and train_time_step_max < 6):
        train_time_step_max += 1
#    assert train_time_step_max > 0
#    print("train_time_step_max:", train_time_step_max)
    test_time_step_max = pred
#    assert(row > train_time_step_max+test_time_step_max)
    train_len = train_time_step_max
    test_len = test_time_step_max
    return gene_train_test_start_end(row, train_len, test_len)

  

def evaluate_answer(result, answer):
    import numpy as np
    result = copy.copy(result)
    for index, _result in enumerate(result):
        result[index] = int(round(_result))
    for index, _answer in enumerate(answer):
        answer[index] = int(round(_answer))
#    result = [result]
#    answer = [answer]
    print("result:", result)
    print("answer:", answer)
    result = np.array(result)
    answer = np.array(answer)
#    def sub(a0, a1):
#        f = lambda x0, x1:x0-x1
#        f
    numerator = np.sqrt(np.sum((result-answer)**2)/result.shape[0])
    denominator = np.sqrt(np.sum((result)**2)/result.shape[0]) + np.sqrt(np.sum((answer)**2)/answer.shape[0])
    print("###################################################################################################")
    print("score you achieve:", 100 * (1-numerator/denominator))
    print("###################################################################################################")  
    return 100 * (1-numerator/denominator)


def API4Time(train_end_day, predict_start_day, predict_days, unit):

    
    interval = (predict_start_day-train_end_day).days - 1
    interval_point = interval / unit
    interval_coe = float(interval % unit) / unit
    
    predict_start = interval 
    predict_end = interval + predict_days
    
    _len = predict_end/unit - predict_start/unit + 1
#    if (predict_end%unit != 0):
#        _len += 1 
    finally_coe = float(predict_end % unit) / unit
    print("predict_start_day:", predict_start_day)
    print("train_end_day:", train_end_day)
    print("interval:", interval)
    print("interval_point:", interval_point)
    print("interval_coe:", interval_coe)
    print("predict_start:", predict_start)
    print("predict_end:", predict_end)
    print("_len:", _len)
    print("finally_coe:", finally_coe)
    predict_point = []
    predict_coe = []
    for i in range(interval_point):
        predict_point.append(i)
        predict_coe.append(1)
    for index, point in enumerate(range(_len)):
        predict_point.append(interval_point+index)
#        if (index == 0):
#            predict_coe.append(1-interval_coe)
        if(index == _len-1):
            predict_coe.append(finally_coe)
        else:
            predict_coe.append(1)
    return [predict_point, predict_coe]

def predict_vm(ecs_lines, input_lines, answer_array):
    server_name, server_limitations, flavor_count, flavor_list, flavor_cpu, flavor_mem, predict_days, predict_start_days = deal_input(input_lines)# cpu, memory, disk, flavor_count, flavors, cpus, memorys, goal, predict_days
#    print("input_info:", input_info)
#    input()
#    print("input_info:", input_info) #(56, 128, 1200, 5, [1, 2, 3, 4, 5], [1, 1, 1, 2, 2], [1024, 2048, 4096, 2048, 4096], 'CPU\n', 7)
    
    
    
#    raw_datas = deal_raw_datas(ecs_lines[-RAW_DATA_SIZE:])#uuid, flavorName, year, month, day, hour, minute, second  
#    print("raw_datas:", raw_datas) # ['56498dc2-a280', 2, 2015, 2, 19, 19, 5, 38, 4]
    raw_datas = deal_raw_datas(ecs_lines)
#    print("raw_datas:", raw_datas)
#    input()
    answer_datas = deal_raw_datas(answer_array)
    print("first answer_datas:", answer_datas)
    train_data, test_datas, answer_datas, train_end_days, data4avg = deal_handleDatas(raw_datas, flavor_list, predict_days, answer_datas)
    print("second answer_datas:", answer_datas)
#    predict_result = sa.predict_mean(test_datas, COE, predict_days)
    predict_info = API4Time(train_end_days, predict_start_days, predict_days, DAYS)
    print("predict_info:", predict_info)
#    input()
    predict_result = sa.predict_exp(test_datas, data4avg, [0.4, 1.6], 0.002, 100000, zip(*predict_info),predict_days, TIMES, NORMAL_TIMES, AFTER_TWO_TIMES)
    predict_result = [int(round(_pre * COE)) for _pre in predict_result]
#    predict_result = sa.predict_exp(test_datas, [0, 0.5], [0.4, 1.6], 10000, 1000, 3, zip(*predict_info))
    predict_result_copy = copy.copy(predict_result)
    print("predict_result:", predict_result)
#    input()
#    print("train_data:", train_data)
#    data = deal_handleDatas(raw_datas, 14)   
#    train_data = train_data[-TRAIN_DATA_SIZE:] if len(train_data) > TRAIN_DATA_SIZE else train_data
    
#    print("train_data:", train_data)
#    for index in range(15):
#        if(index not in input_info[4]):
#            del(train_data[:][0][index])
#    print("train_data:", train_data)
    
#    print("train_data:", len(train_data))
#    print("test_data:", len(test_datas))
#    print("len(train_data[0][0]):", train_data)
#    predict_result = train(len(train_data[0][0][0]),  len(train_data), train_data, test_datas, input_info[-1], start_time)
    
    
    if(answer_datas != None):
        evaluate_answer(predict_result, answer_datas)    
    
    
    
    #boxing data preparing#####################################################################
    
#    predict_result = predict_result.content[0]
    print("predict_result:", predict_result)
    predict_dict={}
    output_sorted = flavor_list
    output_sorted.sort()
    print("output_sorted:", output_sorted)
    for index, key in enumerate(output_sorted):
#        print("index:", index)
#        print("key:", key)
#        print("int(round(predict_result[index])):", int(round(predict_result[index])))
        predict_dict["flavor"+str(key)] = int(round(predict_result[index]))
#    print("predict_dict:", predict_dict)
    flavor_prediction = predict_dict
    flavor_prediction_output = copy.deepcopy(flavor_prediction)
#    print(predict_dict)
#    return result
    flavor_predict_names = ["flavor" + str(ele) for ele in output_sorted]
    flavor_predict_names.sort()
    flavor_type = {}
    flavor_info = zip(flavor_cpu, flavor_mem)
    for index, flavor in enumerate(output_sorted):
        flavor_type["flavor"+str(flavor)] = list(flavor_info[index])
    
    server_cfg = server_limitations
    print("flavor_predict_names:", flavor_predict_names)
    print("flavor_prediction:", flavor_prediction)
    print("flavor_type;", flavor_type)
    print("server_cfg:", server_cfg)

    
    
    
    
    res=deploy_all_flavors.deploy_flavors(flavor_predict_names,flavor_prediction,flavor_type,server_cfg, server_name)
    #res=deploy_all_flavors.deploy_flavors(flavor_predict_names,flavor_prediction,flavor_type,server_cfg,optimize_dim)
    print("res:", res)
    return res
    input()
#    print("check:", check)
    print("flavor_prediction_output:", flavor_prediction_output)
    flavor_prediction_check_list = []
    predict_result = []
    flavor_prediction_output = {}
    for _check in check:
        flavor_prediction_check_list.extend(_check)
    for output in output_sorted:
        flavor_prediction_output['flavor' + str(output)] = flavor_prediction_check_list.count(output)
        predict_result.append(flavor_prediction_check_list.count(output))
    print("flavor_prediction_output_cheat:", flavor_prediction_output)

    print("[Data Add Perventage]:", float(sum(predict_result)) / float(sum(predict_result_copy)))
    if(answer_datas != None):
        print("[finally score]:", score * evaluate_answer(predict_result, answer_datas) / 100) 


    #output#####################################################################
    first_line = str(sum(flavor_prediction_output.values())) + '\n'
    flavor_prediction_output_key = flavor_prediction_output.keys()
#    flavor_prediction_output_key.sort()
    for key in flavor_prediction_output_key:
        first_line += key 
        first_line += ' '
        first_line += str(flavor_prediction_output[key])
        first_line += '\n'
#    print("first_line:", first_line)
#    print(first_line + '\n' + res)
    output = first_line  + '\n' + res
    print("output:" ,output)
    return output
