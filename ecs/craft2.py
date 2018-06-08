#import datetime
#
#start_date = datetime.date(2018, 4, 1)
#end_date = datetime.date(2018, 4, 9)
#print((end_date - start_date).days)
#
#def API4Time(train_end_day, predict_start_day, predict_days, unit):
#
#    
#    interval = (predict_start_day-train_end_day).days - 1
#    interval_point = interval / unit
#    interval_coe = float(interval % unit) / unit
#    
#    predict_start = interval 
#    predict_end = interval + predict_days
#    
#    _len = predict_end/unit - predict_start/unit + 1
##    if (predict_end%unit != 0):
##        _len += 1 
#    finally_coe = float(predict_end % unit) / unit
#    
#    predict_point = []
#    predict_coe = []
#    for index, point in enumerate(range(_len)):
#        predict_point.append(interval_point+index)
#        if (index == 0):
#            predict_coe.append(1-interval_coe)
#        elif(index == _len-1):
#            predict_coe.append(finally_coe)
#        else:
#            predict_coe.append(1)
#    return predict_point, predict_coe
#
#
#print(API4Time(start_date, end_date, 13, 7))
#print(zip(*[[1, 2], [0.6, 0.8]]))
#import time
#import math
#start_time = time.clock()
#for i in range(1000000):
#    pow(2.5456, 3)
#print(time.clock()-start_time)
#
#start_time = time.clock()
#for i in range(1000000):
#    math.pow(2.5456, 3)
#
#print(time.clock()-start_time)
#
#
#start_time = time.clock()
#for i in range(1000000):
#    abs(3.56)
#
#print(time.clock()-start_time)

#def model_out(k, b, x):
#    return [k * xi + b for xi in x]
#def grad_update(x_data, y_data, decay, k_origin, b_origin, alpha, episode):
#    train_data = zip(x_data, y_data, decay)
#    for i in range(episode):
#        k_new = k_origin + alpha * sum([((yi - (k_origin * xi + b_origin)) * xi) * decayi for (xi, yi, decayi) in train_data])
#        b_new = b_origin + alpha * sum([(yi - (k_origin * xi + b_origin)) * decayi for (xi, yi, decayi) in train_data])
#        k_origin = k_new
#        b_origin = b_new
##        print("##########################################################")
##        print("sum([(yi - (k_origin * xi + b_origin) * xi) * decayi for (xi, yi, decayi) in train_data]):", sum([((yi - (k_origin * xi + b_origin)) * xi) * decayi for (xi, yi, decayi) in train_data]))
##        print()
##        print("k_origin:", k_origin)
##        print("b_origin:", b_origin)
#    print("Loss:", sum([pow(yi - (k_origin * xi + b_origin), 2) * decayi for (xi, yi, decayi) in train_data]))
#    return k_origin, b_origin
#
#para = grad_update([0, 1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7, 80], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10], 1, 2, 0.002, 100000)
#print(para)
#print(model_out(para[0], para[1], [0, 1, 2, 3, 4, 5, 6]))
#        
#        
        
#import math
#def API4Grad_update(data, decay_coe):
#    ln_data = [math.log(_data) for _data in data]
#    len_data = len(ln_data)
#    decay_list = [decay_coe[0] + float((decay_coe[1] - decay_coe[0])) / (len_data - 1) * i for i in range(len_data)]
#    return ln_data, decay_list
#
#print(API4Grad_update([1, 2, 3, 4, 5, 6], [0, 1]))
        
import math
def grad_update(x_data, y_data, decay, k_origin, b_origin, alpha, episode):
    train_data = zip(x_data, y_data, decay)
    for i in range(episode):
        now_alpha = (alpha + 0.01 / (1+i)) 
        k_new = k_origin + now_alpha * sum([((yi - (k_origin * xi + b_origin)) * xi) * decayi for (xi, yi, decayi) in train_data])
        b_new = b_origin + now_alpha * sum([(yi - (k_origin * xi + b_origin)) * decayi for (xi, yi, decayi) in train_data])
        k_origin = k_new
        b_origin = b_new
#        print("##########################################################")
#        print("sum([(yi - (k_origin * xi + b_origin) * xi) * decayi for (xi, yi, decayi) in train_data]):", sum([((yi - (k_origin * xi + b_origin)) * xi) * decayi for (xi, yi, decayi) in train_data]))
#        print()
    print("a:", math.exp(b_origin))
    print("b:", k_origin)
    print("Loss:", sum([pow(yi - (k_origin * xi + b_origin), 2) * decayi for (xi, yi, decayi) in train_data]))
    return k_origin, b_origin


def stocGradAscent_2D(dataset, numIter=1000, slide_window=7):
    m = len(dataset)
    n = len(dataset[0])
    weights = []
    for k in range(m):
        print ('dataset[k]',dataset[k])
        data_series=[math.log(i) for i in dataset[k]]
        weight_k = [1, 0.1]
        for j in range(numIter):
            alpha = 0.01 / (1.0 + j) + 0.002
            error=0
            error_b=0
            for randIndex in range(n):
                h = weight_k[0] * randIndex + weight_k[1]
                left_bound=0.6
                right_bound=1.4
#                print("1.0/n*(left_bound+(right_bound-left_bound)*randIndex*1.0/n):", (left_bound+(right_bound-left_bound)*randIndex*1.0/(n-1)))
                error+= (data_series[randIndex] - h)* randIndex*1.0/n*(left_bound+(right_bound-left_bound)*randIndex*1.0/(n-1))
                error_b+=(data_series[randIndex] - h)/n*(left_bound+(right_bound-left_bound)*randIndex*1.0/(n-1))
            # print(error)
            weight_k[0] = weight_k[0] + alpha * error
            weight_k[1] = weight_k[1] + alpha * error_b
        weight_k[1] = math.exp(weight_k[1])
        weights.append(weight_k)
    return weights

print(grad_update([0, 1, 2], [math.log(4), math.log(4), math.log(16)], [0.6, 0.866, 1.13], 1, 0.1, 0.002, 100000))

print("\n\n\n")
#[math.log(4), math.log(4), math.log(16)]

print(stocGradAscent_2D([[4, 4, 16]], 100000))
        
        
print("math.e:", math.e)        
        