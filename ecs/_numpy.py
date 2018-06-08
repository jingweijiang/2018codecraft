import math
import copy
import random
random.seed(0)
eps = 2.220446049250313e-16
class array():
    def __init__(self, _object):
#        self.content = copy.deepcopy(_object)
        self.content = _object
#        print("self.content:", self.content) 
        # assert all ele is array
        if(isinstance(self.content[0], array)):
            for index, ele in enumerate(self.content):
                assert(isinstance(ele, array)), "assert all ele is array"
                self.content[index] = ele.content
#        print("self.content:", self.content)        
        self.origin_shape = self._shape()
        self.shape = self.origin_shape
        self.flatten_array()
        self.content = map(float, self.content)
#        print("self.content:", self.content)
#        print("self.shape:", self.shape)
#        print("self.origin_shape:", self.origin_shape)
        self.reshape_changed_origin(self.origin_shape)
#        print("self.shape:", self.shape)

#        print("self.shape:", self.shape)
        self.size = reduce(lambda x,y:x*y, self.shape)

    def __repr__(self):
        return "arr: shape: " + str(self.shape) + " mat: " + str(self.content)
    
#    def __sub__(self, other):
#        
#        if(isinstance(other, array)):
#            assert self.shape == other.shape, "shape must be matching"
#            a0 = flatten_array(self)
#            a1 = flatten_array(other)
##            print("a0:", a0)
##            print("a1:", a1)
#            result = [(a0.content[i] - a1.content[i]) for i in range((a0.size))]
#            result = array(result)
#            return result.reshape(self.shape)
##            return result
#        elif(isinstance(other, int) or isinstance(other, float)):
#            a0 = flatten_array(self)
#            result = [(a0.content[i] - other) for i in range((a0.size))]
#            result = array(result)
#            return result.reshape(self.shape)
##            return result
    
    def __sub__(self, other):
#        print("self:", self.shape)
#        print("other:", other.shape)
        if(isinstance(other, array)):
#            print("len(other.shape):", len(other.shape))
#            print("self.shape[1]:", self.shape[1])
#            print("other.shape[0]:", other.shape[0])
            if(self.shape == other.shape): # for the strongest condition
                a0 = flatten_array(self)
                a1 = flatten_array(other)
#                print("a0:", a0)
#                print("a1:", a1)
                result = [(a0.content[i] - a1.content[i]) for i in range((a0.size))]
                result = array(result)
                return result.reshape(self.shape)
            
            if(len(other.shape) == 1 and self.shape[1] == other.shape[0]):
                broadcastimes = self.shape[0] 
#                print("broadcastimes:", broadcastimes)
                _other = array([other.content]*broadcastimes)
#                print("_other:", _other)
                return self-_other
            
            assert self.shape[0] == other.shape[0] or self.shape[1] == other.shape[1], "shape must be matching"
            if(self.shape == other.shape):
                a0 = flatten_array(self)
                a1 = flatten_array(other)
#                print("a0:", a0)
#                print("a1:", a1)
                result = [(a0.content[i] - a1.content[i]) for i in range((a0.size))]
                result = array(result)
                return result.reshape(self.shape)
#                return result
            elif(self.shape[0] == other.shape[0]):
                raise NotImplementedError()
            elif(self.shape[1] == other.shape[1]):
                broadcastimes = self.shape[0]/other.shape[0] if self.shape[0]>other.shape[0] else other.shape[0]/self.shape[0] 
#                print("broadcastimes:", broadcastimes)
                if self.shape[0]>other.shape[0]:
                    broadcastimes = self.shape[0]/other.shape[0]
                    _other = array(other.content*broadcastimes)
                    return self-_other
                else:
                    broadcastimes = other.shape[0]/self.shape[0] 
                    _self = array(self.content*broadcastimes)
                    return _self-other
                raise NotImplementedError()    

        elif(isinstance(other, int) or isinstance(other, float)):
            a0 = flatten_array(self)
            result = [(a0.content[i] - other) for i in range((a0.size))]
            result = array(result)
            return result.reshape(self.shape)
            return result


        
    def __add__(self, other):
#        print("self:", self.shape)
#        print("other:", other.shape)
        if(isinstance(other, array)):
            assert self.shape[0] == other.shape[0] or self.shape[1] == other.shape[1], "shape must be matching"
            if(self.shape == other.shape):
                a0 = flatten_array(self)
                a1 = flatten_array(other)
#                print("a0:", a0)
#                print("a1:", a1)
                result = [(a0.content[i] + a1.content[i]) for i in range((a0.size))]
                result = array(result)
                return result.reshape(self.shape)
#                return result
            elif(self.shape[0] == other.shape[0]):
                raise NotImplementedError()
            elif(self.shape[1] == other.shape[1]):
                broadcastimes = self.shape[0]/other.shape[0] if self.shape[0]>other.shape[0] else other.shape[0]/self.shape[0] 
#                print("broadcastimes:", broadcastimes)
                if self.shape[0]>other.shape[0]:
                    broadcastimes = self.shape[0]/other.shape[0]
                    _other = array(other.content*broadcastimes)
                    return self+_other
                else:
                    broadcastimes = other.shape[0]/self.shape[0] 
                    _self = array(self.content*broadcastimes)
                    return _self+other
                raise NotImplementedError()
        elif(isinstance(other, int) or isinstance(other, float)):
            a0 = flatten_array(self)
            result = [(a0.content[i] + other) for i in range((a0.size))]
            result = array(result)
            return result.reshape(self.shape)
            return result

    def __mul__(self, other):
        
        if(isinstance(other, array) or isinstance(other, list)):
            if(isinstance(other, array)):
                assert self.shape == other.shape, "shape must be matching"
            if(isinstance(other, list)):
                assert self.shape[0] == len(other), "shape must be matching"
            a0 = flatten_array(self)
            a1 = flatten_array(other)
#            print("a0:", a0)
#            print("a1:", a1)
            result = [(a0.content[i] * a1.content[i]) for i in range((a0.size))]
            result = array(result)
            return result.reshape(self.shape)
#            return result
        elif(isinstance(other, int) or isinstance(other, float)):
            a0 = flatten_array(self)
            result = [(a0.content[i] * other) for i in range((a0.size))]
            result = array(result)
            return result.reshape(self.shape)
#            return result
        
    def __div__(self, other):
        
        if(isinstance(other, array)):
            assert self.shape == other.shape, "shape must be matching"
            a0 = flatten_array(self)
            a1 = flatten_array(other)
#            print("a0:", a0)
#            print("a1:", a1)
#            result = [(a0.content[i] / a1.content[i]) for i in range((a0.size))]
            result = []
#            for i in range((a0.size)):
#                assert  a1.content[i]>eps, "divisor must be greater than eps"
            result = [(a0.content[i] / (a1.content[i] if abs(a1.content[i]) > 0 else ( eps if a1.content[i] >= 0 else -eps))) for i in range((a0.size))]
#                print("i:", i)
#                print("result:", result)
            result = array(result)
            return result.reshape(self.shape)
#            return result
        elif(isinstance(other, int) or isinstance(other, float)):
            while (abs(other) <= eps):
		if(other > 0):
			other = eps
		elif(other < 0):
			other = -eps
            other = float(other)
            a0 = flatten_array(self)
            result = [(a0.content[i] / other) for i in range((a0.size))]
            result = array(result)
            return result.reshape(self.shape)
#            return result

#        assert len(a0) == len(a1)
#        result = [(a0[i] - a1[i]) for i in range(len(a0))]
#        result = array(result)
#        result.reshape(self.shape)
#        return result
    
    def __pow__(self, value):
        a0 = flatten_array(self)
        result = [(a0.content[i] ** value) for i in range((a0.size))]
        result = array(result)
        return result.reshape(self.shape)
#        return result        


    def pow__opti(self, value):
        a0 = flatten_array(self)
        value_l = [value]*a0.size
        
        result = map(pow, a0.content, value_l)
        result = array(result)
        return result.reshape(self.shape)
#        return return result   

    
    def __getitem__(self, index):
#        print("index:", index)
        result = self.content
        if (isinstance(index, tuple)):
            assert(len(index) <= len(self.shape)), "slices dimension must match the shape dimension."
            index = list(index)
            result = result[index[0]]
#            print("index[0]:", index[0])
#            print("result[index[0]]:", result)
            if(isinstance(index[0], slice)):
                return array([ele[index[1]] for ele in result])
            elif (isinstance(index[1], int)):
                return result[index[1]]
            return array(result[index[1]])
#                result = [result]
#            del(index[0])
#            while(len(index)>0):
#                
##                print("index[0]:", index[0])
#                
#                result = [ele[index[0]] for ele in result]
##                print("result[index[0]]:", result)
#                del(index[0])
            
            
#            if(isinstance(index[0]))
            
            
            
#            return array(result)
        elif(isinstance(index, slice)):
            return array(result[index])
    
    def __setitem__(self, index, value):
#        print("self.content:", self.content)
#        print("index:", index)
#        print("value:", value)
#        if (isinstance(index, tuple)):
#            assert(len(index) == len(self.shape)), "slices dimension must match the shape dimension."
#            index = list(index)
#            result = self.content
#            while(len(index)>1):
#                result = result[index[0]]
#                del(index[0])
##            if (isinstance(result, list)):
##                assert(len(value) == len(result)), "fulfill value must equal "
##                for index, ele in enumerate(result):
#            print("index[0]:", index[0])
#            print("result:", result)
##            for ele in result:
##                print("ele:", ele)
##                ele[index[0]] = value
##                print("ele:", ele)
##            print("result:", result)
##            map(lambda x:x=value, result)
#            print(result[index[0]])
#            result = result[index[0]]
#            for i in range(len(result)):
#                result[i] = value
#            print(result)
#            print(self.content)
#        elif(isinstance(index, slice)):
#            assert(len(self.shape) == 1), "slices dimension must match the shape dimension."
#            self.content[index] = value
#        index = list(index)
#        
        index_ele = array(range(self.size)) # index_ele represent the array of index to be changed
#        print("index_ele:", index_ele)
        index_ele.reshape_changed_origin(self.shape)
        index_ele.content = index_ele.content[index[0]]
        if(isinstance(index[0], slice)):
            index_ele.content = [ele[index[1]] for ele in index_ele.content]
        else:
            index_ele.content = index_ele.content[index[1]]
#        print("index_ele:", index_ele)
#        print("index_ele.content:", index_ele.content)
#        del(index[0])
#        print("index_ele:", index_ele)
#        while(len(index)>0):
#            index_ele.content = [ele[index[0]] for ele in index_ele.content]
#            print("index_ele:", index_ele.content)
#            del(index[0])
#        print("index_ele:", index_ele.content)
#        index_ele = flatten_array(index_ele)
        content_flatten = flatten_array(self.content).content
#        print("content_flatten:", content_flatten)
        if(isinstance(index_ele.content, float)):
#            print("int(index_ele.content):", int(index_ele.content))
            content_flatten[int(index_ele.content)] = value
        elif(isinstance(index_ele.content, list)):
#            print("index_ele.content:", index_ele.content)
            index_ele = flatten_array(index_ele)
#            print("index_ele.content:", index_ele.content)            
            for i in index_ele.content:
                content_flatten[int(i)] = value
#                
#        print("content_flatten:", content_flatten)
#        _temp_array = reshape(content_flatten, self.shape)
        _temp_array = array(content_flatten).reshape(self.shape)
#        print("_temp_array:", _temp_array)
        self.content = _temp_array.content
        self.update_T()
#        assert(len(index) == len(self.shape)), "slices dimension must match the shape dimension."
#        result = copy.deepcopy(self.content)
#        index = list(index)
#        result = result[index[0]]
##            print("index[0]:", index[0])
##            print("result[index[0]]:", result)
#        if(isinstance(index[0], slice)):
#            [ele[index[1]] for ele in result]
#        return array(result[index[1]])
        
    def __gt__(self, value):
        _self = flatten_array(self)
        if(isinstance(value, int) or isinstance(value, float)):
            result = array([ele > value for ele in _self.content])
            return result.reshape(self.shape)
#            return result
        elif(isinstance(value, array)):
#            print([ele0 > ele1 for (ele0, ele1) in zip(_self.content, flatten_array(value).content)])
            result = array([ele0 > ele1 for (ele0, ele1) in zip(_self.content, flatten_array(value).content)])
            return result.reshape(self.shape)
#            return result
#    def _shape(self):
#        _object = copy.deepcopy(self.content)
#        _shape = []
#        while (isinstance(_object, list)):
#            _shape.append(len(_object))
#            _object = _object[0]
#        return _shape

    def _shape(self):
        _object = self.content
        _shape = []
        while (isinstance(_object, list)):
            _shape.append(len(_object))
            _object = _object[0]
        return _shape

    def update_T(self):
        if (len(self.shape) == 1):
            self.T = self.content
        else:
            self.T = [list(_con) for _con in zip(*self.content)]
    def _T(self):
        return array(self.T)
    
    def update_shape(self):
        self.shape = self._shape()
    
    
    def deal_args_shape(self, args): # return [row, col]
        _len = len(args)
        if _len > 1:
            return list(args)
        else:
            return list(args[0])
    
    def reshape(self, *args):  # return a reshaped array without change the origin array
        def _reshape(array, sub_count):
            _array = array
            temp = zip(*[iter(_array)]* sub_count)
            result = []
            for _temp in temp:
                result.append(list(_temp))
            return result
        
#        print("*args:", args)
        shape = self.deal_args_shape(args)
#        print(shape)
#        print("result")
#        assert len(shape) <= 3
#        if (len(shape) > 1):
        _flatten_array = flatten_array(self)
        if(len(shape) == 1):
            return _flatten_array
        _array = _flatten_array.content
#            print("in reshape, _array:", _array)
#            print("self.shape:", self.shape)
        shape_list = list(shape)
        total_size = 1
        for ele in shape_list:
            total_size *= ele
#            print(self.content)
#            print(total_size)
#        print("len(self.content):", len(_array))
#        print("total_size:", total_size)
        assert len(_array) == total_size
        total_size /= shape_list[0]
        result = _reshape(_array, total_size)
#            print("result:", result)
        del(shape_list[0])
#            print("shape_list:", shape_list)
#            print("len(shape_list):", len(shape_list))
        while(len(shape_list) > 1):
            total_size /= shape_list[0]
            for index, _result in enumerate(result):
                result[index] = _reshape(_result,  total_size)
            del(shape_list[0])
        return(array(result))

            
    def reshape_changed_origin(self, shape):
        def _reshape(array, sub_count):
            _array = array
            temp = zip(*[iter(_array)]* sub_count)
            result = []
            for _temp in temp:
                result.append(list(_temp))
            return result
        
        assert len(shape) <= 3
        if (len(shape) > 1):
            _array = self.content
#            print("in reshape, _array:", _array)
#            print("self.shape:", self.shape)
            shape_list = list(shape)
            total_size = 1
            for ele in shape_list:
                total_size *= ele
#            print(self.content)
#            print(total_size)
            assert len(self.content) == total_size
            total_size /= shape_list[0]
            result = _reshape(_array, total_size)
#            print("result:", result)
            del(shape_list[0])
#            print("shape_list:", shape_list)
#            print("len(shape_list):", len(shape_list))
            while(len(shape_list) > 1):
                total_size /= shape_list[0]
                for index, _result in enumerate(result):
                    result[index] = _reshape(_result,  total_size)
                del(shape_list[0])
            self.content = result
            self.update_shape()
            self.update_T()
            
    def flatten_array(self): ##dont save the origin array, change with flatten_array
        def extend_iter(array):
            assert type(array) == list
            iter_list = []
            flatten_array = []
            while(isinstance(array, list)):
                _temp = array.__iter__()
                array = _temp.next()
                iter_list.append(_temp)
            flatten_array.append(array)
            return iter_list, flatten_array
    
        def write_result(iter_list):
            assert len(iter_list) > 0
            result_list = []
            while(True):
                try:
                    result_list.append(iter_list[-1].next())
                except StopIteration:
    #            del(iter_list[-1])
                    break
            return result_list
    
        def del_iter_list(iter_list):
    #    array = 0
            assert len(iter_list) > 0
            while(len(iter_list) > 0):
                try:
                    array = iter_list[-1].next()
                except StopIteration:
                    del(iter_list[-1])
                else:
                    return array
            return None
        if(len(self.shape) != 1):
            array = self.content
        #        print("array:", array)
            iter_list, flatten_result = extend_iter(array)
            flatten_result.extend(write_result(iter_list))
            array = del_iter_list(iter_list)
            while(array != None):
                
                _iter_list, _flatten_result = extend_iter(array)
                iter_list.extend(_iter_list)
                flatten_result.extend(_flatten_result)
                flatten_result.extend(write_result(iter_list))
                array = del_iter_list(iter_list)
        #        print("flatten_result:", flatten_result)
            self.content = flatten_result 
            self.update_shape()
          
    def mean(self, axis = None):
        a = self
        assert len(a.shape) <= 2
    #    shape_list = list(a.shape)
    #    print("shape_list:", shape_list)
        if(axis == None):
            _a = flatten_array(a)
            return reduce(lambda x, y:x+y, _a.content)/_a.size
        else:
            _a = a
            if axis == 0:  # this axis will diminish
                mat = zip(*_a.content)
    #            print(mat)
                return array([reduce(lambda x,y:x+y, _mat)/ a.shape[0] for _mat in mat]) 
            if axis == 1:
                assert len(a.shape) >= 2, "axis == 1, but the mat dimension only 1"
                return array([reduce(lambda x,y:x+y, _mat)/ a.shape[1] for _mat in _a.content])  
    def std(self, axis = None):
        a = self
#        print("a:", a)
#        print("a.mean(axis):", a.mean(axis))
        a -= a.mean(axis)
#        print("a:", a)
        a = a**2
#        print("a:", a)
        a = sum(a, axis=0)
#        print("a:", a)
        a /= self.shape[0]
#        print("a:", a)
        return sqrt(a)
#        assert axis == 0, "now only imple the axis = 0
        
#class zeros(array):
#    def __init__(self, shape):
#        self.size = reduce(lambda x,y:x*y, self.shape)
#        self.content = []
#        for i in range(self.size):
#            self.content.append(0)


def zeros(shape):
    size = reduce(lambda x,y:x*y, shape)
    content = []
    for i in range(size):
        content.append(0)
    return array(content).reshape(shape)
    
def ones(shape):
    size = reduce(lambda x,y:x*y, shape)
    content = []
    for i in range(size):
        content.append(1)
    return array(content).reshape(shape)   
    
    
    
def flatten_array(_array):#without change the origin content, can put array and list
    def extend_iter(array):
        assert type(array) == list
        iter_list = []
        flatten_array = []
        while(isinstance(array, list)):
            _temp = array.__iter__()
            array = _temp.next()
            iter_list.append(_temp)
        flatten_array.append(array)
        return iter_list, flatten_array

    def write_result(iter_list):
        assert len(iter_list) > 0
        result_list = []
        while(True):
            try:
                result_list.append(iter_list[-1].next())
            except StopIteration:
#            del(iter_list[-1])
                break
        return result_list

    def del_iter_list(iter_list):
#    array = 0
        assert len(iter_list) > 0
        while(len(iter_list) > 0):
            try:
                array = iter_list[-1].next()
            except StopIteration:
                del(iter_list[-1])
            else:
                return array
        return None
#    __array = copy.copy(_array)
    __array = _array
    if(isinstance(__array, array)):
        __array = __array.content
    iter_list, flatten_result = extend_iter(__array)
    flatten_result.extend(write_result(iter_list))
    __array = del_iter_list(iter_list)
    while(__array != None):
        
        _iter_list, _flatten_result = extend_iter(__array)
        iter_list.extend(_iter_list)
        flatten_result.extend(_flatten_result)
        flatten_result.extend(write_result(iter_list))
        __array = del_iter_list(iter_list)
    
#    print("flatten_result:", flatten_result)
    return array(flatten_result)

def flatten_array2D(_array):
    
    if(isinstance(_array, array)):
        __array = copy.deepcopy(_array.content)
    else:
        __array = copy.deepcopy(_array)
    flatten_result = []
    for _a in __array:
        flatten_result.extend(_a)
    
#    print("flatten_result:", flatten_result)
    return array(flatten_result)

#def reshape(a, shape):
#    def _reshape(array, sub_count):
#        _array = copy.deepcopy(array)
#        temp = zip(*[iter(_array)]* sub_count)
#        result = []
#        for _temp in temp:
#            result.append(list(_temp))
#        return result
#    
#    assert len(shape) <= 3
#    assert len(shape) != 1, "if you want to flatten the array, please use the flatten_array function"
#    if type(a) == list:
#        a = array(a)
#    
#    _array = flatten_array(a)
##    print("flatten_array:", _array)
#    shape_list = list(shape)
#    total_size = 1
#    for ele in shape_list:
#        total_size *= ele
#    assert len(_array.content) == total_size, "shape is not suitable to array"
#    total_size /= shape_list[0]
#    result = _reshape(_array.content, total_size)
#    del(shape_list[0])
#
#    while(len(shape_list) != 1):
#        total_size /= shape_list[0]
#        for index, _result in enumerate(result):
#            result[index] = _reshape(_result,  total_size)
#        del(shape_list[0])
#
#    return array(result)

def power(x1, x2):
    _power = lambda x, exp:x**exp
    _result = [_power(_x1, x2) for _x1 in flatten_array(x1.content).content]
    _temp_array = array(_result)
    return _temp_array.reshape(x1.shape)
#    return _temp_array

def sum(a, axis=None):
    assert len(a.shape) <= 2
#    shape_list = list(a.shape)
#    print("shape_list:", shape_list)
    if(axis == None):
        _a = flatten_array(a)
        return reduce(lambda x, y:x+y, _a.content)
    else:
        _a = a
        if axis == 0:
            mat = zip(*_a.content)
#            print(mat)
            return array([reduce(lambda x,y:x+y, _mat) for _mat in mat])
        if axis == 1:
            assert len(a.shape) >= 2, "axis == 1, but the mat dimension only 1"
            return array([reduce(lambda x,y:x+y, _mat) for _mat in _a.content])
        
def shape(array):
    if type(array) != list:
        _object = array.content
    else:
        _object = array
    _shape = []
    while (isinstance(_object, list)):
        _shape.append(len(_object))
        _object = _object[0]
    return _shape

def mean(a, axis=None):
    assert len(a.shape) <= 2
#    shape_list = list(a.shape)
#    print("shape_list:", shape_list)
    if(axis == None):
        _a = flatten_array(a)
        return reduce(lambda x, y:x+y, _a.content)/_a.size
    else:
        _a = a
        if axis == 0:  # this axis will diminish
            mat = zip(*_a.content)
#            print(mat)
            return array([reduce(lambda x,y:x+y, _mat)/ a.shape[0] for _mat in mat]) 
        if axis == 1:
            assert len(a.shape) >= 2, "axis == 1, but the mat dimension only 1"
            return array([reduce(lambda x,y:x+y, _mat)/ a.shape[1] for _mat in _a.content])

def sqrt(a):
    origin_shape = a.shape
    _a = flatten_array(a)
#    print(_a)
    for i in range(_a.size):
        _a.content[i] = _a.content[i] ** (0.5)
    return _a.reshape(origin_shape)

def dot(a, b):
    A = a.content
    B = b.content
    
    assert len(A[0]) == len(B),"dimension A:{}, dimension B:{}".format(a.shape, b.shape)
#    print([[reduce(lambda _a, _b: _a+_b, (a * b for a, b in zip(a, b))) for b in zip(*B)] for a in A])
    return array([[reduce(lambda _a, _b: _a+_b, (a * b for a, b in zip(a, b))) for b in zip(*B)] for a in A])

def array_split(ary, indices_or_sections):
    _ary = ary
    total_len = _ary.shape[0]
    _ele_count = int(round(float(total_len)/indices_or_sections))
#    print("_ele_count:", _ele_count)
    return [array(_ary.content[i:i+_ele_count]) for i in range(0, total_len, _ele_count)]

def maximum(x1, x2):
    assert isinstance(x1, array), "first element must be array"
    _x1 = flatten_array(x1)
    if(isinstance(x2, int) or (isinstance(x2, float))):
        result = array([_con if _con > x2 else x2 for _con in _x1.content])
        return result.reshape(x1.shape)
#        return result
    if(isinstance(x2, array)):
        assert x1.shape == x2.shape, "shape must be matching"
        _x2 = flatten_array(x2)
        result = array([_x1.content[i] if _x1.content[i] > _x2.content[i] else _x2.content[i] for i in range(_x1.size)])
        return result.reshape(x1.shape)
#        return result

def uniform(low, high, size):
    if(isinstance(size, tuple) or isinstance(size, list)):
        total_size = reduce(lambda x,y:x*y, size)
        result = array([random.uniform(low, high) for i in range(total_size)])
        return result.reshape(size)
#        return result
    elif(isinstance(size, int)):
        return array([random.uniform(low, high) for i in range(size)])
    
def hstack(tup):
    assert len(tup) == 2;"must two ele"
    a0 = tup[0]
    a1 = tup[1]
    concentrate = lambda x,y:x+y
    return array(map(concentrate, a0.content, a1.content))

def _max(a):
#    global max
    a = flatten_array(a)
    return max(a.content)
def _min(a):
    a = flatten_array(a)
    return min(a.content)


def clip(a, a_min, a_max):
    _a = flatten_array(a)
    result = []
    for ele in _a.content:
        if ele > a_max:
            ele = a_max
        elif ele < a_min:
            ele = a_min
        result.append(ele)
    return array(result).reshape(a.shape)
print("import _numpy success.")
#a0 = [[7, 8, 9], [10, 11, 12]]
#a1 = [[1, 2, 3], [4, 5, 6]]
#b0 = array(a0)
#b1 = array(a1)
#print(dot(b1, b0))
#print("flatten_array(b0):", reshape(flatten_array(b0), [3, 2]))
#print("b0:", b0)
#print("b0.size:", b0.size)
#print("b0.shape:", b0.shape)
#b0.reshape((3, 2))
#print("b0_reshape:", reshape(b0, (3, 1)))
#print("mean(b0):", mean(b0, axis = 1))
#print(zeros((3, 2)))
#print(b1)
#print(b0)
#print(b0.T)
#print(array_split(b0, 1))
#print(b0 - b1)
#print(b0)
#print(b1)
#_sum(b0, 0)
#print(isinstance(b0, array))
#print(maximum(b1, 5))


#a = array(a0)
#import time
#
#t1 = time.time()
#for i in range(100000):
#    flatten_array2D(a)
#t2 = time.time()
#print (
#    "@timefn:" +  " took " + str(t2 - t1) + " seconds")
#
#t1 = time.time()
#for i in range(100000):
#    flatten_array(a)
#t2 = time.time()
#print (
#    "@timefn:" +  " took " + str(t2 - t1) + " seconds")
