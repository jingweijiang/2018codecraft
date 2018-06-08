# coding=utf-8
import sys
import os
import predictor


def main():
    print 'main function begin.'
#    if len(sys.argv) != 4:
#        print 'parameter is incorrect!'
#        print 'Usage: python esc.py ecsDataPath inputFilePath resultFilePath'
#        exit(1)
    # Read the input files
    ecsDataPath = "data_input/TrainData_2015.12.txt"
    inputFilePath = "data_input/input_3hosttypes_all_flavors_1week.txt"
    resultFilePath = "data_input/result.txt"
    answerFilePath = "data_input/TestData_2016.1.8_2016.1.14.txt"
    ecs_infor_array = read_lines(ecsDataPath)
    input_file_array = read_lines(inputFilePath)
    answer_array = read_lines(answerFilePath)
    
    # implementation the function predictVm
    predic_result = predictor.predict_vm(ecs_infor_array, input_file_array, answer_array)
    print("predic_result：", predic_result)
    # write the result to output file
    if len(predic_result) != 0:
        write_result(predic_result, resultFilePath)
    else:
        predic_result.append("NA")
        write_result(predic_result, resultFilePath)
    print 'main function end.'


def write_result(array, outpuFilePath):
    with open(outpuFilePath, 'w') as output_file:
#        for item in array:
##            print(item)
#            output_file.write("%s\n" % item)
        output_file.write(array[:-1])


def read_lines(file_path):
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                array.append(line)
        return array
    else:
        print 'file not exist: ' + file_path
        return None


if __name__ == "__main__":
    main()
#58 76 coe=1.2
#55 coe=1.3
#62, 61, 61, 62, 69, 72, 64, 66 coe=1.1