# -*- coding: utf-8 -*-
import inv_matrix_gen
import edit_operations

def channel_prob_data(file,path):
    with open(file) as file:
        lines = [line.rstrip('\n') for line in file]
    
    
    channel = []
    
    for line in lines:
        channel.append(line.split("\t"))
    sum_prob = 0
    
    for i in range(len(channel)):
        for j in range(len(channel[i])):
            sum_prob = sum_prob + int(channel[i][j])

    for i in range(len(channel)):
        for j in range(len(channel[i])):
            channel[i][j] = int(channel[i][j]) / sum_prob
            
    inv_matrix_gen.save_obj(channel,path)


channel_prob_data("../data/addoneAddXY","../data/channel_data/add_X_Y")
channel_prob_data("../data/addoneDelXY","../data/channel_data/del_X_Y")
channel_prob_data("../data/addoneRevXY","../data/channel_data/rev_X_Y")
channel_prob_data("../data/addoneSubXY","../data/channel_data/sub_X_Y")


