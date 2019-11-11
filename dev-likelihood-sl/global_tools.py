#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Mauricio Bustamante"
__email__ = "mbustamante@nbi.ku.dk"


"""
global_tools.py:
    Routines for general uses, such as file reading and writing

Created: 2018/03/30 12:28
Last modified: 2018/06/22 15:44
"""

# try:
   # import cPickle as pickle
# except:
   # import pickle

# from interval import interval, inf

import pandas as pd


Flatten = lambda l: [item for sublist in l for item in sublist]


def Write_Data_File(data, filename_data, lst_headers, \
    output_format='{:7.1E}'):

    file = open(filename_data,"w")

    # Write headers as comments
    if (lst_headers != None):
        for header in lst_headers:
            file.write("# "+header+'\n')

    num_lines = len(data)
    # num_columns = len(data[0])

    out_data = []

    for i in range(num_lines):
        out = ''
        if isinstance(data[i], list):
            num_columns = len(data[i])
            for j in range(num_columns):
                # out = out+str(data[i][j])+' '
                if isinstance(output_format, list):
                    out = out+str(output_format[j].format(data[i][j]))+' '
                else:
                    out = out+str(output_format.format(data[i][j]))+' '
        else:
            out = out+str(output_format.format(data[i]))+' '
        out = out+'\n'
        file.write(out)

    file.close()

    return


def Write_Data_File_Append(data, file, lst_headers, output_format='{:10.9f}'):

    # Write headers as comments
    if (lst_headers != None):
        for header in lst_headers:
            file.write("# "+header+'\n')

    num_lines = len(data)
    # num_columns = len(data[0])

    out_data = []

    for i in range(num_lines):
        out = ''
        if isinstance(data[i], list):
            num_columns = len(data[i])
            for j in range(num_columns):
                # out = out+str(data[i][j])+' '
                out = out+str(output_format.format(data[i][j]))+' '
        else:
            out = out+str(output_format.format(data[i]))+' '
        out = out+'\n'
        file.write(out)

    return


def Write_Data_File_Binary_Append_v2(data, file, lst_headers):

    # Write headers as comments
    if (lst_headers != None):
        for header in lst_headers:
            file.write("# "+header+'\n')

    for item in data:
        file.write(item)

    return


def Write_Data_File_Binary_Append(data, file, lst_headers):

    # Write headers as comments
    if (lst_headers != None):
        for header in lst_headers:
            # hickle.dump("# "+header+'\n', file)
            pickle.dump("# "+header+'\n', file)

    for item in data:
        # hickle.dump(item, file)
        pickle.dump(item, file)

    return


def Read_Data_File_old(filename_data):

    # Deprecatd: too slow compared to Read_Data_File, which uses Pandas

    file = open(filename_data,"r")
    lines = []

    # Ignore the first count_ignore lines in the file
    count_ignore = 0

    while 1:
        line = file.readline()
        if not line: break
        if line.isspace(): break
        if ( count_ignore == 0 ):
            line_split = [s for s in line.split()]
            if (line_split[0] != "#"): # Ignore comment lines
                lines.append(line_split)
        else:
            countIgnore -= 1
    file.close()

    output = [[float(line[i]) for line in lines] \
                for i in range(0,len(lines[0]))]

    return output


def Read_Data_File(filename_data, header=None):

    # Pandas dataframe
    df = pd.read_csv(filename_data, delim_whitespace=True, dtype=float,
            header=header, comment='#')

    # We extract and return individual columns as numpy arrays
    data = [df[df.columns[i]].to_numpy(dtype=float, copy=False) \
            for i in range(len(df.columns))]

    return data


"""
# filename_data = '/mnt/sda2/Research/ICNuDetection/dev/out/event_rate_alpha_20_detector/shower_rate_nue_nuebar.dat'
filename_data = '/mnt/sda2/Research/ICNuDetection/dev/in/dsdy_electron/dsdy_electron_nuebar_e_to_nutaubar_tau_water.dat'

ntests = 30
start = time.time()
for i in range(ntests):
    data = Read_Data_File(filename_data)
    # data = Read_Data_File_old(filename_data)
stop = time.time()
print('Elapsed avg. [s]: ', (stop-start)/ntests)
# print(filename_data)
quit()

# data = Read_Data_File(filename_data)
# print(len(data))
# print(data[0])

# data = Read_Data_File_old(filename_data)
# print(len(data[0]))

# quit()
"""


def Read_Data_File_As_Rows(filename_data):

    file = open(filename_data,"r")
    lines = []

    # Ignore the first count_ignore lines in the file
    count_ignore = 0

    while 1:
        line = file.readline()
        if not line: break
        if line.isspace(): break
        if ( count_ignore == 0 ):
            line_split = [s for s in line.split()]
            if (line_split[0] != "#"): # Ignore comment lines
                lines.append(line_split)
        else:
            countIgnore -= 1
    file.close()

    # output = [[float(line[i]) for line in lines] \
                # for i in range(0,len(lines[0]))]
    output = [ [float(element) for element in line] for line in lines]

    return output


def Read_Data_File_Next_Line(file_handler):

    while 1:
        line = file_handler.readline()
        if ((not line) or line.isspace()): # End of file?
            output = None
            flag_success = False
            break
        line_split = [s for s in line.split()]
        # Ignore comment lines until we find a line with data
        if (line_split[0] != "#"):
            output = [float(line_split[i]) for i in range(0,len(line_split))]
            flag_success = True
            break

    return flag_success, output


def Read_Data_File_Next_Chunk(file_handler, chunk_size=10000):

    # Read and return the next "chunk_size" elements in the file

    lst_lines = []
    count_lines = 0

    flag_eof_reached = False

    while (count_lines < chunk_size):
        line = file_handler.readline()
        if ((not line) or line.isspace()): # End of file?
            flag_eof_reached = True
            break
        # If not EOF, then split the line into sub-elements
        line_split = [s for s in line.split()]
        # Ignore comment lines until we find a line with data
        if (line_split[0] != "#"):
            lst_lines.append( \
                [float(line_split[i]) for i in range(0,len(line_split))])
            count_lines += 1

    return flag_eof_reached, lst_lines


def Skip_Header_Lines_Old(file_handler, num_lines_header):

    for i in range(num_lines_header):
        # next(file_handler)
        x = file_handler.readline()

    return


def Skip_Header_Lines(file_handler):

    file_handler_copy = file_handler
    num_chars_to_skip = 0

    while 1:

        line = file_handler_copy.readline()
        if ((not line) or line.isspace()): # End of file?
            flag_eof_reached = True
            break
        # If not EOF, then split the line into sub-elements
        line_split = [s for s in line.split()]
        # Ignore comment lines until we find a line with data
        if (line_split[0] == "#"):
            num_chars_to_skip += len(line)
        else:
            break

    file_handler.seek(num_chars_to_skip, 0)

    return


def Number_Chars_Header(file_handler):

    file_handler_copy = file_handler
    num_chars_to_skip = 0

    while 1:

        line = file_handler_copy.readline()
        if ((not line) or line.isspace()): # End of file?
            flag_eof_reached = True
            break
        # If not EOF, then split the line into sub-elements
        line_split = [s for s in line.split()]
        # Ignore comment lines until we find a line with data
        if (line_split[0] == "#"):
            num_chars_to_skip += len(line)
        else:
            break

    return num_chars_to_skip


def Skip_Lines(file_handler, num_lines, header_offset=0, chars_per_line=17):

    # Skip num_lines lines from the current position
    file_handler.seek(num_lines*chars_per_line+header_offset, 0)

    return


def Intersection_Interval_Old(lst_intervals):

    intersection_interval = interval(lst_intervals[0])

    for i in range(1,len(lst_intervals)):
        intersection_interval = intersection_interval & \
                                interval(lst_intervals[i])

    if (len(intersection_interval.extrema) == 2):
        return [intersection_interval.extrema[0][0], \
                intersection_interval.extrema[1][1]]
    elif (len(intersection_interval.extrema) == 1):
        return [intersection_interval.extrema[0][0], \
                intersection_interval.extrema[0][0]]


def Intersection_Interval(lst_intervals):

    # lst_intervals is a list of lists of intervals.
    #     lst_intervals = [
    #         [[int_0_0_lo, int_0_0_hi], [int_0_1_lo, int_0_1_hi], ...],
    #         [[int_1_0_lo, int_1_0_hi], [int_1_1_lo, int_1_1_hi], ...],
    #         ...
    #         [[int_N_0_lo, int_N_0_hi], [int_N_1_lo, int_N_1_hi], ...]]
    # Each element of lst_intervals corresponds to a different trajectory.
    # So, the elements of lst_intervals[i] are the allowed intervals for
    # the i-th trajectory.

    lst_intersection_intervals = []

    for i in range(len(lst_intervals[0])):

        # print("i =", i)
        # Read the next interval of the first element of lst_intervals
        intersection_intervals = interval(lst_intervals[0][i])
        # print(intersection_intervals)

        # Loop over all trajectories and find the intersection of all intervals
        for j in range(1,len(lst_intervals)):
            # print("j =", j)
            for k in range(len(lst_intervals[j])):
                if (k == 0):
                    intervals = interval(lst_intervals[j][k])
                else:
                    intervals = intervals | interval(lst_intervals[j][k])
                # print(intervals)
            intersection_intervals = intersection_intervals & intervals
            # print(intersection_intervals)

        # Save the intersection intervals
        for intersection_interval in intersection_intervals:
            intersection_interval = interval(intersection_interval)
            if (len(intersection_interval.extrema) == 2):
                lst_this_intersection_interval = \
                    [intersection_interval.extrema[0][0], \
                            intersection_interval.extrema[1][1]]
            elif (len(intersection_interval.extrema) == 1):
                lst_this_intersection_interval = \
                    [intersection_interval.extrema[0][0], \
                            intersection_interval.extrema[0][0]]
            lst_intersection_intervals.append(lst_this_intersection_interval)
        # print()

    return lst_intersection_intervals


def Union_Interval(lst_intervals):

    # lst_intervals is a list of lists of intervals.
    #     lst_intervals = [
    #         [[int_0_0_lo, int_0_0_hi], [int_0_1_lo, int_0_1_hi], ...],
    #         [[int_1_0_lo, int_1_0_hi], [int_1_1_lo, int_1_1_hi], ...],
    #         ...
    #         [[int_N_0_lo, int_N_0_hi], [int_N_1_lo, int_N_1_hi], ...]]
    # Each element of lst_intervals corresponds to a different trajectory.
    # So, the elements of lst_intervals[i] are the allowed intervals for
    # the i-th trajectory.

    if (len(lst_intervals) == 0):
        print("ERROR Union_Interval: lst_intervals cannot have length 0")
        quit()

    lst_sets = [[interval(interv) \
                for interv in intervals] for intervals in lst_intervals]
    lst_sets_flattened = \
        [interv for subset in lst_sets for interv in subset]
    lst_sets = []

    # Find the union of all interval sets
    set_union = interval([lst_sets_flattened[0]])
    if (len(lst_intervals) > 1):
        for k in range(1,len(lst_sets_flattened)):
            set_union = set_union | interval([lst_sets_flattened[k]])

    # Transform the set objects into simple lists
    lst_union_intervals = []
    for this_set in set_union:
        this_set = interval(this_set)
        if (len(this_set.extrema) == 2):
            lst_this_interval = [this_set.extrema[0][0],this_set.extrema[1][1]]
        elif (len(this_set.extrema) == 1):
            lst_this_interval = [this_set.extrema[0][0],this_set.extrema[0][0]]
        lst_union_intervals.append(lst_this_interval)

    return lst_union_intervals

# print(interval([1.4,23.4]) & interval([10.0,25.0]))
# print(interval([1.4,23.4]) & interval([10.0,25.0]) & interval(11.0, 12.0) & \
    # interval(14.0, 16.0))
# lst_intervals = [[1.0,20.0],[3.4,19.0],[6.0,19.5]]
# print(Intersection_Interval_Old(lst_intervals))


# x = interval([1.e-34,1.e-32]) & interval([1.e-34,1.e-32])
# print(x)
# y = interval([1.e-3,1.e-1]) & interval([1.e-2,2.e-1])
# print(y)

# print(Intersection_Interval_Old([[0.0,10.0],[1.3,4.5]])
# print(Intersection_Interval_Old([[1.e-32,1.e-32],[1.e-32,1.e-32]]))

# print(Intersection_Interval_Old([[0.0,10.0],[1.3,4.5]]))
# print(Intersection_Interval([
    # [[0.0,10.0],[11.0,20.0]],
    # [[1.0,8.0],[9.0,15.0]],
    # [[1.0,6.0],[7.0,18.0]]
    # ]))

# l1 = [[0,2],[3,4],[5,6]]
# l2 = [[0,0]]
# l = [l1,l2]
# l_union = Union_Interval(l)
# print(l_union)
# print(len(l_union))
