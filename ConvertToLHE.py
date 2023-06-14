## Convert MC_Events_LHE to .lhe file ##
import numpy as np

c = 3 * 10**8   
def arr_to_MyLHE(filename, array, dictionaries):
    with open(filename + '.mylhe', 'w') as f:
        f.write('<MGGenerationInfo>')
        f.write('\n#  Number of Events        :      %i'%len(array))
        integral = dictionaries[0]['Integral']*dictionaries[1]['Integral']
        f.write('\n#  Integrated weight (ab)  :      %f'%integral)
        f.write('\n</MGGenerationInfo>')
        f.write('\n</header>')
        f.write('\n<init>')
        f.write('\n ...')
        f.write('\n</init>\n')
        for i in range(len(array)):
            f.write('<event>\n')
            for j in range(len(array[i])):
                f.write(str(array[i][j][0]) + '   ')
                f.write(str(array[i][j][1]) + '   ')
                for k in range(len(array[i][0])-3):
                    f.write(str(np.format_float_scientific(np.round(array[i][j][k+2],4))) + '   ')
                f.write(str(array[i][j][-1]) + '   ')
                f.write('\n')
            f.write('...\n')
            f.write('</event>\n')
            
# def arr_to_LHE(filename, array, dictionary):
#     with open(filename + '.lhe', 'w') as f:
#         f.write('\n<MGGenerationInfo>')
#         f.write('\n#  Number of Events        :      %i'%len(array))
#         f.write('\n#  Integrated weight (ab)  :      %f'%dictionary['Integral'])
#         f.write('\n</MGGenerationInfo>')
#         f.write('\n</header>')
#         f.write('\n<init>')
#         f.write('\n ...')
#         f.write('\n</init>\n')
#         for i in range(len(array)):
#             f.write('<event>\n')
#             for j in range(len(array[i])):
#                 f.write(str(array[i][j][0]) + '   ')
#                 f.write(str(array[i][j][1]) + '   ')
#                 for k in range(len(array[i][0])-3):
#                     f.write(str(np.format_float_scientific(np.round(array[i][j][k+2],4))) + '   ')
#                 f.write(str(array[i][j][-1]) + '   ')
#                 f.write('\n')
#             f.write('...\n')
#             f.write('</event>\n')
                
            
def arr_to_LHE_temp(filename, array, dictionaries):
    with open(filename + '.lhe_temp', 'w') as f:
        f.write('\n<MGGenerationInfo>')
        f.write('\n#  Number of Events        :      %i'%len(array))
        integral = dictionaries[0]['Integral']*dictionaries[1]['Integral']
        f.write('\n#  Integrated weight (ab)  :      %f'%integral)
        f.write('\n</MGGenerationInfo>')
        f.write('\n</header>')
        f.write('\n<init>')
        f.write('\n {0} {1} {2} {3} {4} {5} {6} {7} {8} {9}'.format(
                     11,                                                         # Beam1 ID
                     11,                                                         # Beam2 ID
                     np.format_float_scientific(250,6, min_digits=6, sign=True), # Beam1 Energy
                     np.format_float_scientific(250,6, min_digits=6, sign=True), # Beam2 Energy
                     -1,                                                         # Beam1 PDFG
                     -1,                                                         # Beam2 PDFG
                     -1,                                                         # Beam1 PDFS
                     -1,                                                         # Beam2 PDFS
                     -4,                                                         # Weight ID
                      1                                                      ))  # NPR
        f.write('\n {0} {1} {2} {3}'.format(
                     np.format_float_scientific(integral, 6, min_digits=6, sign=False), # XSEC                                                        # XSEC
                     np.format_float_scientific(dictionaries[0]['Error'], 6, min_digits=6, sign=False), # XERR                                                     # XERR
                     np.format_float_scientific(integral, 6, min_digits=6, sign=False), # XMAX
                      1                                                                  ))           # LPR
        f.write('\n<generator name=\'VegasMC_WIC\' version=\'0.0.0\'> please cite ...')
        f.write('\n</init>\n')
        for i in range(len(array)):
            f.write('<event>\n')
            f.write('{0} {1:>6} {2} {3} {4} {5}'.format(len(array[i])+3, 'id', 'weight', -1, 'qed', 'qcd'))
            f.write('\n')
            ################################### Incoming Electrons ###################################
            f.write('{0:>8} {1:>4} {2:>4} {3:>4} {4:>4} {5:>4} {6} {7} {8} {9} {10} {11} {12}'.format(
                                            11,-1,0,0,0,0, ## PID, StatusCode, MotherIndex, ColorIndex
                                            np.format_float_scientific(0,6, min_digits=6, sign=True),
                                            np.format_float_scientific(0,6, min_digits=6, sign=True),
                                            np.format_float_scientific(250,6, min_digits=6, sign=True),
                                            np.format_float_scientific(250,6, min_digits=6, sign=True),
                                            np.format_float_scientific(0,6, min_digits=6, sign=True),
                                            np.format_float_scientific(0,4, min_digits=4, sign=True),
                                            np.format_float_scientific(0,4, min_digits=4, sign=True)))
            f.write('\n')
            f.write('{0:>8} {1:>4} {2:>4} {3:>4} {4:>4} {5:>4} {6} {7} {8} {9} {10} {11} {12}'.format(
                                            -11,-1,0,0,0,0, ## PID, StatusCode, MotherIndex, ColorIndex
                                            np.format_float_scientific(0,6, min_digits=6, sign=True),
                                            np.format_float_scientific(0,6, min_digits=6, sign=True),
                                            np.format_float_scientific(250,6, min_digits=6, sign=True),
                                            np.format_float_scientific(250,6, min_digits=6, sign=True),
                                            np.format_float_scientific(0,6, min_digits=6, sign=True),
                                            np.format_float_scientific(0,4, min_digits=4, sign=True),
                                            np.format_float_scientific(0,4, min_digits=4, sign=True)))
            f.write('\n')
            ################################### Inermediate Z Boson ###################################
            f.write('{0:>8} {1:>4} {2:>4} {3:>4} {4:>4} {5:>4} {6} {7} {8} {9} {10} {11} {12}'.format(
                                            23,2,1,2,0,0, ## PID, StatusCode, MotherIndex, ColorIndex
                                            np.format_float_scientific(0,6, min_digits=6, sign=True),
                                            np.format_float_scientific(0,6, min_digits=6, sign=True),
                                            np.format_float_scientific(0,6, min_digits=6, sign=True),
                                            np.format_float_scientific(0,6, min_digits=6, sign=True),
                                            np.format_float_scientific(0,6, min_digits=6, sign=True),
                                            np.format_float_scientific(0,4, min_digits=4, sign=True),
                                            np.format_float_scientific(0,4, min_digits=4, sign=True)))
            f.write('\n')
            ################################### Outgoing Particles ###################################
            label = 3
            row = [1,2,3]
            for j in range(len(array[i])):  ##### for particle in event #####
                #### Status Code ####
                if array[i][j][-2]==1: statuscode=2  # (if it decayed)  
                if array[i][j][-2]==0: statuscode=1  # (if it didn't decay)
                #### Mother Index #### (labelling is difficult here)
                row.append(j+4)
                if array[i][j][0]=='PID1' or array[i][j][0]=='PID1p': 
                    mother = [row[2],row[2]] 
                    if array[i][j][-2]==1 and array[i][j][0]=='PID1': # (if μ1 decayed)
                        label = label + 1
                else:
                    PhiN = eval(array[i][j][0][3])
                    if PhiN==2: 
                        if   array[i][j][0][-1]=='a' or array[i][j][0][-2]=='a':
                            mother = [row[-1]-2,row[-1]-2]
                        elif array[i][j][0][-1]=='b' or array[i][j][0][-2]=='b':
                            mother = [row[-1]-3,row[-1]-3]
                        else:
                            mother = [row[-1]-1, row[-1]-1]
                            label = label + 1
                    else:
                        if   array[i][j][0][-1]=='a' or array[i][j][0][-2]=='a':
                            mother = [row[-1]-4,row[-1]-4]
                        elif array[i][j][0][-1]=='b' or array[i][j][0][-2]=='b':
                            mother = [row[-1]-5,row[-1]-5]
                        else:
                            mother = [row[-1]-3, row[-1]-3]
                            label = label + 1
                #### Color Index #### 
                if    array[i][j][1]> 6: color = [0,0]     # (if it's neither)
                if 1<=array[i][j][1]<=6: color = [0,501]   # (if it's a quark)
                if 1<=-array[i][j][1]<=6: color = [501,0]   # (if it's an antiquark)
                #### c*Lifetime #### 
                if array[i][j][-2]==1: # (if it decayed) 
                    lifetime = c*array[i][j][-1]
                else: lifetime = 0
                #### Spin #### 
                if array[i][j][0][-1]=='a' or array[i][j][0][-2]=='a':
                    spin = 1
                elif array[i][j][0][-1]=='b' or array[i][j][0][-2]=='b':
                    spin = -1
                else:
                    spin = 0
                f.write('{0:>8} {1:>4} {2:>4} {3:>4} {4:>4} {5:>4} {6:>2} {7:>2} {8:>2} {9:>2} {10:>2} {11:>2} {12:>2}'.format(
                    array[i][j][1], statuscode, mother[0], mother[1], color[0], color[1],
                    np.format_float_scientific(array[i][j][4], 6, min_digits=6, sign=True),  # Px
                    np.format_float_scientific(array[i][j][5], 6, min_digits=6, sign=True),  # Py
                    np.format_float_scientific(array[i][j][6], 6, min_digits=6, sign=True),  # Pz 
                    np.format_float_scientific(array[i][j][3], 6, min_digits=6, sign=True),  # E 
                    np.format_float_scientific(array[i][j][2], 6, min_digits=6, sign=True),  # M
                    np.format_float_scientific(lifetime      , 4, min_digits=4, sign=True),  # ct
                    np.format_float_scientific(spin          , 4, min_digits=4, sign=True))) # spin
                f.write('\n')
            ################################### Extra info ... ###################################
            f.write('<mgrwt>')
            f.write('<rscale>  0 0 </rscale>')
            f.write('\n')
            f.write('<asrwt>  0 </asrwt>')
            f.write('\n')
            f.write('<pdfrwt beam="2">  1      0 0 0 </pdfrwt')
            f.write('\n')
            f.write('<pdfrwt beam="1">  1      0 0 0 </pdfrwt')
            f.write('\n')
            f.write('<totfact> 0 </totfact>')
            f.write('\n')
            f.write('</mgrwt')
            f.write('\n')
            f.write('</event>')
            f.write('\n')
        f.write('</LesHoucesEvents>')