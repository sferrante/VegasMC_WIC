## Convert MC_Events_LHE to .lhe file ##
import numpy as np
def arr_to_LHE(filename, array, dictionary):
    with open(filename + '.lhe', 'w') as f:
        f.write('heyyy this is an .lhe file')
        f.write('\n<MGGenerationInfo>')
        f.write('\n#  Number of Events        :      %i'%len(array))
        f.write('\n#  Integrated weight (ab)  :      %f'%dictionary['Integral'])
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