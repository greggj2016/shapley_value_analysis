import numpy as np
import os

for P in [25, 50]:
    for a in [0.25, 0.5]:
        type = "P_" + str(P) + "_a_" + str(a) + "_no_corr"

        if os.path.isdir("Alena_Job_runners_" + type) == False:
            os.mkdir("Alena_Job_runners_" + type)

        vals = np.arange(100) + 1
        file_main = open("Alena_Job_runners_" + type + ".sh", "w")
        file_main.write('#!/bin/bash\n')
        file_main.write('#BSUB -J Alena_Job_runners_' + type + '\n')
        file_main.write('#BSUB -o Alena_Job_runners_' + type + '.out\n')
        file_main.write('#BSUB -e Alena_Job_runners_' + type + '.err\n\n')

        for i, v in enumerate(vals):
            file = open("Alena_Job_runners_" + type + "/Alena_Job_runner" + str(v) + ".sh", "w")
            file.write('#!/bin/bash\n')
            file.write('#BSUB -J Alena_Job_runners_' + type + "_" + str(i) + '\n')
            file.write('#BSUB -o Alena_Job_runners_' + type + '/Alena_job_runner' + str(v) + '.out\n')
            file.write('#BSUB -e Alena_Job_runners_' + type + '/Alena_job_runner' + str(v) + '.err\n')
            file.write('#BSUB -n 4\n')
            file.write('#BSUB -M 4000MB\n')
            file.write('#BSUB -R "span[hosts=1] rusage[mem=4000MB]"\n')
            file.write('source activate hibachi\n\n')

            file.write('python -m scoop -n 4 hib.py -f random -s ' + str(v) + ' -P ' + str(P) + ' -R 10000 -C 10 -p 1000 -g 1000 -i 3 ')
            file.write('-a ' + str(a) + ' -y 100 -o hibachi_' + type + ' -effs 3 -ps True\n')
            file_main.write("bsub < Alena_Job_runners_" + type + "/Alena_Job_runner" + str(v) + ".sh\n")

            file.close()