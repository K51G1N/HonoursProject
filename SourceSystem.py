import os
import numpy as np
import DataAcquisition
import keagan_first_sim
import Snippy
import SnipPS
import MultiModel
import MultiModel_Test
absPath = input("Give absolute path to Experiment 1:")
absPath2 = input("Give absolute path to Experiment 2:")
absPath4 = input("Give absolute path to Experiment 3:")

# ExpPaths = [absPath, absPath2, absPath4]

# complexity =  input("Will there be Extended Sources ? True/False")
# nsources = input("Give the number of sources Hundred or Thousand: (H, T) \n")
# ES = input("Will there be ES or Noise: (T, F) \n")
# pf_supplied = input("Give Hogbom PF value: 0.01-0.9")




def create_output_dirs(outdir, N=10):
    """create the output folders for the simulations"""

    base_path = outdir + ""
    train_path = outdir + "/TrainData"
    unseen_path = outdir + "/UnseenData"

    train_Extract = train_path + "/Extract"
    train_Extract_NPS = train_path + "/Extract/NPS"
    train_Extract_PS = train_path + "/Extract/PS"

    unseen_Extract = unseen_path + "/Extract"
    unseen_Extract_NPS = unseen_path + "/Extract/NPS"
    unseen_Extract_PS = unseen_path + "/Extract/PS"

    ML = base_path + "/ML"
    ML_SVM = base_path + "/ML/SVM"
    ML_SVM_Training = ML_SVM + "/Training"
    ML_SVM_Testing = ML_SVM  + "/Testing"
    ML_RF = base_path + "/ML/RF"
    ML_RF_Training = ML_RF + "/Training"
    ML_RF_Testing = ML_RF + "/Testing"
    ML_DT = base_path + "/ML/DT"
    ML_DT_Training = ML_DT + "/Training"
    ML_DT_Testing = ML_DT + "/Testing"
    ML_kNN = base_path + "/ML/kNN"
    ML_kNN_Training = ML_kNN + "/Training"
    ML_kNN_Testing = ML_kNN + "/Testing"

    folders = [base_path, train_path, unseen_path, #0,1,2
               train_Extract, train_Extract_NPS, train_Extract_PS, #3,4,5
               unseen_Extract, unseen_Extract_NPS, unseen_Extract_PS, #6,7,8
               ML, #9
               ML_SVM, ML_SVM_Training, ML_SVM_Testing, #10, 11, 12
               ML_RF, ML_RF_Training, ML_RF_Testing, #13, 14, 15
               ML_DT, ML_DT_Training, ML_DT_Testing, #16, 17, 18
               ML_kNN, ML_kNN_Training, ML_kNN_Testing] #19, 20, 21

    # # for i in range(1, N):
    # folders.append(folders[1] + "/Extract")
    # folders.append(folders[1]  + "/Extract/NPS")
    # folders.append(folders[1] + "/Extract/PS")
    #
    # folders.append(folders[2] + "/Extract")
    # folders.append(folders[2] + "/Extract/NPS")
    # folders.append(folders[2] + "/Extract/PS")
    #
    # folders.append(folders[0] +"/ML")
    # folders.append(folders[0] + "/ML/SVM")
    # folders.append(folders[0] + "/ML/RF")
    # folders.append(folders[0] + "/ML/DT")
    # folders.append(folders[0] + "/ML/kNN")
    # folders.append(outdir + "/")

    try:
        os.system("rm -rf %s" % base_path)
    except:
        pass

    # create data folders if they do not exist
    for folder in folders:
        if os.path.isdir(folder):
            print(folder, "already exist")
        else:
            os.mkdir(folder)

    print("Output directory for simulations setup successfully")
    print(folders)
    # quit()
    return folders


def Main(absPath, ES, nsources):

    print("Output directory setup Commencing ... ")
    paths = create_output_dirs(absPath)
    Train = True

    # print('Data Acquisition Commencining ... ')
    #Test Data Generation and Extraction
    np.random.seed(123)
    if ES == 'T':
        print('Test Data Acquisition Commencing ... ')
        DataAcquisition.main(paths, 1, nsources, Train)
        print('Data Acquisition Complete!')
        print('Object Detection and Extraction Commencing ... ')
        Snippy.MainMethod(paths, Train)
        print('Object Detection and Extraction Complete! ')
    else:
        print('Test Data Acquisition Commencing ... ')
        keagan_first_sim.main(paths, 1, nsources, Train)
        print('Data Acquisition Complete!')
        print('Object Detection and Extraction Commencing ... ')
        SnipPS.MainMethod(paths, nsources, Train)
        print('Object Detection and Extraction Complete! ')

    print('Training Machine Learning Models ...')
    MultiModel.MainMethod(paths)
    print('Models trained and saved ...')


    Train = False
    np.random.seed(1234)
    #Unseen Data Generation and Extraction
    if ES == 'T':
        print('Unseen Data Acquisition Commencing ... ')
        DataAcquisition.main(paths, 1, nsources, Train)
        print('Data Acquisition Complete!')
        print('Object Detection and Extraction Commencing ... ')
        Snippy.MainMethod(paths, Train)
        print('Object Detection and ExtraFaction Complete! ')
    else:
        print('Unseen Data Acquisition Commencing ... ')
        keagan_first_sim.main(paths, 1, nsources, Train)
        print('Data Acquisition Complete!')
        print('Object Detection and Extraction Commencing ... ')
        SnipPS.MainMethod(paths, nsources, Train)
        print('Object Detection and Extraction Complete! ')

    print('Testing saved Machine Learning Models ...')
    MultiModel_Test.MainMethod(paths)
    print('Testing of models complete!')



print('Experiment 1: ')
Main(absPath, 'F', 'H')
print('Experiment 2: ')
Main(absPath2, 'F', 'T')
print('Experiment 3: ')
Main(absPath4, 'T', 'T')





