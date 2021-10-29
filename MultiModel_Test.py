'''
Currently:
We have the point sources (images) as npy arrays.
We want to get those files in and then test them with the model.

We need to implement kNN, RF, ERF to also accept training.
    -> Requires us to re-run keagan_first_sim
    -> Generate massive dataset i.e repeat 10 times have 10 of each split it up. compile into two sets (1000+|1000-)
    -> Train and Test the 4 ML models
        -> Gridsearch to find optimal parameters
        -> Score pick best model: lightweight, fast and accuracy
    -> Test Hogbom candidate locations based off of PF values
    -> Accuracy(PF)K
'''
# ---------------------------------------------------------------------------------------------------------------
#Load your own image files with categories as subfolder names
# This example assumes that the images are preprocessed, and classifies using tuned SVM
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import skimage
from skimage.io import imread
from skimage.transform import resize
# For model saving and loading:
import os
import joblib

np.random.seed(100)

# load images as 60 x 60:
def load_image_files(container_path, dimension=(60, 60)):
    image_dir = Path(container_path[6])
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]


    descr = "Your own dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            ps = np.load(file)
            flat_data.append(ps.flatten())
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    # return in the exact same format as the built-in datasets
    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

def MainMethod(paths):
    image_dataset = load_image_files(paths)
    pathModel = ""
    pathFile  = ""
    with open(paths[9]+"/Test_Accuracies.txt", 'a') as A:
        with open(paths[9]+'/Test_results_All.txt', 'w') as f:
            # f.write('readme')
            figure, axes = plt.subplots()
            # Split data, but randomly allocate to training/tesSVMt sets

            X_train, X_test, y_train, y_test = train_test_split(
                image_dataset.data, image_dataset.target, test_size=0.6,random_state=100)

            Scaler = MinMaxScaler()
            X_train = Scaler.fit_transform(X_train)
            X_test = Scaler.transform(X_test)

            ns_probs = [0 for _ in range(len(y_test))]

            # Train data with parameter optimization for linear and Gaussian
            tune_param_svm =[
                            {'C': [0.01, 0.1, 1, 10.0, 100.0], 'gamma': [0.001, 0.0001], 'kernel': ['linear', 'rbf']},
                            {'C': [0.01, 0.1, 1, 10.0, 100.0], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
                        ]

            tune_param_rf =[
                {'n_estimators': [100, 200, 300, 500, 600], 'criterion': ['gini', 'entropy'],
                 'max_depth': [30, 50, 100], 'min_samples_leaf': [2, 3, 5, 7]}
            ]
            # tune_param_erf =[
            #     {'n_estimators': [300, 500, 700, 1000], 'criterion': ['gini', 'entropy'], 'min_samples_leaf': [2, 3, 5, 7], 'max_depth': [30, 50, 100]}
            # ]

            tune_param_knn =[
                {'n_neighbors': [3, 5, 7]}
            ]
            tune_param_dt =[
                {'max_depth': [30, 50, 100], 'criterion': ['gini','entropy'], 'min_samples_leaf': [2, 3, 5, 7]}
            ]

            tune_param_xgb =[{
                'max_depth': range (20, 100, 20),
                'n_estimators': range(200, 1000, 200),
                'learning_rate': [0.1, 0.01, 0.05]
            }]
            # Output a pickle file for the model if it does not exist
            print("SVM:")
            f.write('SVM: \n')

            if(not os.path.isfile(paths[10] + '/saved_model_SVM.pkl')):
                print("no existo...saving model")
                svc = svm.SVC(probability=True)
                grid = GridSearchCV(svc, tune_param_svm, cv=3)
                grid.fit(X_train, y_train)
                joblib.dump(grid, paths[10] + '/saved_model_SVM.pkl')
                modelLoad = grid
            else:
                # print("already exists...loading model")
                modelLoad = joblib.load(paths[10] + '/saved_model_SVM.pkl')

            # Predict
            y_pred = modelLoad.predict(X_test)

            # Model Information
            print("Best Parameters: \n{}\n".format(modelLoad.best_params_))
            print("Best Test Score: \n{}\n".format(modelLoad.best_score_))

            f.write("Best Parameters: \n{}\n".format(modelLoad.best_params_))
            f.write("Best Test Score: \n{}\n".format(modelLoad.best_score_))

            # Evaluate
            print("Classification report for - \n{}:\n{}\n".format(
                modelLoad, metrics.classification_report(y_test, y_pred, target_names=["Non Point Source", "Point Source"], digits=4)))
            f.write("Classification report for - \n{}:\n{}\n".format(
                modelLoad, metrics.classification_report(y_test, y_pred, target_names=["Non Point Source", "Point Source"], digits=4)))

            acc = metrics.accuracy_score(y_test, y_pred)
            print("Accuracy %s"%(acc))
            f.write("Accuracy %s \n"%(acc))
            A.write('SVM: %s \n'%(acc))

            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            print(cnf_matrix)
            cmtext = np.array2string(cnf_matrix)
            f.write("%s \n"%(cmtext))

            metrics.plot_confusion_matrix(modelLoad, X_test, y_test, display_labels=["Non PS", "PS"], cmap=plt.cm.Blues)
            plt.title("Confusion Matrix for SVM")
            # plt.show()
            plt.savefig(paths[12] + '/CNF_SVM.png', dpi=300, bbox_inches = "tight")
            # plt.show()
            plt.close()
            #
            #
            #AUC and ROC curves
            svm_probs = modelLoad.predict_proba(X_test)
            svm_probs = svm_probs[:,1]

            ns_auc = roc_auc_score(y_test, ns_probs)
            svm_auc = roc_auc_score(y_test, svm_probs)

            print('Random Chance: ROC AUC=%.3f' % (ns_auc))
            print('SVM: ROC AUC=%.3f' % (svm_auc))

            f.write('Random Chance: ROC AUC=%.3f' % (ns_auc))
            f.write('SVM: ROC AUC=%.3f' % (svm_auc))

            ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
            svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)

            # plot the roc curve for the model
            plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Chance')
            plt.plot(svm_fpr, svm_tpr, marker='.', label='SVM')

            axes.plot(svm_fpr, svm_tpr, marker='.', label='SVM', color='r')

            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # show the legend
            plt.legend()
            # show the plot
            # plt.show()
            plt.savefig(paths[12] + '/ROC_SVM.png')
            # plt.show()
            plt.close()



            print("***********************************************************************************************")
            f.write("*********************************************************************************************** \n")

            # quit()

            print("Random Forrest")
            f.write("\n Random Forrest")

            if(not os.path.isfile(paths[13]+'/saved_model_rf.pkl')):
                print("RF no existo...saving model")
                RFC = RandomForestClassifier()
                grid = GridSearchCV(RFC, tune_param_rf, cv=3)
                grid.fit(X_train, y_train)
                joblib.dump(grid, paths[13]+'/saved_model_rf.pkl')
                modelLoad = grid
            else:
                print("already exists...loading model")
                modelLoad = joblib.load(paths[13]+'/saved_model_rf.pkl')

            # Predict
            y_pred = modelLoad.predict(X_test)

            # Model Information
            print("Best Parameters: \n{}\n".format(modelLoad.best_params_))
            print("Best Test Score: \n{}\n".format(modelLoad.best_score_))
            f.write("\n Best Parameters: \n{}\n".format(modelLoad.best_params_))
            f.write("\n Best Test Score: \n{}\n".format(modelLoad.best_score_))
            # Evaluate
            print("Classification report for - \n{}:\n{}\n".format(
                modelLoad, metrics.classification_report(y_test, y_pred , target_names=["Non Point Source", "Point Source"], digits=4)))
            f.write("Classification report for - \n{}:\n{}\n".format(
                modelLoad, metrics.classification_report(y_test, y_pred, target_names=["Non Point Source", "Point Source"], digits=4)))

            print("Accuracy",metrics.accuracy_score(y_test, y_pred))
            f.write("Accuracy {}".format(metrics.accuracy_score(y_test, y_pred)))
            A.write('RF: {} \n'.format(metrics.accuracy_score(y_test, y_pred)))

            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            print("{}\n".format(cnf_matrix))
            f.write("\n {}".format(np.array2string(cnf_matrix)))

            metrics.plot_confusion_matrix(modelLoad, X_test, y_test, display_labels=["Non PS", "PS"], cmap=plt.cm.Blues)
            plt.title("Confusion Matrix for RF")

            #
            plt.savefig(paths[15]+'/CNF_RF.png', dpi=300, bbox_inches = "tight")
            # plt.show()
            plt.close()
            f.write('\n')

            # AUC and ROC curves
            RF_probs = modelLoad.predict_proba(X_test)
            RF_probs = RF_probs[:,1]

            ns_auc = roc_auc_score(y_test, ns_probs)
            RF_auc = roc_auc_score(y_test, RF_probs)

            print('Random Chance: ROC AUC=%.3f' % (ns_auc))
            print('RF: ROC AUC=%.3f' % (RF_auc))
            f.write('Random Chance: ROC AUC=%.3f \n' % (ns_auc))
            f.write('RF: ROC AUC=%.3f \n' % (RF_auc))

            ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
            RF_fpr, RF_tpr, _ = roc_curve(y_test, RF_probs)

            # plot the roc curve for the model
            plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Chance')
            plt.plot(RF_fpr, RF_tpr, marker='.', label='RF')

            axes.plot(RF_fpr, RF_tpr, marker='.', label='RF', color='g')

            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # show the legend
            plt.legend()
            # show the plot
            # plt.show()
            plt.savefig(paths[15]+'/ROC_RF.png', dpi=300, bbox_inches = "tight")
            # plt.show()
            plt.close()
            print("***********************************************************************************************")
            f.write("*********************************************************************************************** \n")

            print("Decision Tree")
            f.write("Decision Tree: \n")

            if (not os.path.isfile(paths[16] + '/saved_model_DT.pkl')):
                print("DT no existo...saving model")
                DT = DecisionTreeClassifier()
                grid = GridSearchCV(DT, tune_param_dt, cv=3)
                grid.fit(X_train, y_train)

                joblib.dump(grid, paths[16]+'/saved_model_DT.pkl')
                modelLoad = grid
            else:
                # print("already exists...loading model")
                modelLoad = joblib.load(paths[16]+'/saved_model_DT.pkl')

            # Predict
            y_pred = modelLoad.predict(X_test)

            # Model Information
            print("Best Parameters: \n{}\n".format(modelLoad.best_params_))
            print("Best Test Score: \n{}\n".format(modelLoad.best_score_))

            f.write("Best Parameters: \n{}\n".format(modelLoad.best_params_))
            f.write("Best Test Score: \n{}\n".format(modelLoad.best_score_))
            # Evaluate
            print("Classification report for - \n{}:\n{}\n".format(
                modelLoad, metrics.classification_report(y_test, y_pred, target_names=["Non Point Source", "Point Source"], digits=4)))

            f.write("Classification report for - \n{}:\n{}\n".format(
                modelLoad, metrics.classification_report(y_test, y_pred, target_names=["Non Point Source", "Point Source"], digits=4)))

            print("Accuracy", metrics.accuracy_score(y_test, y_pred))
            f.write("Accuracy {} \n".format(metrics.accuracy_score(y_test, y_pred)))
            A.write('DT: {} \n'.format(metrics.accuracy_score(y_test, y_pred)))
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            print(cnf_matrix)

            f.write("{} \n".format(np.array2string(cnf_matrix)))

            metrics.plot_confusion_matrix(modelLoad, X_test, y_test, display_labels=["Non PS", "PS"], cmap=plt.cm.Blues)
            plt.title("Confusion Matrix for DT")
            # plt.show()
            plt.savefig(paths[18]+'/CNF_DT.png', dpi=300, bbox_inches = "tight")
            # plt.show()
            plt.close()

            #AUC and ROC curves
            DT_probs = modelLoad.predict_proba(X_test)
            DT_probs = DT_probs[:,1]

            ns_auc = roc_auc_score(y_test, ns_probs)
            DT_auc = roc_auc_score(y_test, DT_probs)

            f.write('\n')
            print('Random Chance: ROC AUC=%.3f' % (ns_auc))
            print('DT: ROC AUC=%.3f' % (DT_auc))

            f.write('Random Chance: ROC AUC=%.3f' % (ns_auc))
            f.write('DT: ROC AUC=%.3f' % (DT_auc))

            ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
            DT_fpr, DT_tpr, _ = roc_curve(y_test, DT_probs)

            # plot the roc curve for the model
            plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Chance')
            plt.plot(DT_fpr, DT_tpr, marker='.', label='DT')

            axes.plot(DT_fpr, DT_tpr, marker='.', label='DT', color='b')

            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # show the legend
            plt.legend()
            # show the plot
            plt.savefig(paths[18] + '/ROC_DT.png', dpi=300, bbox_inches = "tight")
            # plt.show()
            plt.close()

            print("***********************************************************************************************")
            f.write("*********************************************************************************************** \n")

            print("kNN Model")
            f.write('kNN: ')
            if (not os.path.isfile( paths[19]+'/saved_model_KNN.pkl')):
                print("kNN no existo...saving model")
                kNN = KNeighborsClassifier()
                grid = GridSearchCV(kNN, tune_param_knn, cv=3)
                grid.fit(X_train, y_train)
                joblib.dump(grid, paths[19]+'/saved_model_KNN.pkl')
                modelLoad = grid
            else:
                # print("already exists...loading model")
                modelLoad = joblib.load(paths[19]+'/saved_model_KNN.pkl')

            # Predict
            y_pred = modelLoad.predict(X_test)

            # # Model Information
            # print("Best Parameters: \n{}\n".format(modelLoad.best_params_))
            # print("Best Test Score: \n{}\n".format(modelLoad.best_score_))

            # Model Information
            print("Best Parameters: \n{}\n".format(modelLoad.best_params_))
            print("Best Test Score: \n{}\n".format(modelLoad.best_score_))

            f.write("Best Parameters: \n{}\n".format(modelLoad.best_params_))
            f.write("Best Test Score: \n{}\n".format(modelLoad.best_score_))

            f.write('\n')
            # Evaluate
            print("Classification report for - \n{}:\n{}\n".format(
                modelLoad, metrics.classification_report(y_test, y_pred), target_names=["Non Point Source", "Point Source"], digits=4))

            f.write("Classification report for - \n{}:\n{}\n".format(
                modelLoad, metrics.classification_report(y_test, y_pred), target_names=["Non Point Source", "Point Source"], digits=4))
            print("Accuracy ", metrics.accuracy_score(y_test, y_pred))
            f.write("Accuracy {} \n".format(metrics.accuracy_score(y_test, y_pred)))
            A.write('kNN: {} \n'.format(metrics.accuracy_score(y_test, y_pred)))
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            print(cnf_matrix)
            f.write(np.array2string(cnf_matrix))

            metrics.plot_confusion_matrix(modelLoad, X_test, y_test, display_labels=["Non PS", "PS"], cmap=plt.cm.Blues)
            plt.title("Confusion Matrix for kNN")
            # plt.show()
            plt.savefig(paths[21]+'/CNF_kNN.png', dpi=300, bbox_inches = "tight")
            # plt.show()
            plt.close()

            #AUC and ROC curves
            kNN_probs = modelLoad.predict_proba(X_test)
            kNN_probs = kNN_probs[:,1]

            ns_auc = roc_auc_score(y_test, ns_probs)
            kNN_auc = roc_auc_score(y_test, kNN_probs)

            print('Random Chance: ROC AUC=%.3f' % (ns_auc))
            print('kNN: ROC AUC=%.3f' % (kNN_auc))

            f.write('Random Chance: ROC AUC=%.3f \n' % (ns_auc))
            f.write('kNN: ROC AUC=%.3f \n' % (kNN_auc))

            ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
            kNN_fpr, kNN_tpr, _ = roc_curve(y_test, kNN_probs)

            # plot the roc curve for the model
            plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Chance')
            plt.plot(kNN_fpr, kNN_tpr, marker='.', label='kNN')

            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # show the legend
            plt.legend()
            # show the plot
            plt.savefig(paths[21]+'/ROC_kNN.png',dpi=300, bbox_inches = "tight")
            # plt.show()
            plt.close()

            axes.plot(kNN_fpr, kNN_tpr, marker='.', label='kNN', color='orange')
            axes.set_xlabel('False Positive Rate')
            axes.set_ylabel('True Positive Rate')
            axes.legend(loc = 'lower right')

            figure.savefig(paths[9] + '/Collective_Test_ROC.png', dpi=300, bbox_inches="tight")
            # figure.show()
            # figure.close()
            print("***********************************************************************************************")

            f.write("***********************************************************************************************")

            f.close()
        A.close()

#
# path = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_T_1_Test/Test/1"
# pathFile = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_T_1_Test/Test"
# pathModel = "/media/keagan/Digitide 1TB/Keagan/ZETA/X_T_1_Train/Test/1"
#
# path = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_T_1_Test/Test/1"
# pathFile = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_T_1_Test/Test"
# pathModel = "/media/keagan/Digitide 1TB/Keagan/ZETA/X_T_1_Train/Test/1"

# path = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_T_0_Test/Test/1"
# pathFile = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_T_0_Test/Test"
# pathModel = "/media/keagan/Digitide 1TB/Keagan/ZETA/X_T_0_Train/Test/1"
#
# "/media/keagan/Digitide 1TB/Keagan/ZETA/X_ES_T_1"
# "/media/keagan/Digitide 1TB/Keagan/ZETA/X_ES_T_0"

# path = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_ES_H_1_final/Test/1"
# pathFile = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_ES_H_1_final/Test"
# pathModel = "/media/keagan/Digitide 1TB/Keagan/ZETA/X_ES_H_1_final/Test/1"
#
# MainMethod(path, pathModel, pathFile)
# #
# path = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_ES_T_0/Test/1"
# pathFile = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_ES_T_0/Test"
# pathModel = "/media/keagan/Digitide 1TB/Keagan/ZETA/X_ES_T_1/Test/1"
# MainMethod(path, pathModel, pathFile)



# path = "/media/keagan/Digitide 1TB/FINAL/ES_H_1/Y_ES_H_1/Test/1" #To dataset
# pathFile = "/media/keagan/Digitide 1TB/FINAL/ES_H_1/Y_ES_H_1" #To save the results
# pathModel = "/media/keagan/Digitide 1TB/FINAL/ES_H_1/X_ES_H_1/Test/1" #To the trained model
# MainMethod(path, pathModel, pathFile)
'''
# path = "/media/keagan/Digitide 1TB/FINAL/ES_T_1/Y_ES_T_1/Test/1" #To dataset
# pathFile = "/media/keagan/Digitide 1TB/FINAL/MultimodelTest/Y_ES_T_1" #To save the results
# pathModel = "/media/keagan/Digitide 1TB/FINAL/ES_T_1/X_ES_T_1/Test/1"
# MainMethod(path, pathModel, pathFile)

path = "/media/keagan/Digitide 1TB/FINAL/H_1/Y_H_1/Test/1" #To dataset
pathFile = "/media/keagan/Digitide 1TB/FINAL/MultimodelTest/Y_H_1" #To save the results
pathModel = "/media/keagan/Digitide 1TB/FINAL/H_1/X_H_1/Test/1"
MainMethod(path, pathModel, pathFile)

path = "/media/keagan/Digitide 1TB/FINAL/T_1/Y_T_1/Test/1" #To dataset
pathFile = "/media/keagan/Digitide 1TB/FINAL/MultimodelTest/Y_T_1" #To save the results
pathModel = "/media/keagan/Digitide 1TB/FINAL/T_1/X_T_1/Test/1"
MainMethod(path, pathModel, pathFile)


# path = "/media/keagan/Digitide 1TB/FINAL/T_1/Y_T_1/Test/1" #To dataset
# pathFile = "/media/keagan/Digitide 1TB/FINAL/T_1/Y_T_1" #To save the results
# pathModel = "/media/keagan/Digitide 1TB/FINAL/T_1/X_T_1/Test/1"
# MainMethod(path, pathModel, pathFile)
'''

###########################################################################################

# path = "/media/keagan/Digitide 1TB/Compare Models/Test/1" #To dataset
# pathFile = "/media/keagan/Digitide 1TB/" #To save the results
# pathModel = "/media/keagan/Digitide 1TB/FINAL/ES_T_1/X_ES_T_1/Test/1"
# MainMethod(path, pathModel, pathFile)

# path = "/media/keagan/Digitide 1TB/Compare Models/Test/1" #To dataset
# pathFile = "/media/keagan/Digitide 1TB/Compare Models/Test/X_H" #To save the results
# pathModel = "/media/keagan/Digitide 1TB/FINAL/H_1/X_H_1/Test/1"
# MainMethod(path, pathModel, pathFile)
#
# path = "/media/keagan/Digitide 1TB/Compare Models/Test/1" #To dataset
# pathFile = "/media/keagan/Digitide 1TB/Compare Models/Test/ES_H" #To save the results
# pathModel = "/media/keagan/Digitide 1TB/FINAL/ES_H_1/X_ES_H_1/Test/1"
# MainMethod(path, pathModel, pathFile)
#
# path = "/media/keagan/Digitide 1TB/Compare Models/Test/1" #To dataset
# pathFile = "/media/keagan/Digitide 1TB/Compare Models/Test/X_T" #To save the results
# pathModel = "/media/keagan/Digitide 1TB/FINAL/T_1/X_T_1/Test/1"
# MainMethod(path, pathModel, pathFile)

########################################## 2.1

# path = "/media/keagan/Digitide 1TB/FINAL/H_1/Y_H_1/Test/1" #To dataset
# pathFile = "/media/keagan/Digitide 1TB/FINAL/MultimodelTest/2point1/H_1_TestedOn_T_1" #To save the results
# pathModel = "/media/keagan/Digitide 1TB/FINAL/T_1/X_T_1/Test/1"
# MainMethod(path, pathModel, pathFile)
#
# path = "/media/keagan/Digitide 1TB/FINAL/T_1/Y_T_1/Test/1" #To dataset
# pathFile = "/media/keagan/Digitide 1TB/FINAL/MultimodelTest/2point1/T_1_TestedOn_H_1" #To save the results
# pathModel = "/media/keagan/Digitide 1TB/FINAL/H_1/X_H_1/Test/1"
# MainMethod(path, pathModel, pathFile)










