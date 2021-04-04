import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io


# intialize dicts
total, results = {}, {}
for sz in range(1, 5):
	for contx  in range(0, 4):
		cond = (sz, contx)
		results[cond], total[cond] = 0, 0



scores_all = ["individual_scores_epoch_10_exp_J.json", "individual_scores_epoch_10_exp_I.json"]
annotations_all = ["test_annotations_exp_J.json", "test_annotations_exp_I.json"]

for ind in range(2):
	scores, annotations = scores_all[ind], annotations_all[ind]

	with open(scores) as f:
		data = json.load(f)
		scores_json = data

	with open(annotations) as f:
		data = json.load(f)
		annotations_json = data


	id_to_file_name = {}
	for img in annotations_json['images']:
		id_to_file_name[img['id']] = img['file_name']


	for individual_score in scores_json:
		img_id, true_prediction, _, real_label, _, predicted_label = individual_score
		file_name = id_to_file_name[img_id]


		_, sz, _, _, _, _, type_img = file_name.strip('.jpg').split('_')
		if ind == 1:
			if type_img == '8':
				type_img = '0'
			elif type_img == '2':
				type_img = '3'
			else:
				continue

		cond = (int(sz), int(type_img))
		results[cond] += float(true_prediction)
		total[cond] += 1.0


sem = {}
for cond in total:

	accuracy_list = [1.]*int(results[cond]) + [0.]*int(total[cond]-results[cond])
	sem[cond] = np.std(accuracy_list)/np.sqrt(total[cond])
	results[cond] /= total[cond]

results_mean1 = np.array([results[(1, 0)], results[(2, 0)], results[(3, 0)], results[(4, 0)]]) 
results_mean2 = np.array([results[(1, 1)], results[(2, 1)], results[(3, 1)], results[(4, 1)]]) 
results_mean3 = np.array([results[(1, 2)], results[(2, 2)], results[(3, 2)], results[(4, 2)]]) 
results_mean4 = np.array([results[(1, 3)], results[(2, 3)], results[(3, 3)], results[(4, 3)]])
results_mean = np.concatenate((results_mean1, results_mean2, results_mean3, results_mean4), axis=0)

results_sem1 = np.array([sem[(1, 0)], sem[(2, 0)], sem[(3, 0)], sem[(4, 0)]]) 
results_sem2 = np.array([sem[(1, 1)], sem[(2, 1)], sem[(3, 1)], sem[(4, 1)]]) 
results_sem3 = np.array([sem[(1, 2)], sem[(2, 2)], sem[(3, 2)], sem[(4, 2)]]) 
results_sem4 = np.array([sem[(1, 3)], sem[(2, 3)], sem[(3, 3)], sem[(4, 3)]])
results_sem = np.concatenate((results_sem1, results_sem2, results_sem3, results_sem4), axis=0)

scipy.io.savemat('resultMat.mat', {'mean': results_mean , 'sem':results_sem})

plotdata = pd.DataFrame({
	"Full context":[results[(1, 0)], results[(2, 0)], results[(3, 0)], results[(4, 0)]],
    "Congruent context":[results[(1, 1)], results[(2, 1)], results[(3, 1)], results[(4, 1)]],
    "Incongruent context":[results[(1, 2)], results[(2, 2)], results[(3, 2)], results[(4, 2)]],
    "Minimal context":[results[(1, 3)], results[(2, 3)], results[(3, 3)], results[(4, 3)]]
    }, 
    index=["1", "2", "4", "8"]
)

errors = pd.DataFrame({
	"Full context":[sem[(1, 0)], sem[(2, 0)], sem[(3, 0)], sem[(4, 0)]],
    "Congruent context":[sem[(1, 1)], sem[(2, 1)], sem[(3, 1)], sem[(4, 1)]],
    "Incongruent context":[sem[(1, 2)], sem[(2, 2)], sem[(3, 2)], sem[(4, 2)]],
    "Minimal context":[sem[(1, 3)], sem[(2, 3)], sem[(3, 3)], sem[(4, 3)]]
    }, 
    index=["1", "2", "4", "8"]
)

plotdata.plot(kind="bar", color=['black', '#deebfb', '#6379a2', 'white'], edgecolor = 'black', yerr = errors, capsize = 2)
plt.title("DenseNet")
plt.xlabel("Size (degrees)")
plt.ylabel("Accuracy")
plt.show()


'''
path_prefix = 'accuracy_json/'

train_accuracies, test_accuracies = [], []
train_data, test_data = [], []



for i in range(1, 21):
	test_file_name = path_prefix + 'test' + '_accuracies_epoch_' + str(i) + '.json'
	train_file_name = path_prefix + 'train' + '_accuracies_epoch_' + str(i) + '.json' 

	with open(train_file_name) as f:
		data = json.load(f)
		train_data.append(data)

	with open(test_file_name) as f:
		data = json.load(f)
		test_data.append(data)

for i in range(len(test_data)):
	train_accuracies.append(train_data[i]['total_accuracy'])
	test_accuracies.append(test_data[i]['total_accuracy'])

print(np.array(train_data[19]['distributions']).shape)

pprint(train_accuracies)
print(test_accuracies)

plt.plot(range(1, len(train_accuracies)+1), train_accuracies)
plt.ylabel('train accuracy')
plt.xlabel('epoch')
plt.title('Train accuracy')
plt.show()


plt.plot(range(1, len(test_accuracies)+1), test_accuracies)
plt.ylabel('test accuracy')
plt.xlabel('epoch')
plt.title('Test accuracy')
plt.show()


'''
