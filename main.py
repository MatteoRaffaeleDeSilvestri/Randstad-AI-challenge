import csv
import pickle
from sklearn import svm
from os.path import isfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score

def dataset_reader(dataset_file_path, train):
    
    # Open dataset
    with open(dataset_file_path) as data_file:
        data = csv.DictReader(data_file)

        # Data preparation
        data_job_offer = list()
        data_label = list()
        
        for record in data:

            record_job_offer = record['Job_offer']
            
            # This operation is performed on train dataset only (check out the documentation for more details)
            if train:
                record_job_offer = record_job_offer.replace('. ', '').replace('.', '')
            
            data_job_offer.append(record_job_offer)
            data_label.append(record['Label'])
        
    return data_job_offer, data_label

def save_vectorizer(vectorizer):
    
    # Save model
    with open('vectorizer.pickle', 'wb') as vector:
        pickle.dump(vectorizer, vector)

def save_model(classifier):
    
    # Save model
    with open('model.pickle', 'wb') as model:
        pickle.dump(classifier, model)

def output_file_generator(classifier, test_data_label, test_data_job_offer, test_data_job_offer_vectorized):
    
    # Create csv document
    with open('output.csv', 'w') as output:
        
        # Create csv writer
        writer = csv.writer(output, delimiter=';')
        
        # Set csv fields
        fields = ('Job_description', 'Label_true', 'Label_pred')
        writer.writerow(fields)

        # Generate and write data for every record in the given dataset 
        for i in range(len(test_data_job_offer)):
            record = (test_data_job_offer[i], test_data_label[i], classifier.predict(test_data_job_offer_vectorized[i])[0])
            writer.writerow(record)

def metrics_output(test_data_label, test_data_job_offer_vectorized):
    
    # Metrics calculation
    precision = precision_score(test_data_label, classifier.predict(test_data_job_offer_vectorized), average=None, labels=['Java Developer', 'Web Developer', 'Programmer', 'System Analyst', 'Software Engineer'])
    recall = recall_score(test_data_label, classifier.predict(test_data_job_offer_vectorized), average=None, labels=['Java Developer', 'Web Developer', 'Programmer', 'System Analyst', 'Software Engineer'])
    f1 = f1_score(test_data_label, classifier.predict(test_data_job_offer_vectorized), average=None, labels=['Java Developer', 'Web Developer', 'Programmer', 'System Analyst', 'Software Engineer'])
    
    # Output
    print('\n-------------------------------------------------------------------------------------------------------------------------------------------')
    print('\t\t\tJAVA DEVELOPER\t\tWEB DEVELOPER\t\tPROGRAMMER\t\tSYSTEM ANALYST\t\tSOFTWARE ENGINEER')
    print('-------------------------------------------------------------------------------------------------------------------------------------------')
    print('Precision\t\t{}\t{}\t{}\t{}\t{}'.format(precision[0], precision[1], precision[2], precision[3], precision[4]))
    print('Recall\t\t\t{}\t{}\t{}\t\t\t{}\t{}'.format(recall[0], recall[1], recall[2], recall[3], recall[4]))
    print('F1-score\t\t{}\t{}\t{}\t{}\t{}'.format(f1[0], f1[1], f1[2], f1[3], f1[4]))
    print('-------------------------------------------------------------------------------------------------------------------------------------------\n')

if __name__ == '__main__':
    
    # Data preparation
    train_data_job_offer, train_data_label = dataset_reader('train_set.csv', True)
    test_data_job_offer, test_data_label = dataset_reader('test_set.csv', False)

    # Data vectorization
    vectorizer = TfidfVectorizer(min_df=0, norm='l1')
    train_data_job_offer_vectorized = vectorizer.fit_transform(train_data_job_offer)
    test_data_job_offer_vectorized = vectorizer.transform(test_data_job_offer)
    
    # NOT REQUIRED BY THE CHALLANGE
    # Save vectorizer 
    # if not isfile('vectorizer.pickle'):
    #     save_vectorizer(vectorizer)

    # Use existing model or generate and save it (if not already saved)
    if isfile('model.pickle'):
        
        # Use existing model
        classifier = pickle.load(open('model.pickle', 'rb'))
    
    else:

        # Classifier setup and training
        classifier = svm.SVC(C=10, gamma=25, class_weight='balanced') 
        classifier.fit(train_data_job_offer_vectorized, train_data_label)

        # Generate model file (if not already present)    
        if not isfile('model.pickle'):
            save_model(classifier)

    # Generate output file (if not already generated)
    if not isfile('output.csv'):
        output_file_generator(classifier, test_data_label, test_data_job_offer, test_data_job_offer_vectorized)

    # Print metrics values
    metrics_output(test_data_label, test_data_job_offer_vectorized)
