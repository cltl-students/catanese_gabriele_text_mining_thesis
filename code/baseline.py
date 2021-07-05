""" Script that runs the Majority Baseline and evaluates its performance """

from sklearn.metrics import classification_report
import pandas as pd
import argparse


def read_data(data_path):
    """ 
    Reads in the .csv test set.
    Returns three lists: the sentences, the gold AC, and the gold SP
    """

    test_df = pd.read_csv(data_path, delimiter=';', 
    header= 0, dtype= str, keep_default_na=False, encoding= 'utf-8')

    sentences = test_df['Sentence'].astype(str)
    gold_aspects = test_df['Aspect_Category'].astype(str)
    gold_sentiment = test_df['Sentiment'].astype(str)

    return sentences, gold_aspects, gold_sentiment

def assign_labels(sentences):
    """ 
    Assigns the majority class label to each sentence for each task.
    Returns two lists: predicted AC and predicted SP 
    """

    predictions_category = []
    predictions_sentiment = []

    for sentence in sentences:
        prediction_cat = 'NA'
        predictions_category.append(prediction_cat)

        prediction_sent = 'Neutral'
        predictions_sentiment.append(prediction_sent)
    
    return predictions_category, predictions_sentiment


def evaluate_performance(gold_aspects, predictions_category, gold_sentiment, predictions_sentiment):
    """
    Prints a classification report regarding the performance 
    of the Majority Baseline on both ACD and SP. 
    It saves the output for each task to an output file
    """

    # Create the evaluation report for ACD.
    print()
    print('CLASSIFICATION REPORT FOR ASPECT CATEGORY:')
    print()
    evaluation_report = classification_report(gold_aspects, predictions_category)

    # Show the evaluation report.
    print()
    print(evaluation_report)
    print('_________________________________________________________________')
    print()
    print('CLASSIFICATION REPORT FOR SENTIMENT:')
    print()
    #save
    evaluation_report = classification_report(gold_aspects, predictions_category, output_dict=True)
    clsf_report = pd.DataFrame(evaluation_report).transpose()
    clsf_report.to_excel('Baseline_Classification_Report_a.xlsx', index= True)

    # sentiment
    evaluation_report2 = classification_report(gold_sentiment, predictions_sentiment)
    # Show the evaluation report.
    print(evaluation_report2)
    # save
    evaluation_report2 = classification_report(gold_sentiment, predictions_sentiment, output_dict=True)
    clsf_report2 = pd.DataFrame(evaluation_report2).transpose()
    clsf_report2.to_excel('Baseline_Classification_Report_s.xlsx', index= True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('test_path', help='file path to the .csv test set.')
    args = parser.parse_args()


    feedbacks, g_categories, g_sentiment = read_data(args.test_path)
    p_categories, p_sentiment = assign_labels(feedbacks)
    evaluate_performance(g_categories, p_categories, g_sentiment, p_sentiment)



if __name__ == '__main__':
    main()