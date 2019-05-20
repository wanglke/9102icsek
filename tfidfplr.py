import numpy as np
import pandas as pd

# 对句子进行分词，并取掉换行
def tokenizer(text):

    '''Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [(document.replace('\n', '')).lower().split() for document in text]
    for document in text:
        (document.replace('\n', '')).lower().split()
    return np.array(text)
# 加载文件
def loadfile():
    # neg = pd.read_excel('../data/neg.xls', sheet_name=0, header=None, index=None)
    # pos = pd.read_excel('../data/pos.xls', sheet_name=0, header=None, index=None)
    train_data = pd.read_csv('data/train.csv', delimiter=',', low_memory=False, lineterminator="\n")
    test = pd.read_csv('data/20190513_test.csv', delimiter=',', low_memory=False, lineterminator="\n")
    y = list(train_data.label)
    a = []
    for i in y:
        if i == 'Negative':
            a.append(0)
        else:
            a.append(1)
    y = a
    combined = train_data.review
    train_len = len(combined)
    # combined = np.concatenate((pos, neg))
    #
    test = test.loc[test.ID != 0, 'review']
    combined = np.concatenate((combined, test))
    # y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))
    y = np.array(y)
    combined = np.array(combined.tolist())
    return combined, y, test, train_len


def train():
    print('Loading Data...')
    combined0, y, test, train_len = loadfile()
    # combined = tokenizer(combined0)
    # train_len = len(combined0)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_df=0.4)
    tfidf_matrix = tfidf.fit_transform(combined0)
    X = tfidf_matrix[:train_len]
    test = tfidf_matrix[train_len:]
    print('词表大小:', len(tfidf.vocabulary_))
    print(X.shape)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import roc_auc_score
    LR = LogisticRegression()
    MNB = MultinomialNB()
    sfd = StratifiedKFold(n_splits=10)
    sfd.get_n_splits(X, y)
    predicts = []
    test_splits = [340, 270, 270, 280, 265, 200, 380, 172, 265, 270]
    count = 0
    for train_index, test_index in sfd.split(X, y):
        X_train = X[train_index]
        Y_train = y[train_index]
        x_test = X[test_index]
        y_test = y[test_index]
        MNB.fit(X_train, Y_train)
        print(roc_auc_score(y_test, MNB.predict_proba(x_test)[:, 1]))
        flag = test_splits[count]
        predicts.append(MNB.predict_proba(test[:flag])[:, 1])
        test = test[flag:]
        count += 1
    predicts = [y for x in predicts for y in x]
    print(len(predicts))
    result = pd.DataFrame({
        'ID': list(range(1, len(combined0)-train_len + 1)),
        'Pred': predicts
    })
    result
    result.to_csv('./submitL.csv', index=False)
    print("Done!")




if __name__ == '__main__':
    train()
    test1 = pd.read_csv('data/20190513_test.csv', delimiter=',', low_memory=False, lineterminator="\n")