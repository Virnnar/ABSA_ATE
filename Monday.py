import process.file2vec as f2v

if __name__ == '__main__':
    path = {"train": r"D:\\Code\\ABSA\\data\\train.raw", "test": "__"}
    data = f2v.file2data(path['train'])

    vocab, idx_data, embedding = f2v.handle(data)

    




