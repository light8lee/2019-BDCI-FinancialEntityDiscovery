from proj_utils.files import create_embedding
import pickle

def propare_embeddings_from_pretrained(input_embed_filename, output_embed_filename, output_vocab_filename):
    additional_words = [
        '[PAD]',
        '[UNK]',
        '[CLS]',
        '[SEP]',
        '[MASK]',
        '<S>',
        '<T>',
    ]

    embeddings, vocabs = create_embedding(additional_words, input_embed_filename)
    pickle.dump(embeddings, open(output_embed_filename, 'wb'))
    with open(output_vocab_filename, 'w') as f:
        vocabs = [v+'\n' for v in vocabs]
        f.writelines(vocabs)


if __name__ == '__main__':
    propare_embeddings_from_pretrained('./data/cn_char_embedding.txt', './data/med.vec', './data/med.vocab')