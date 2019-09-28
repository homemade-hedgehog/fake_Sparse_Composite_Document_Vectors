import joblib
from tqdm import tqdm
import numpy as np
import sentencepiece as spm
import fasttext
from gensim.models import LdaMulticore
from gensim.corpora.dictionary import Dictionary
import nmslib
import time
import hashlib
import os


def aggregate_adopted_vocab(model_lda_gensim: LdaMulticore, model_fast_text: fasttext.FastText._FastText) -> set:
    """
    言語モデルで意味が獲得できている語に限定する
    :param model_lda_gensim: LdaMulticore, latent dirichlet allocation model
    :param model_fast_text: fasttext, word embedding model
    :return: set, valid vocabulary set
    """
    vocab_lda = set(model_lda_gensim.id2word.values())
    return vocab_lda.intersection(model_fast_text.words)


def leave_valid(tokens: list, dict_is_valid: dict) -> list:
    """
    指定された語だけ残す
    :param tokens: list of str, tokenのlist
    :param dict_is_valid: dict, {valid_token: True}
    :return: list of str
    """
    list_is_valid = [dict_is_valid.get(token, False) for token in tokens]
    return np.array(tokens, dtype=object)[list_is_valid]


def topic_distribution2vector(distribution: list, num_topics: int) -> np.array:
    """
    LDA(latent dirichlet allocation)のtopic mixtureをベクトルに変換
    :param distribution: list of tuple, [(topic_id, mixture), ...]
    :param num_topics: int, num_topics of your LDA model
    :return:
    """
    array = np.zeros(num_topics)
    for _id, mixture in distribution:
        array[_id] = mixture
    return array


def lda_token2vector(token: str, model_lda_gensim: LdaMulticore, dictionary_lda: Dictionary) -> np.array:
    """
    tokenをLDAのトピック分布を示すベクトルに変換
    :param token: str, LDAモデルで意味が獲得されているtoken
    :param model_lda_gensim: LdaMulticore, LDAモデル
    :param dictionary_lda: Dictionary, LDA構築時に使ったtoken -> token_idの辞書
    :return:
    """
    num_topics = model_lda_gensim.num_topics
    topic_vector = topic_distribution2vector(distribution=model_lda_gensim[dictionary_lda.doc2bow([token, ])],
                                             num_topics=num_topics)
    return topic_vector


def expand_token_vector(token: str, model_lda_gensim: LdaMulticore, model_fast_text: fasttext.FastText._FastText,
                        dictionary_lda: Dictionary) -> np.array:
    """
    SCDVの理論と説明は以下を参照
    [本家](https://dheeraj7596.github.io/SDV/)
    [実験記事](https://qiita.com/fufufukakaka/items/a7316273908a7c400868)

    単語分散表現の考案によって、単語の意味を示すベクトルが獲得て着るようになった
    しかし、これは単語レベルであり、文書の意味を示すベクトルの獲得は2019年でも混迷の時代である
    2018年にはbidirectional LSTM を2つ使ったbertという文のベクトル化手法が一世を風靡しているが、
    LSTMは長い系列の初めの方は覚えていないという問題がある。
    また、特に、なんらかの評価付き文書群が手に入らない場合には、何を基準に学習するのかという問題がある。
    SCDVは2017年くらいに発表されてイマイチ市民権を得る前にbertが始まってしまった不遇な手法。
    評価付き文書群がなくても文のベクトルを獲得できる。
    SCDVは単語分散表現空間でのまとまりで重み付けして単語分散表現を拡張するが、
    文書の話題の特徴を加味せずに議論するのが気持ち悪い。
    ここでは、文書の話題を加味しうるLDAを用いて同様のことを実施する（自分で発案した理屈なのである程度怪しい）
    この系譜はgaussian LDAなどにも受け継がれているので、そっちを使っても良いかもしれない
    :param token:
    :param model_lda_gensim: LdaMulticore, LDAモデル
    :param model_fast_text:fasttext, word embedding model
    :param dictionary_lda: Dictionary, LDA構築時に使ったtoken -> token_idの辞書
    :return:
    """
    dimension = model_fast_text.get_dimension()
    num_topics = model_lda_gensim.num_topics

    vector_fast = model_fast_text[token]
    vector_lda = lda_token2vector(token=token, model_lda_gensim=model_lda_gensim, dictionary_lda=dictionary_lda)
    vector = vector_lda.reshape(-1, 1).dot(vector_fast.reshape(1, -1)).flatten()
    return vector


def make_dict_token2vector(tokens: list, model_lda_gensim: LdaMulticore, model_fast_text: fasttext.FastText._FastText,
                           dictionary_lda: Dictionary):
    """

    :param tokens:
    :param model_lda_gensim:
    :param model_fast_text:
    :param dictionary_lda:
    :return:
    """
    dict_token2vector = {
        token: expand_token_vector(token=token, model_lda_gensim=model_lda_gensim, model_fast_text=model_fast_text,
                                   dictionary_lda=dictionary_lda) for token in tokens}
    return dict_token2vector


def wrapper_make_dict_token2vector(model_lda_gensim: LdaMulticore, model_fast_text: fasttext.FastText._FastText,
                                   dictionary_lda: Dictionary) -> dict:
    """
    lda, fasttext言語モデルで意味が獲得できている共通の語に対して、単語分散表現をLDAで拡張したベクトルの辞書を返す
    拡張は、単語分散表現とLDAのトピック分布の行列席のflat tensor
    単語分散表現を各話題ごとに重み付けする
    基本的にLDAは少数のトピックに対して分布を持つ傾向があるので、関係ない話題は重み0になり、メリハリの効いたベクトルになる
    :param model_lda_gensim: LdaMulticore, LDAモデル
    :param model_fast_text: fasttext, word embedding model
    :param dictionary_lda: Dictionary, LDA構築時に使ったtoken -> token_idの辞書
    :return:
    """
    # TODO トピック分布が平均的に値を持つ or 文意に関係ない語の重みがソコソコあって積み重なると馬鹿にならない場合は除外する
    vocab = aggregate_adopted_vocab(model_lda_gensim=model_lda_gensim, model_fast_text=model_fast_text)
    dict_token2vector = make_dict_token2vector(tokens=vocab, model_lda_gensim=model_lda_gensim,
                                               model_fast_text=model_fast_text,
                                               dictionary_lda=dictionary_lda)
    return dict_token2vector


def sentence2vector(sentence, model_sentence_piece: spm.SentencePieceProcessor, dict_token2vector: dict,
                    dict_is_valid: dict) -> [np.array, bool]:
    """
    文をベクトルに変換
    戻り値の第2引数は、意味を獲得できたベクトルが１つもないときで分岐
    - True: 意味を獲得できたtokenがあった
    - False: なかった
    :param sentence: str, 対象の文 
    :param model_sentence_piece: spm.SentencePieceProcessor, sentencepieceモデル
    :param dict_token2vector: dict, token から拡張単語分散表現を得る辞書
    :param dict_is_valid: dict, token からモデルに含まれるかTrueで返す
    :return: 
    """
    tokens_raw = model_sentence_piece.EncodeAsPieces(sentence)
    tokens = leave_valid(tokens=tokens_raw, dict_is_valid=dict_is_valid)
    vector_size = len(list(dict_token2vector.values())[0])
    # 意味を獲得できたtokenが１つもないときは、原点に集めるため、zerosで初期化
    vector = np.zeros((vector_size, max(len(tokens), 1)), dtype=np.float64)
    for i, token in enumerate(tokens):
        vector[:, i] = dict_token2vector[token]

    # いちいちifで分岐しなくてもvector自体は正しく返せるけど、エラーメッセージ表示用に分ける
    # ## exceptionを発行して止めるかどうかが悩ましい
    if len(tokens) != 0:
        # valid token exists
        return [vector.mean(axis=1), len(tokens) != 0]
    else:
        # no valid token
        # assert len(tokens) != 0, "no valid token, change phrase or word"
        print(f" this sentence has no valid token, change phrase or word\n{''.join(tokens_raw[:20]).replace('_', '')}")
        return [vector.mean(axis=1), len(tokens) != 0]


class SentenceEmbedding:
    def __init__(self, model_lda_gensim: LdaMulticore, model_fast_text: fasttext.FastText._FastText,
                 model_sentence_piece: spm.SentencePieceProcessor,
                 dictionary_lda: Dictionary, path_to_save=""):
        self.model_LDA = model_lda_gensim
        self.model_fast = model_fast_text
        self.model_sp = model_sentence_piece
        self.dictionary_LDA = dictionary_lda
        self.vocab = aggregate_adopted_vocab(model_lda_gensim=model_lda_gensim, model_fast_text=model_fast_text)
        self.dict_token2vector = make_dict_token2vector(tokens=self.vocab, model_lda_gensim=model_lda_gensim,
                                                        model_fast_text=model_fast_text, dictionary_lda=dictionary_lda)
        self.dict_is_valid = {token: True for token in self.vocab}
        self.vectors_collected = []
        self.docs_collected = []
        self.index = nmslib.init(method="hnsw", space="cosinesimil")  # 樹形図探索アルゴリズム
        self.path_to_save = "placeholder"
        self.set_save_directory(path_to_save)
        joblib.dump(self.vocab, f"{self.path_to_save}SCDV_vocab.joblib")
        joblib.dump(self.dict_token2vector, f"{self.path_to_save}SCDV_dict_token2vector.joblib")
        joblib.dump(self.dict_is_valid, f"{self.path_to_save}SCDV_dict_is_valid.joblib")

    def set_save_directory(self, path_to_save):
        if path_to_save == "":
            path_to_save = f"scdv_{hashlib.sha1(time.ctime().encode()).hexdigest()}/"
        if not path_to_save.endswith("/"):
            path_to_save += "/"
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        self.path_to_save = path_to_save.replace("\\", "/")

    def sentence2vector(self, sentence: str) -> np.array:
        return sentence2vector(sentence=sentence, model_sentence_piece=self.model_sp,
                               dict_token2vector=self.dict_token2vector, dict_is_valid=self.dict_is_valid)

    def vectorize_collected_documents(self, documents: list):
        """
        文書群を対応するベクトルに変換して、格納する
        :param documents: list of str
        :return:
        """
        self.docs_collected = documents
        self.vectors_collected = [self.sentence2vector(sentence=sentence)[0] for sentence in
                                  tqdm(documents, desc="embed @ scdv")]
        self.index.addDataPointBatch(self.vectors_collected)
        self.index.createIndex({"post": 2})  # この引数の意味がどこかに書いてあったらお知らせを

    def fetch_similar_doc_and_info(self, target_sentence: str, topn=10):
        """
        target_sentenceに似た文を格納済みの中から探して、文書, id, コサイン距離を返す
        :param target_sentence: str, query
        :param topn: int, top n docs
        :return:
        """
        target_vector, flag_exists_valid_token = self.sentence2vector(sentence=target_sentence)
        if flag_exists_valid_token:
            # queryの意味が獲得できるとき
            ids, distances = self.index.knnQuery(target_vector, k=topn)
            return flag_exists_valid_token, [self.docs_collected[_id] for _id in ids], ids, distances
        else:
            # queryの意味が獲得できないとき
            return flag_exists_valid_token, [], [], []
