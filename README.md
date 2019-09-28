# fake_Sparse_Composite_Document_Vectors
make "Sparse Composite Document Vectors", use latent dirichlet allocation instead of kmeans

### なんで作ったの？
教師ラベルのない文書群を扱うときに、文の意味をある程度それらしいベクトルにしたかった
弊社内の人に、"教師なし機械学習が優れてるわけじゃなくって、人間の知見を注入できるほどデータセットを整備できないので仕方なく教師なしなのであって、性能でないよ？"と主張しても、理解が得られる雰囲気すら醸されないので、教師なしである程度戦えるものが必要っぽかった(けど業務でやってる暇がないので自宅で)

### コレは何？
- 単語分散表現(from fasttext)を、文書群のトピック(from LDA)の所属確率を使って重み付けしたベクトルを平均することで文ベクトルとした
- 基本的には[Sparse Composite Documents Vector](https://arxiv.org/pdf/1612.06778.pdf)やら[gaussian LDA](https://www.aclweb.org/anthology/P15-1077)のオマージュ

### 理論背景(読み飛ばし)
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

### usage
```python
import fasttext
import sentencepiece as spm
from expand_vector import SentenceEmbedding
import joblib

# load models
# ## if you do not have enough knowledge, use my repositories.
# ## usually, i use 'joblib' for saving python data.
# ## however, any ways are ok if it can load gensim LDA model and corpora.dictionary.Dictionary
model_sentence = spm.SentencePieceProcessor()
model_sentence.load("path_to_spm/model_sentence_piece.model")
model_fast = fasttext.load_model("path_to_fast/model_fast_text.bin")
model_lda = joblib.load('path_to_lda/model_of_gensim_LDA.joblib')
dictionary_lda = joblib.load("path_to_lda/dictionary_of_gensim_Dictionary.joblib")

# init and make token vector
sentence_embedder = SentenceEmbedding(model_lda_gensim=model_lda, model_fast_text=model_fast, model_sentence_piece=model_sentence, dictionary_lda=dictionary_lda)

# input documents
docunmtens = ["マイクロ化学デバイスおよび解析装置\n本発明は、マイクロ化学デバイスおよび解析装置に係り、特に、細胞を保持するマイクロウエルが多数形成されたマイクロ化学デバイスおよび解析装置に関する。", ...]

# document -> vector
vector, flag_exist_valid_token = sentence_embedder.sentence2vector(documents[0])  # [np.array, bool]

# documents -> vectors
# ## 文書群documentsを文の意味ベクトルに変換して格納　＆　似ている文書探索用の樹形図探索アルゴリズムのインデックス作成
sentence_embedder.vectorize_collected_documents(documents=documents)
sentence_embedder.vectors_collected[0]  # np.array, vector of documents[0]

# search similar documents
# ## 入力文字列と意味の似ている文書topn個を抽出する
flag_exists_valid_token, similar_docs, ids, distances = sentence_embedder.fetch_similar_doc_and_info("細胞の検査をする", topn=10)
# ## flag_exists_valid_token: 入力文字列に少なくとも1つ、意味を獲得できている単語が含まれたらTrue, 無いとFalse(他の戻り値は無意味になる)
# ## similar_docs: 抽出した文書の文字列
# ## ids: 抽出した文書のid(入力documentsの戦闘から何番目だったか？)
# ## distances: cosine distance, 似ているほど0に近づく。経験的に0.25を超えたら似ていないように見える
```
