from __init__ import *

class BERTEmbedder:
 
  def __init__(self, bert_model:BertModel, 
               bert_tokenizer:BertTokenizer, 
               device:str):
    """
    Args:
      bert_model (BertModel): The pretrained BERT model.
      bert_tokenizer (BertTokenizer): The pretrained BERT tokenizer.
      token_limit (integer): The maximum number of tokens to give as input to BERT.
      device (string): The device on which BERT should run, either cuda or cpu.
    """
    super(BERTEmbedder, self).__init__()
    self.bert_model = bert_model
    self.bert_model.to(device)
    # modeli eval modunda çalıştırıyoruz embedderi eğitmiyoruz.
    self.bert_model.eval()
    self.bert_tokenizer = bert_tokenizer
    self.device = device
 
  def _prepare_input(self, sentences:List[str]):

    # input_ids: list of sentences, each sentence is a list of tokens
    input_ids = []
    
    # kelimeleri parçara ayırıyor, playing play + ##ing bunları kaydediyor.
    to_merge_wordpieces = []
    
    # BERT requires the attention mask in order to know on which tokens it has to attend to
    # padded indices do not have to be attended to so will be 0
    attention_masks = []

    # next word prediction veya 2 sentence prediction (QA) yaparken kullanılıyormuş
    # biz bunları yapmadığımız için hepsi 0 olacak normalde 0 ve 1 hangi sentence olduğunu gösteriyor
    token_type_ids = []

    # en uzun cümleye göre padding yapıyoruz
    max_len = max([len(self._tokenize_sentence(s)[0]) for s in sentences]) 

    for sentence in sentences:

      # encode olmuş sentence ve kelimeleri merge etmek için indexleri alıyoruz
      encoded_sentence, to_merge_wordpiece = self._tokenize_sentence(sentence)

      # paddlenmiş kelimelere attende etmicez ilk n kelime 1 kalan max-n kelime 0
      # 1 lere attend ediyoruz
      att_mask = [1] * len(encoded_sentence)
      att_mask = att_mask + [0] * (max_len - len(encoded_sentence))

      # we pad sentences shorter than the max length of the batch
      encoded_sentence = encoded_sentence + [0] * (max_len - len(encoded_sentence)) 

      input_ids.append(encoded_sentence)
      to_merge_wordpieces.append(to_merge_wordpiece)
      attention_masks.append(att_mask)
      token_type_ids.append([0] * len(encoded_sentence))
    input_ids = torch.LongTensor(input_ids).to(self.device)
    attention_masks = torch.LongTensor(attention_masks).to(self.device)
    token_type_ids = torch.LongTensor(token_type_ids).to(self.device)
    return input_ids, to_merge_wordpieces, attention_masks, token_type_ids


  def _tokenize_sentence(self, sentence:List[str]):

    # Sentence başına cls token ekliyoruz
    encoded_sentence = [self.bert_tokenizer.cls_token_id]
    # word pieceleri tutmak için list of lists: playing -> ["play","##ing"]
    to_merge_wordpiece = []
    for word in sentence:

      # kelimeyi tokenlara çeviriyoruz
      # bu int yerine tokenlaştırıyor sadce geldim -> gel+dim gibi
      encoded_word = self.bert_tokenizer.tokenize(word)
      
      # kelimenin başladığı ve bittiği indexleri not alıyoruz
      # kelimeşer bölündüğü için play+ing birleştirirken işimize yarayacak
      strting_index = len(encoded_sentence)-1
      end_index = len(encoded_sentence)+len(encoded_word)-1
      to_merge_wordpiece.append([i for i in range(strting_index, end_index)]) 

      # tokenları integerlara çeviriyoruz convert_tokens_to_ids(["play","##ing"]) → [509, 510]
      encoded_sentence.extend(self.bert_tokenizer.convert_tokens_to_ids(encoded_word))

    # sentence sonuna sep token ekliyoruz
    encoded_sentence.append(self.bert_tokenizer.sep_token_id)

    # encode edilmiş cümleyi ve bölünme indexlerini döndürüyoruz
    return encoded_sentence, to_merge_wordpiece


  def embed_sentences(self, sentences:List[str]):
      # we convert the sentences to an input that can be fed to BERT
      input_ids, to_merge_wordpieces, attention_mask, token_type_ids = self._prepare_input(sentences)
      # we set output_all_encoded_layers to True cause we want to sum the representations of the last four hidden layers

      # bo grad çalıştırıyoruz train etmiyoruz (embedder çünkü)
      with torch.no_grad():

        # bert output = (last_hidden_states, pooler_output, hidden_states)
        # last_hidden_states = the sequence of hidden states of the last layer of the model
        # pooler_output = the hidden states of the first token of the sequence (the CLS token)
        # hidden_states = a tuple of FloatTensors, each of shape (batch_size x sequence_length x hidden_size)
        bert_output = self.bert_model(input_ids=input_ids, 
                                              token_type_ids=token_type_ids,
                                              attention_mask=attention_mask,
                                              output_hidden_states=True,
                                              return_dict=True)
        

        # pick the last 4 layers of hidden_states, stack → shape (4, batch, seq, hidden_size)
        last_hidden_states = bert_output[-1]
        layers_to_sum = torch.stack([last_hidden_states[x] for x in [-1, -2, -3, -4]], dim=0)
        summed_layers = torch.sum(layers_to_sum, dim=0)

        # collapse subtoken pieces back to per-word embeddings
        merged_output = self._merge_embeddings(summed_layers, to_merge_wordpieces)
      
      return merged_output
  

 
  # aggregated_layers has shape: shape batch_size x sequence_length x hidden_size
  def _merge_embeddings(self, aggregated_layers:List[List[float]],
                          to_merge_wordpieces:List[List[int]]):
    

    merged_output = []
    # first we remove the [CLS] and [SEP] tokens from the output embeddings
    token_embeddings = aggregated_layers[:, 1:-1 ,:]

    # tokenlarla hangi indexlerde başlayıp bittiklerini alıyoruz
    for sent_embs, merge_idxs in zip(token_embeddings, to_merge_wordpieces):
      word_vectors = []
      # her kelimenin başlayıp bittiği index için
      for idxs in merge_idxs:
          # average all subtoken embeddings for that word
          # BERT often splits a word into several pieces (“playing” → “play” + “##ing”),
          # and each piece gets its own hidden‐state embedding
          # kelimeyi oluşturan parçaların embeddinglerinin meanini alıyoruz
          # bana garip geldi but if it works who am i to judge
          word_vectors.append(torch.mean(sent_embs[idxs], dim=0))
      # stack → Tensor(shape=(num_words, hidden_size))
      merged_output.append(torch.stack(word_vectors).to(self.device))

    return merged_output