def normalize(token):
    penn_tokens = {
        '-LRB-': '(',
        '-RRB-': ')',
        '-LSB-': '[',
        '-RSB-': ']',
        '-LCB-': '{',
        '-RCB-': '}'
    }

    if token in penn_tokens:
        return penn_tokens[token]

    token = token.lower()
    try:
        int(token)
        return "<NUM>"
    except:
        pass
    try:
        float(token.replace(',', ''))
        return "<FLOAT>"
    except:
        pass
    return token

Vocabulary = set()
file_voc = open('words.voc.conll2009', 'r')
for line in file_voc.readlines():
    word = line.strip()
    Vocabulary.add(word)

file_in_1 = open('news.en-00001-of-00100', 'r')
file_out = open('1BilionBenchMark', 'w')
idx = 0

for line in file_in_1.readlines():
    sents = line.strip().split()
    if len(sents) > 60 or len(sents) < 15:
        continue
    words = [normalize(w) for w in sents]
    all_in = True
    for w in words:
        if w not in Vocabulary:
            all_in = False
            break
    if all_in:
        sent = ' '.join(words)
        file_out.write(sent)
        file_out.write('\n')
        idx += 1

file_in_2 = open('news.en-00002-of-00100', 'r')

for line in file_in_2.readlines():
    sents = line.strip().split()
    if len(sents) > 60 or len(sents) < 15:
        continue
    words = [normalize(w) for w in sents]
    all_in = True
    for w in words:
        if w not in Vocabulary:
            all_in = False
            break
    if all_in:
        sent = ' '.join(words)
        file_out.write(sent)
        file_out.write('\n')
        idx += 1

file_in_3 = open('news.en-00003-of-00100', 'r')

for line in file_in_3.readlines():
    sents = line.strip().split()
    if len(sents) > 60 or len(sents) < 15:
        continue
    words = [normalize(w) for w in sents]
    all_in = True
    for w in words:
        if w not in Vocabulary:
            all_in = False
            break
    if all_in:
        sent = ' '.join(words)
        file_out.write(sent)
        file_out.write('\n')
        idx += 1


print(idx)
file_in_1.close()
file_in_2.close()
file_out.close()