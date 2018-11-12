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


file_in = open('news.en-00001-of-00100', 'r')
file_in_2 = open('news.en-00001-of-00200', 'r')
file_out = open('1BilionBenchMark', 'w')

for line in file_in.readlines():
    sents = line.strip().split()
    if len(sents) > 100 or len(sents) < 10:
        continue
    sent = ' '.join([normalize(w) for w in sents])
    file_out.write(sent)
    file_out.write('\n')

file_in.close()
file_out.close()