file = open('conll2009.train', 'r')
file2 = open('conll2009_batch.train', 'w')
idx = 1

for line in file.readlines():
    file2.write(line)
    if idx == 178800:
        break
    idx += 1
print(idx)
file.close()
file2.close()


file = open('conll2009.dev', 'r')
file2 = open('conll2009_batch.dev', 'w')
idx = 1

for line in file.readlines():
    file2.write(line)
    if idx == 6360:
        break
    idx += 1
print(idx)
file.close()
file2.close()
