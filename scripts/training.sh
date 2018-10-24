#!/usr/bin/env bash
cd ..
python -mnnet.run.srl.run \
--train conll2009_batch.train.dep_l \
--test conll2009_batch.dev.dep_l \
--data_partition dev \
--batch 1 \
--freq-voc freq.voc_unk.conll2009 \
--word-voc words.voc.conll2009 \
--p-word-voc p.words.voc_sskip.conll2009 \
--role-voc labels.voc.conll2009 \
--frame-voc frames.voc.conll2009 \
--pos-voc pos.voc.conll2009 \
--dep-voc dep.voc.conll2009_2 \
--specific-dep-voc Specific_Dep.voc \
--word-embeddings word_embeddings_proper.sskip.conll2009.txt \
--elmo-embeddings-0 elmo_embeddings_1 \
--elmo-embeddings-1 elmo_embeddings_2 \
--dbg-print-rate 500 \
--eval-dir ./data/ \
--epochs 30 \
--out conll2009_rm0_pl_a.25_sskip_h512_d.0_l4 \
--params-path Less_annotation_512_0.9.pkl \
--hps "{'id': 1, 'sent_edim': 100, 'sent_hdim': 512, \
'frame_edim': 128, 'role_edim': 128, 'pos_edim': 16, 'rec_layers': 1, 'gc_layers': 0, \
'pos': True, 'rm':0, 'alpha': 0.25, \
'p_lemmas':True}"


