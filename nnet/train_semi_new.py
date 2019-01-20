import torch
import torch.tensor
from nnet.util import *
import torch.autograd
from torch.autograd import Variable
from torch.nn.utils import clip_grad_value_
from torch import optim
import time
import random
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_local_voc(labels):
    return {i: label for i, label in enumerate(labels)}


def train_semi(model, train_set, dev_set, unlabeled_set, epochs, converter, unlabeled_converter, dbg_print_rate, params_path):
    idx = 0
    sample_count = 0.0
    best_F1 = -0.1
    Precision_Link_best = 0.
    Precision_POS_best = 0.
    Precision_PI_best = 0.
    #optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6)
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.002, betas=(0.9, 0.9), eps=1e-12)
    #log(optimizer.param_groups[0]['lr'])
    #optimizer.param_groups[0]['lr'] = 0.001

    Best_DEP_score = -0.1

    random.seed(1234)
    for e in range(epochs):
        tic = time.time()
        dataset = [batch for batch in train_set.batches()]
        init_dataset = [batch for batch in dataset]
        unlabeled_dataset = [batch for batch in unlabeled_set.batches()]
        random.shuffle(dataset)
        random.shuffle(unlabeled_dataset)
        dataset_len = len(dataset)
        unlabeled_dataset_len = len(unlabeled_dataset)
        unlabeled_idx = 0
        for batch in dataset:

            batch_idx = init_dataset.index(batch)
            sample_count += len(batch)

            model.zero_grad()
            optimizer.zero_grad()
            model.train()
            record_ids, batch = zip(*batch)
            model_input = converter(batch)

            unlabeled_batch = unlabeled_dataset[unlabeled_idx % unlabeled_dataset_len]
            unlabeled_record_ids, unlabeled_batch = zip(*unlabeled_batch)
            unlabeled_model_input = unlabeled_converter(unlabeled_batch)
            unlabeled_idx += 1

            model.hidden = model.init_hidden_spe()
            #model.hidden_0 = model.init_hidden_spe()
            model.hidden_2 = model.init_hidden_spe()
            model.hidden_3 = model.init_hidden_spe()
            model.hidden_4 = model.init_hidden_share()

            sentence = model_input[0]
            p_sentence = model_input[1]

            sentence_in = torch.from_numpy(sentence).to(device)
            p_sentence_in = torch.from_numpy(p_sentence).to(device)

            unlabeled_sentence = unlabeled_model_input[0]
            p_unlabeled_sentence = unlabeled_model_input[1]
            unlabeled_sentence_in = torch.from_numpy(unlabeled_sentence).to(device)
            p_unlabeled_sentence_in = torch.from_numpy(p_unlabeled_sentence).to(device)
            unlabeled_sen_lengths = unlabeled_model_input[2].sum(axis=1)
            unlabeled_sent_mask = torch.from_numpy(model_input[2]).to(device)


            #log(sentence_in)
            #log(p_sentence_in)
            #sentence_in.requires_grad_(False)
            #p_sentence_in.requires_grad_(False)

            pos_tags = model_input[2]
            pos_tags_in = torch.from_numpy(pos_tags).to(device)
            #pos_tags_in.requires_grad_(False)

            sen_lengths = model_input[3].sum(axis=1)

            sent_mask = torch.from_numpy(model_input[3]).to(device)

            target_idx_in = model_input[4]

            frames = model_input[5]
            frames_in = torch.from_numpy(frames).to(device)
            #frames_in.requires_grad_(False)

            local_roles_voc = model_input[6]
            local_roles_voc_in = torch.from_numpy(local_roles_voc).to(device)
            #local_roles_voc_in.requires_grad_(False)

            local_roles_mask = model_input[7]
            local_roles_mask_in = torch.from_numpy(local_roles_mask).to(device)
            #local_roles_mask_in.requires_grad_(False)

            region_mark = model_input[9]

            # region_mark_in = Variable(torch.LongTensor(region_mark))
            region_mark_in = torch.from_numpy(region_mark).to(device)
            #region_mark_in.requires_grad_(False)

            sent_pred_lemmas_idx = model_input[10]
            sent_pred_lemmas_idx_in = torch.from_numpy(sent_pred_lemmas_idx).to(device)
            #sent_pred_lemmas_idx_in.requires_grad_(False)

            dep_tags = model_input[11]
            dep_tags_in = torch.from_numpy(dep_tags).to(device)


            dep_heads = model_input[12]


            tags = model_input[13]
            targets = torch.tensor(tags).to(device)

            gold_pos_tags = model_input[14]
            gold_pos_tags_in = torch.from_numpy(gold_pos_tags).to(device)

            specific_dep_relations = model_input[15]
            specific_dep_relations_in = torch.from_numpy(specific_dep_relations).to(device)

            Chars = model_input[16]
            Chars_in = torch.from_numpy(Chars).to(device)

            Predicate_indicator = model_input[17]
            Predicate_indicator_in = torch.from_numpy(Predicate_indicator).to(device)


            #log(dep_tags_in)
            #log(specific_dep_relations)
            SRLloss, Link_DEPloss, Tag_DEPloss, POS_loss, PI_loss, SRLprobs, \
            Link_right, Link_all, \
            POS_right, POS_all, \
            PI_right, PI_all \
                = model(sentence_in, p_sentence_in, pos_tags_in, sent_mask, sen_lengths, target_idx_in, region_mark_in,
                        local_roles_voc_in,
                        frames_in, local_roles_mask_in, sent_pred_lemmas_idx_in, dep_tags_in, dep_heads,
                        targets, gold_pos_tags_in, specific_dep_relations_in, Chars_in, Predicate_indicator_in, False,
                        unlabeled_sentence_in, p_unlabeled_sentence_in, unlabeled_sent_mask, unlabeled_sen_lengths, False)




            idx += 1

            Final_loss = SRLloss + Link_DEPloss + Tag_DEPloss + POS_loss + PI_loss

            Final_loss.backward()
            #clip_grad_norm_(parameters=model.hidden2tag_M.parameters(), max_norm=norm)
            #clip_grad_norm_(parameters=model.hidden2tag_H.parameters(), max_norm=norm)
            #clip_grad_value_(parameters=model.parameters(), clip_value=3)
            #DEPloss.backward()
            optimizer.step()

            del model.hidden
            del model.hidden_2
            del model.hidden_3
            del model.hidden_4



            model.zero_grad()
            optimizer.zero_grad()
            model.train()
            model.hidden = model.init_hidden_spe()
            # model.hidden_0 = model.init_hidden_spe()
            model.hidden_2 = model.init_hidden_spe()
            model.hidden_3 = model.init_hidden_spe()
            model.hidden_4 = model.init_hidden_share()
            CVT_SRL_Loss = model(sentence_in, p_sentence_in, pos_tags_in, sent_mask, sen_lengths, target_idx_in, region_mark_in,
                        local_roles_voc_in,
                        frames_in, local_roles_mask_in, sent_pred_lemmas_idx_in, dep_tags_in, dep_heads,
                        targets, gold_pos_tags_in, specific_dep_relations_in, Chars_in, Predicate_indicator_in, False,
                        unlabeled_sentence_in, p_unlabeled_sentence_in, unlabeled_sent_mask, unlabeled_sen_lengths, True)
            Loss_CVT = CVT_SRL_Loss
            Loss_CVT.backward()
            optimizer.step()
            del model.hidden
            del model.hidden_2
            del model.hidden_3
            del model.hidden_4


            #if idx % 10000 == 0:
            #    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.75

            if idx % 100 ==0:
                log(idx)
                log("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                log('SRLloss')
                log(SRLloss)
                log("Link_DEPloss")
                log(Link_DEPloss)
                log("Tag_DEPloss")
                log(Tag_DEPloss)
                log("POS_loss")
                log(POS_loss)
                log("PI_loss")
                log(PI_loss)
                log("Loss_CVT")
                log(Loss_CVT)
                #log("sum")
                #log(loss)



            if idx % dbg_print_rate == 0:
                log('[epoch %i, %i * %i] ' %
                    (e, idx, len(batch)))

                log("start test...")
                losses, errors, errors_w, NonNullPredicts, right_NonNullPredicts, NonNullTruths = 0., 0, 0., 0., 0., 0.

                Link_right, Link_all, \
                POS_right, POS_all, \
                PI_right, PI_all = 0., 0.1, 0., 0.1, 0., 0.1



                log('now dev test')
                index = 0

                model.eval()
                with torch.no_grad():
                    for batch in dev_set.batches():
                        index += 1
                        # loss, e, e_w, NonNullPredict, right_NonNullPredict, NonNullTruth = self.error_computer.compute(model, batch)
                        errors, errors_w = 0, 0.0
                        NonNullPredict = 0
                        NonNullTruth = 0
                        right_NonNullPredict = 0

                        model.zero_grad()
                        optimizer.zero_grad()

                        record_ids, batch = zip(*batch)
                        model_input = converter(batch)
                        model.hidden = model.init_hidden_spe()
                        #model.hidden_0 = model.init_hidden_spe()
                        model.hidden_2 = model.init_hidden_spe()
                        model.hidden_3 = model.init_hidden_spe()
                        model.hidden_4 = model.init_hidden_share()


                        sentence = model_input[0]
                        p_sentence = model_input[1]

                        sentence_in = torch.from_numpy(sentence).to(device)
                        p_sentence_in = torch.from_numpy(p_sentence).to(device)
                        sentence_in.requires_grad_(False)
                        p_sentence_in.requires_grad_(False)

                        pos_tags = model_input[2]
                        pos_tags_in = torch.from_numpy(pos_tags).to(device)
                        pos_tags_in.requires_grad_(False)

                        sen_lengths = model_input[3].sum(axis=1)

                        target_idx_in = model_input[4]

                        frames = model_input[5]
                        frames_in = torch.from_numpy(frames).to(device)
                        frames_in.requires_grad_(False)

                        local_roles_voc = model_input[6]
                        local_roles_voc_in = torch.from_numpy(local_roles_voc).to(device)
                        local_roles_voc_in.requires_grad_(False)

                        local_roles_mask = model_input[7]
                        local_roles_mask_in = torch.from_numpy(local_roles_mask).to(device)
                        local_roles_mask_in.requires_grad_(False)

                        region_mark = model_input[9]

                        # region_mark_in = Variable(torch.LongTensor(region_mark))
                        region_mark_in = torch.from_numpy(region_mark).to(device)
                        region_mark_in.requires_grad_(False)

                        sent_pred_lemmas_idx = model_input[10]
                        sent_pred_lemmas_idx_in = torch.from_numpy(sent_pred_lemmas_idx).to(device)
                        sent_pred_lemmas_idx_in.requires_grad_(False)

                        dep_tags = model_input[11]
                        dep_tags_in = torch.from_numpy(dep_tags).to(device)

                        dep_heads = model_input[12]

                        # root_dep_tags = model_input[12]
                        # root_dep_tags_in = Variable(torch.from_numpy(root_dep_tags), requires_grad=False)

                        tags = model_input[13]
                        targets = torch.tensor(tags).to(device)

                        gold_pos_tags = model_input[14]
                        gold_pos_tags_in = torch.from_numpy(gold_pos_tags).to(device)

                        specific_dep_relations = model_input[15]
                        specific_dep_relations_in = torch.from_numpy(specific_dep_relations).to(device)

                        Chars = model_input[16]
                        Chars_in = torch.from_numpy(Chars).to(device)

                        Predicate_indicator = model_input[17]
                        Predicate_indicator_in = torch.from_numpy(Predicate_indicator).to(device)

                        # log(dep_tags_in)
                        # log(specific_dep_relations)
                        SRLloss, Link_DEPloss, Tag_DEPloss, POS_loss, PI_loss, SRLprobs, \
                        Link_right_b, Link_all_b, \
                        POS_right_b, POS_all_b, \
                        PI_right_b, PI_all_b \
                            = model(sentence_in, p_sentence_in, pos_tags_in, sent_mask, sen_lengths, target_idx_in, region_mark_in,
                        local_roles_voc_in,
                        frames_in, local_roles_mask_in, sent_pred_lemmas_idx_in, dep_tags_in, dep_heads,
                        targets, gold_pos_tags_in, specific_dep_relations_in, Chars_in, Predicate_indicator_in, False,
                        None, None, None, None, False)


                        Link_right += Link_right_b
                        Link_all += Link_all_b
                        POS_right += POS_right_b
                        POS_all += POS_all_b
                        PI_right += PI_right_b
                        PI_all += PI_all_b

                        if SRLprobs == 0:
                            continue
                        labels = np.argmax(SRLprobs.cpu().data.numpy(), axis=1)
                        labels = np.reshape(labels, sentence.shape)

                        for i, sent_labels in enumerate(labels):
                            labels_voc = batch[i][-4]
                            local_voc = make_local_voc(labels_voc)
                            for j in range(len(labels[i])):
                                best = local_voc[labels[i][j]]
                                true = local_voc[tags[i][j]]

                                if true != '<pad>' and true != 'O':
                                    NonNullTruth += 1
                                if true != best:
                                    errors += 1
                                if best != '<pad>' and best != 'O' and true != '<pad>':
                                    NonNullPredict += 1
                                    if true == best:
                                        right_NonNullPredict += 1


                        NonNullPredicts += NonNullPredict
                        right_NonNullPredicts += right_NonNullPredict
                        NonNullTruths += NonNullTruth

                        del model.hidden
                        del model.hidden_2
                        del model.hidden_3
                        del model.hidden_4



                Predicat_num = 6300
                P = (right_NonNullPredicts + Predicat_num) / (NonNullPredicts + Predicat_num)
                R = (right_NonNullPredicts + Predicat_num) / (NonNullTruths + Predicat_num)
                F1 = 2 * P * R / (P + R)
                log(right_NonNullPredicts)
                log(NonNullPredicts)
                log(NonNullTruths)
                log('Precision: ' + str(P), 'recall: ' + str(R), 'F1: ' + str(F1))
                P = (right_NonNullPredicts) / (NonNullPredicts + 1)
                R = (right_NonNullPredicts) / (NonNullTruths+1)
                F1 = 2 * P * R / (P + R + 0.0001)
                log('Precision: ' + str(P), 'recall: ' + str(R), 'F1: ' + str(F1))

                log('Best F1: ' + str(best_F1))
                if F1 > best_F1:
                    best_F1 = F1
                    torch.save(model.state_dict(), params_path)
                    log('New best, model saved')


                P_Link =  Link_right/Link_all
                log('Link_Precision' + str(P_Link))
                if P_Link > Precision_Link_best:
                    Precision_Link_best = P_Link
                    log('New Link best!: ' + str(Precision_Link_best))
                else:
                    log('Link best!: ' + str(Precision_Link_best))

                P_POS = POS_right/POS_all
                log('POS_Precision' + str(P_POS))
                if P_POS > Precision_POS_best:
                    Precision_POS_best = P_POS
                    log('New POS best!: ' + str(Precision_POS_best))
                else:
                    log('POS best!: ' + str(Precision_POS_best))

                P_PI = PI_right/PI_all
                log('PI precision' + str(P_PI))
                if P_PI > Precision_PI_best:
                    Precision_PI_best = P_PI
                    log('New PI best!: ' + str(Precision_PI_best))
                else:
                    log('PI best!: ' + str(Precision_PI_best))




                log('Best F1: ' + str(best_F1))
                if F1 > best_F1:
                    best_F1 = F1
                    torch.save(model.state_dict(), params_path)
                    log('New best, model saved')






       ##########################################################################################


        tac = time.time()

        passed = tac - tic
        log("epoch %i took %f min (~%f sec per sample)" % (
            e, passed / 60, passed / sample_count
        ))





