__author__ = 'tylin'
from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from spice.spice import Spice

class COCOEvalCap:
    def __init__(self, annotation, prediction):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.annotation = annotation
        self.prediction = prediction
        self.params = {'question_id': list(annotation.keys())}

    def covert_format(self,original,qid):
        converted = []
        for i in range(len(original)):
            tmp_dict = dict()
            tmp_dict['question_id'] = qid
            tmp_dict['id'] = original[i]['id']
            tmp_dict['caption'] = original[i]['explanation']
            converted.append(tmp_dict)
        return converted


    def evaluate(self):
        qids = self.params['question_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for qid in qids:
            gts[qid] = self.covert_format(self.annotation[qid],qid)
            res[qid] = self.covert_format(self.prediction[qid],qid)

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)


        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE") # comment SPICE for faster prototyping
        ]

        report_score = dict()
        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    # print("%s: %0.3f"%(m, sc))
                    report_score[m] = sc*100
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                # print("%s: %0.3f"%(method, score))
                report_score[method] = score*100
        self.setEvalImgs()
        return report_score

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]