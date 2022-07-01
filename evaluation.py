import sys
from sklearn.metrics import f1_score

label_dic = ['[微笑]', '[嘻嘻]', '[笑cry]', '[怒]', '[泪]', '[允悲]', '[憧憬]', '[doge]', '[并不简单]', '[思考]', '[费解]', '[吃惊]',
                     '[拜拜]',
                     '[吃瓜]', '[赞]', '[心]', '[伤心]', '[蜡烛]', '[给力]', '[威武]', '[跪了]', '[中国赞]', '[给你小心心]', '[酸]']
id2label = {k: v for k, v in enumerate(label_dic)}  # 用于标签的部分
label2id = {v: k for k, v in enumerate(label_dic)}

def convert_label(fn_result):
    convert_label = []

    for line in open(fn_result, 'r', encoding='utf-8'):
        labellist = line.strip().split(' ')[1:]

        onehot_label = [0] * 24
        for label in labellist:
            onehot_label[label2id[label]] = 1
        convert_label.append(onehot_label)
    return convert_label

def score_calculation(pred, true):
    macro_f1 = f1_score(pred, true, average='macro')
    return macro_f1


def main():
    """
    Generate classification_report from given pred and gold tsv files.
    """

    # Usage:
    # python evaluate PREDICTION_FILE GOLD_ANSWER_FILE
    # Example:
    # python evaluate pred.tsv gold.tsv

    if len(sys.argv) < 3:
        print("Please indicate the prediction and gold tsvs.")
        quit()
    else:
        pred_fn = sys.argv[1]
        gold_fn = sys.argv[2]

    print('Loading the datasets ...')
    pred_lbl = convert_label(pred_fn)
    gold_lbl = convert_label(gold_fn)

    print("Evaluating ...")
    try:
        macro_f1 = score_calculation(pred_lbl, gold_lbl)

        print('macro_f1: {:.4f}'.format(macro_f1))
    except Exception as ex:
        print('error:', ex)


if __name__ == '__main__':
    main()