"""
2018/10/11
MMD AAE Reappear On VLCS Datasets.
All settings following the original feature.
"""
import numpy as np
from utils.models import MMD_AAE
from utils.datasets import VLSC
from utils.logging import printRed
from sklearn.metrics import accuracy_score
from keras.losses import sparse_categorical_crossentropy
import os
import gc
from keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
patience = 10
repeat_times = 20
best_accs = np.zeros([repeat_times, 4])
for i in range(repeat_times):
    datasets = VLSC('data/VLSC', test_split=0.3)
    source_name = datasets.source_name
    best_acc = np.zeros([4])
    for idxSource, name in enumerate(source_name):
        print('Testing on {}, training on remain...'.format(name))
        datasets.set_mean_std(name)
        generator = datasets.generator(name, batch_size=100)
        val_data, val_y, val_domain = datasets.getValData(name)
        test_data, test_y = datasets.getTestData(name)

        mmd_aae = MMD_AAE(3, 4096, 5, ae=0.1, mmd=2, adv=0.1, cls=1,
                          taskLoss=sparse_categorical_crossentropy)
        model, advModel = mmd_aae.makeModel()

        metrics_name = model.metrics_names
        advGenerator = datasets.adversarialGenerator(name, model, batch_size=100)
        tol_step = 0
        count = 0
        while True:
            batch_x, batch_y = next(advGenerator)
            adv_loss, adv_acc = advModel.train_on_batch(batch_x, batch_y)
            print('[Step {} Adv] loss: {:.4}, acc: {:.4}'.format(tol_step, adv_loss, adv_acc))
            model.fit_generator(generator, steps_per_epoch=1, epochs=1)
            tol_step += 1

            results = model.evaluate(val_data, [val_data, val_domain,
                                                np.ones((val_y.shape[0], 1)),
                                                val_y], batch_size=300)
            printRed('[Step {}]'.format(tol_step), end=' ')
            for idx, mname in enumerate(metrics_name):
                # if 'decoder' in mname or 'task' in mname:
                printRed('{}: {:.4}'.format(mname, results[idx]), end=', ')
            print()
            _, _, _, test_pred = model.predict(test_data)
            label = np.argmax(test_pred, axis=1)
            # print(label.shape, np.unique(label))
            acc = accuracy_score(test_y, label)
            printRed('[TEST] ACC: {:.4}'.format(acc*100))

            if acc > best_acc[idxSource]:
                best_acc[idxSource] = acc
                count = 0
            else:
                count += 1

            if count >= patience:
                break
    print(source_name)
    print(best_acc)
    best_accs[i, :] = best_acc
    np.savez('results.npz', accs=best_accs, name=source_name)
    del mmd_aae
    gc.collect()
    K.clear_session()
    print(best_accs)
