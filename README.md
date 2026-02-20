# CVLabs
Я заполнил форму, но пока что не доделал демонстрацию работы.
Цель - получить дедлайн (пожалуйста), я уже в процессе доработки.
Вот вывод с cmd с обучение ResNet50, на момент val_acc ≈ 0.91 — я подумал, что это почти наверняка даст F1_macro > 0.8 на test, но видимо классов все равно было многовато.

```
Epoch 01/8 | loss=0.7345 | val_acc=0.8481 | 4233.3s saved: outputs\resnet50_ft\best.pt Epoch 02/8 | loss=0.5468 | val_acc=0.8649 | 4588.6s saved: outputs\resnet50_ft\best.pt Epoch 03/8 | loss=0.4796 | val_acc=0.8556 | 5010.8s Epoch 04/8 | loss=0.4122 | val_acc=0.8857 | 4309.8s saved: outputs\resnet50_ft\best.pt Epoch 05/8 | loss=0.3444 | val_acc=0.8987 | 4026.9s saved: outputs\resnet50_ft\best.pt Epoch 06/8 | loss=0.2770 | val_acc=0.9007 | 3957.8s saved: outputs\resnet50_ft\best.pt Epoch 07/8 | loss=0.2186 | val_acc=0.9064 | 3965.1s saved: outputs\resnet50_ft\best.pt Epoch 08/8 | loss=0.1812 | val_acc=0.9099 | 4005.5s saved: outputs\resnet50_ft\best.pt Done. Best val_acc: 0.9098555100280353 C:\Users\user\Desktop\CV_ITMO>
```

А вот вывод для макро:
```
C:\Users\user\Desktop\CV_ITMO>C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe eval.py --ckpt outputs\resnet50_ft\best.pt --data_root Confirmed_fronts
Device: cpu
[make_splits] Удалил 2 примеров из слишком редких классов (<5).
arch: resnet50_ft
F1_macro: 0.6367628634805814
              precision    recall  f1-score   support

       Beige       0.76      0.82      0.79        90
       Black       0.94      0.96      0.95      2148
        Blue       0.93      0.94      0.93      1273
      Bronze       0.53      0.41      0.46        49
       Brown       0.60      0.68      0.64       137
    Burgundy       0.00      0.00      0.00         1
        Gold       0.62      0.48      0.54        33
       Green       0.90      0.80      0.85       117
        Grey       0.86      0.89      0.87      1421
     Magenta       0.00      0.00      0.00         1
      Maroon       1.00      0.25      0.40         4
 Multicolour       0.13      0.07      0.09        29
      Orange       0.75      0.80      0.77        84
        Pink       0.75      0.69      0.72        13
      Purple       0.75      0.76      0.75        54
         Red       0.95      0.96      0.96       914
      Silver       0.90      0.91      0.90      1166
   Turquoise       0.67      0.50      0.57         4
    Unlisted       0.44      0.18      0.26       227
       White       0.97      0.98      0.97      1409
      Yellow       0.90      0.97      0.93       100

    accuracy                           0.91      9274
   macro avg       0.68      0.62      0.64      9274
weighted avg       0.90      0.91      0.90      9274

Saved: f1.json, confusion_matrix.json, classification_report.txt -> outputs\resnet50_ft

C:\Users\user\Desktop\CV_ITMO>
```

То есть жеский дисбаланс классов, нужно чуть подчистить.
