

LGBMC_param = {
#  "LGBMClassifier_parameters_ua": [
#    {
#      "category": "Основні параметри керування навчанням (Core Parameters)",
#      "description": "Контролюють загальний процес навчання та функцію втрат.",
#      "list": [
#        {
#          "parameter": "objective",
#          "explanation": "Мета навчання (функція втрат).",
#          "default_value": "'binary', 'multiclass', 'regression' та інші"
#        },
"objective": 'binary',
#        {
#          "parameter": "boosting_type",
#          "explanation": "Тип бустингу: gbdt (дерева градієнтного бустингу), dart, rf (випадковий ліс).",
#          "default_value": "'gbdt'"
#        },
"boosting_type": 'gbdt',
#        {
#          "parameter": "n_estimators / num_iterations",
#          "explanation": "Кількість бустингових ітерацій (дерев).",
#          "default_value": "100"
#        },
"n_estimators": 50,
#        {
#          "parameter": "learning_rate / eta",
#          "explanation": "Швидкість навчання. Контролює розмір кроку на кожній ітерації.",
#          "default_value": "0.1"
#        },
"learning_rate": 0.018,
#    {
#          "parameter": "num_class",
#          "explanation": "Кількість класів (для мультикласової класифікації).",
#          "default_value": "1 (для бінарної)"
#        },
"num_class": 1,
#        {
#          "parameter": "metric",
#          "explanation": "Метрики для оцінки моделі, що використовуються для ранньої зупинки.",
#          "default_value": "'l1', 'l2', 'auc', 'binary_logloss' тощо"
#        },
"metric": "'l1', 'l2', 'auc', 'binary_logloss'",
#        {
#          "parameter": "seed / random_state",
#          "explanation": "Насіння для генератора випадкових чисел (для відтворюваності).",
#          "default_value": "None або ціле число"
#        }
#      ]
"seed": 123,
#    },
#    {
#      "category": "Параметри структури дерева та складності (Tree Parameters)",
#      "description": "Контролюють складність кожного дерева і є критичними для боротьби з перенавчанням.",
#      "list": [
#        {
#          "parameter": "num_leaves",
#          "explanation": "Максимальна кількість листків в одному дереві. Головний регулятор складності LightGBM.",
#          "default_value": "31"
#        },
"num_leaves": 31,
#
#        {
#          "parameter": "max_depth",
#          "explanation": "Максимальна глибина дерева. Використовується для обмеження росту дерева явним чином.",
#          "default_value": "-1 (необмежена)"
#        },
"max_depth": -1,
#        {
#          "parameter": "min_child_samples / min_data_in_leaf",
#          "explanation": "Мінімальна кількість об'єктів (зразків) у листі дерева.",
#          "default_value": "20"
#        },
"min_child_samples": 25,
#        {
#          "parameter": "min_child_weight",
#          "explanation": "Мінімальна сума Гессіанів (других похідних) у листі.",
#          "default_value": "1e-3"
#        },
"min_child_weight": 0.0001,
#       {
#          "parameter": "max_bin",
#          "explanation": "Максимальна кількість сегментів (бінів), на які розділятимуться ознаки.",
#          "default_value": "255"
#        }
"max_bin": 255,
#      ]
#    },
#    {
#      "category": "Параметри вибірки та регуляризації (Sampling & Regularization)",
#      "description": "Допомагають моделі узагальнювати дані та зменшувати перенавчання.",
#      "list": [
#        {
#          "parameter": "feature_fraction / colsample_bytree",
#          "explanation": "Частка ознак, що випадковим чином обирається для побудови кожного дерева.",
#          "default_value": "1.0 (відключено)"
#        },
"feature_fraction": 0.8,
#    {
#          "parameter": "bagging_fraction / subsample",
#          "explanation": "Частка даних, що випадковим чином обирається для побудови кожного дерева (вибірка рядків).",
#          "default_value": "1.0 (відключено)"
#        },
"bagging_fraction": 0.8,
#        {
#          "parameter": "bagging_freq",
#          "explanation": "Частота виконання беггінгу.",
#          "default_value": "0"
#        },
"bagging_freq": 1,
#        {
#          "parameter": "lambda_l1 / reg_alpha",
#          "explanation": "Коефіцієнт L1-регуляризації.",
#          "default_value": "0.0"
#        },
"lambda_l1": 0.1,
#        {
#          "parameter": "lambda_l2 / reg_lambda",
#          "explanation": "Коефіцієнт L2-регуляризації.",
#          "default_value": "0.0"
#        },
"lambda_l2": 0.1,
#    {
#          "parameter": "min_gain_to_split",
#          "explanation": "Мінімальний виграш (gain) для виконання поділу у вузлі.",
#          "default_value": "0.0"
#        }
"min_gain_to_split": 0.0,
#      ]
#    },
#    {
#      "category": "Параметри продуктивності та оптимізації (Performance & Optimization)",
#      "description": "Впливають на швидкість навчання та обробку спеціальних випадків (наприклад, незбалансованість).",
#      "list": [
#        {
#          "parameter": "n_jobs",
#          "explanation": "Кількість потоків (ядер процесора) для навчання.",
#          "default_value": "-1 (використовувати всі ядра)"
#        },
"n_jobs":-1,
#        {
#          "parameter": "device",
#          "explanation": "Використовувати CPU або GPU для прискорення.",
#          "default_value": "'cpu' (або 'gpu')"
#        },
"device": 'gpu',
#        {
#          "parameter": "early_stopping_round",
#          "explanation": "Кількість ітерацій без покращення метрики, після якої навчання зупиняється.",
#          "default_value": "Немає (потрібно встановлювати вручну)"
#        },
#        {
#          "parameter": "verbose",
#          "explanation": "Виводить інформацію під час навчання.",
#          "default_value": "1"
#        },
"verbose": 1,
#        {
#          "parameter": "tree_learner",
#          "explanation": "Тип паралельного навчання дерев.",
#          "default_value": "'serial', 'feature', 'data', 'voting'"
#        },
"tree_learner": 'data',
#        {
#          "parameter": "is_unbalance",
#          "explanation": "Використовується для незбалансованих даних (альтернатива scale_pos_weight).",
#          "default_value": "False"
#        },
#        {
#          "parameter": "scale_pos_weight",
#          "explanation": "Коефіцієнт для масштабування позитивного класу (для незбалансованих бінарних даних).",
#          "default_value": "1.0"
#        }
"scale_pos_weight": 1.0,
#      ]
#    }
#  ]
}