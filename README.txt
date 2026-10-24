Q-Net Lab
Десктопное приложение для обучения и проверки нейросетей (FNN), распознающих
состояния кудитов и кубитных регистров по кривым дисперсионного считывания
(временные трассы квадратур I и Q).

ИСХОДНЫЙ КОД — 4 ФАЙЛА
  main.py            — точка входа: заставка, проверка лицензии, запуск окна (классов нет)
  neural_service.py  — класс NeuralService: данные, фильтры, сеть, обучение, метрики
  mainwindow.py      — класс MainWindow: окно, проект, все файлы, графики
  license_manager.py — класс LicenseManager: RSA-лицензия (проверка подписи)
Остальное — не код: public_key.pem и license.json (лицензия), requirements.txt,
UserGuide.html (руководство пользователя), examples\ (синтетические демо-данные).

УСТАНОВКА И ЗАПУСК В PYCHARM (Windows)
 1. File > Open… > папка QNetLab.
 2. Settings > Project > Python Interpreter > Add Interpreter > Virtualenv
    (базовый интерпретатор — Python 3.10+; подойдёт и Python из Anaconda).
 3. В Terminal PyCharm:
      python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
      python -m pip install -r requirements.txt
 4. Откройте main.py > Run 'main'. Аргументы не нужны.
 Из командной строки: python main.py
 Python из Anaconda и PyQt6 6.10+: в Library\bin Anaconda лежат свои ICU и Qt 6, которые
 подменяют нужные PyQt6 библиотеки («DLL load failed … Не найдена указанная процедура»).
 main.py это учитывает: заранее загружает системную C:\Windows\System32\icuuc.dll и
 импортирует все модули Qt ДО PyTorch (torch в venv добавляет Library\bin Anaconda в поиск DLL).
 В своих скриптах делайте так же: main.preload_system_icu(); from PyQt6 import QtSvg;
 import mainwindow — и только потом neural_service.

ЛИЦЕНЗИЯ
 При запуске показывается заставка с полосой загрузки, проверяется RSA-подпись
 license.json открытым ключом public_key.pem (оба лежат рядом с main.py) и срок.
 Без действительной лицензии главное окно не открывается; можно выбрать другой файл.
 В поставке: license.json на 365 дней, подписанная ключом издателя. Закрытый ключ
 издателя лежит ВНЕ папки приложения: ..\QNetLab_issuer\private_key.pem. Не передавайте
 его пользователям: владелец закрытого ключа может выпускать лицензии.
 Команды издателя:
   python main.py --issue-license --private-key ..\QNetLab_issuer\private_key.pem --subject "Лаборатория" --days 365 --overwrite
   python main.py --verify-license [--license путь\license.json]
   python main.py --generate-keys --private-key путь\private_key.pem --public-key путь\public_key.pem
 Это обычная офлайн-лицензия: без привязки к ПК и сети, не защищает от правки исходников
 или перевода часов.

ФОРМАТЫ ДАННЫХ (ТЗ: .csv и .npz)
 NPZ (numpy.savez):
   iq      complex [N, A, T]   — трассы (I = Re, Q = Im); либо iq real [N, A, T, 2];
           либо пара I и Q real [N, A, T]. Для одного атома допускается [N, T].
   labels  int [N, A]          — подготовленный уровень каждого атома (необязательно)
   time_ns [T], shot_ids [N], ideal_probs [D] или [N, D] (для linear XEB) — необязательно
 CSV «длинный» (одна строка — один отсчёт):  shot,atom,sample,I,Q[,label][,time_ns]
 CSV «широкий» (одна строка — трасса атома): shot,atom[,label],I_0..I_{T-1},Q_0..Q_{T-1}
 Разделитель , ; или TAB определяется автоматически; десятичная точка.
 N — измерения (shots), A — атомы (каналы), T — временные отсчёты.

ФАЙЛ ВЕСОВ (.pt)
 torch.save-словарь: конфигурация, веса сети, откалиброванные фильтры, нормировка,
 форма входа, состояние оптимизатора (для дообучения), метаданные (дата, метрики).
 Загружается безопасно (torch.load(weights_only=True)).

ЖУРНАЛ ОБУЧЕНИЯ
 logs\training_ДАТА.csv: epoch,loss,accuracy,val_loss,val_accuracy,
 val_assignment_fidelity,val_macro_f1,val_xeb,val_xeb_linear,lr.
 Кнопка «Загрузить журнал» принимает такой CSV или JSON (список записей или project.json).

ПАПКА ПРОЕКТА (создаёт пользователь кнопкой «Новый проект»)
 project.json, logs.txt (системный журнал, ТЗ 4.4), data\, weights\, logs\, reports\.

ДВЕ МОДЕЛИ (ТЗ, разд. 6)
 1) FNN по временным I/Q (базовая FNN из [1] статьи);
 2) фильтры QMF/RMF/EMF + FNN по статье Mude et al., arXiv:2405.08982
    (по умолчанию — отдельная сеть на атом со слоями ⌊P/2⌋, ⌊P/4⌋, не уже 2k+2 и k+1).
 Отличие от статьи: в ядре согласованного фильтра используется регуляризованная сумма
 дисперсий (отдельно по I и Q) вместо разности — разность неустойчива при σ0≈σ1.

ДЕМО-ДАННЫЕ (examples\, синтетические)
 demo_register_5x3.npz     — 6000 измерений, 5 кутритов, 120 отсчётов × 4 нс, с метками
                             (распад T1 во время считывания, термальное возбуждение 2 %, перекрёстные помехи)
 demo_experiment_5x3.npz   — 1500 независимых измерений той же модели (для валидации)
 demo_unlabeled_5x3.npz    — 500 измерений без меток (I и Q отдельными массивами)
 demo_qubit_leakage_wide.csv — 1 кубит, подготовка 0/1, 4 % естественной утечки |1⟩→|2⟩
                             (для «Поиска утечки»), широкий CSV
 demo_long_2atoms.csv      — 200 измерений × 2 атома, длинный CSV с разделителем «;»
 Синтетические данные проверяют работоспособность и не подтверждают точность на эксперименте.

ПРОВЕРЕНО
 Windows 11, Python 3.14.6 (Anaconda 2026.07, venv .venv), torch 2.14.1+cpu, PyQt6 6.11 (Qt 6.11.2),
 numpy 2.5, pandas 3.1, matplotlib 3.11; ранее также Python 3.13.9 + PyQt6 6.9.1.
 На demo_register_5x3 (30 эпох): фильтры+FNN — 6505 параметров (5 сетей 45-22-11-3),
 F5Q 0.976 на независимых данных; FNN по I/Q — 4.5 млн параметров, F5Q 0.980;
 LDA/QDA на MTV — 84/87 % accuracy регистра.
