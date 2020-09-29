import os

# os.system('python .\execute-rnn.py --data=wpg-4 --target=ItemShortName --generated_num=960 --total_batch=100 --batch_size=32 --seq_length=96')
os.system('python .\execute-seqgan.py --data=wpg-1 --target=ItemShortName')
os.system('python .\execute-seqgan.py --data=wpg-2 --target=ItemShortName')
os.system('python .\execute-seqgan.py --data=wpg-3 --target=ItemShortName')
os.system('python .\execute-seqgan.py --data=wpg-4 --target=ItemShortName')
os.system('python .\execute-seqgan.py --data=wpg-5 --target=ItemShortName')