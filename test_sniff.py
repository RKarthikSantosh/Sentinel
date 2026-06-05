from scapy.all import sniff, conf
print('Trying to sniff...')
try:
  sniff(count=1, L2socket=conf.L3socket)
  print('Success!')
except Exception as e:
  print('Error:', e)
