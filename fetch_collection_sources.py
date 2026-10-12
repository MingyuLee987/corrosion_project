"""Cache primary-source documents for auditable literature extraction."""
from pathlib import Path
from urllib.request import urlopen, Request
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import pymupdf

SOURCES = {
 'FeMnAl25': 'https://mdpi-res.com/d_attachment/metals/metals-15-01249/article_deploy/metals-15-01249.pdf',
 'X65_FSW21': 'https://www.jcscp.org/article/2021/1005-4537/1005-4537-2021-41-1-87.shtml',
 'MnNiMo24': 'https://mdpi-res.com/d_attachment/coatings/coatings-14-00588/article_deploy/coatings-14-00588.pdf',
 'DSS_LST': 'https://upcommons.upc.edu/server/api/core/bitstreams/1ab06bd2-7039-40a6-b4da-bb39aea85913/content',
 'HRB500E24': 'https://www.nature.com/articles/s41598-024-67799-9.pdf',
 'QC304_20': 'https://www.ams.org.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=28480',
}
def fetch(item):
 name,url=item
 dest=Path('papers/time_verified')/f'{name}.pdf'
 try:
  if not dest.exists():
   body=urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30).read()
   if not body.startswith(b'%PDF'):
    if name=='X65_FSW21':
     dest=dest.with_suffix('.html');dest.write_bytes(body)
     dest.with_suffix('.txt').write_text(BeautifulSoup(body,'html.parser').get_text(' ',strip=True),encoding='utf8')
     return f'{name}: primary HTML saved'
    raise ValueError('Not PDF')
   dest.write_bytes(body)
  doc=pymupdf.open(dest)
  dest.with_suffix('.txt').write_text('\n'.join(f'PAGE {i+1}\n{p.get_text()}' for i,p in enumerate(doc)),encoding='utf8')
  return f'{name}: {len(doc)} pages'
 except Exception as e: return f'{name}: {e}'
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:
  for result in pool.map(fetch,SOURCES.items()): print(result,flush=True)
