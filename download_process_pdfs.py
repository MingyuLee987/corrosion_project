from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
URLS = {
    'LowC_heat_2012':'https://www.electrochemsci.org/papers/vol7/7086677.pdf',
    'TurbineSteel_2019':'https://revistadechimie.ro/pdf/40%20PREDESCU%207%2019.pdf',
    'SpringSteel_2023':'https://e-university.tu-sofia.bg/e-conf/files/169/paper_10.47978%40TUS.2023.73.04.002.pdf',
    'AustemperHighSi_2021':'https://www.research.unipd.it/bitstream/11577/3364383/1/materials-14-00288.pdf',
}


def fetch(key,url):
    path=ROOT/'sources'/f'{key}.pdf'
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(url,timeout=60).read())
    reader=PdfReader(path)
    text='\n'.join(f'PDF_PAGE {i+1}\n'+p.extract_text() for i,p in enumerate(reader.pages))
    (ROOT/'sources'/f'{key}.txt').write_text(text,encoding='utf-8')
    return key,len(reader.pages)


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,k,u):k for k,u in URLS.items()}
        for f in as_completed(futures):
            try:
                print(fetch_result:=f.result(),flush=True)
            except Exception as exc:
                print(futures[f],str(exc),flush=True)
