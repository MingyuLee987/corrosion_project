"""Archive public publisher text for manual, source-located data verification."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import urllib.request
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / 'sources/literature'


def fetch(item):
    path = SOURCES / (item['reference_id']+'.md')
    try:
        if not path.exists():
            request = urllib.request.Request('https://r.jina.ai/'+item['source_url'], headers={'User-Agent':'Mozilla/5.0'})
            content = urllib.request.urlopen(request, timeout=90).read()
            path.write_bytes(content)
        content = path.read_bytes()
        text = content.decode('utf-8')
        if len(content) < 2000 or 'Just a moment...' in text[:1500]:
            return dict(**item, status='unusable_text', bytes=len(content))
        html_path = SOURCES / (item['reference_id']+'.html')
        if not html_path.exists():
            request = urllib.request.Request('https://r.jina.ai/'+item['source_url'],
                headers={'X-Respond-With':'html','User-Agent':'Mozilla/5.0'})
            html_path.write_bytes(urllib.request.urlopen(request,timeout=90).read())
        html = html_path.read_bytes()
        soup = BeautifulSoup(html,'html.parser')
        if not soup.find('table'):
            # Keep usable cached text; flag a blocked live HTML response accurately.
            return dict(**item, status='text_archived_html_unusable', bytes=len(content),
                        source_text_sha256=hashlib.sha256(content).hexdigest())
        tables = []
        for table in soup.find_all('table'):
            cells = [[cell.get_text(' ',strip=True) for cell in row.find_all(['th','td'])]
                     for row in table.find_all('tr')]
            if cells:
                tables.append(cells)
        (SOURCES / (item['reference_id']+'_tables.json')).write_text(json.dumps(tables,ensure_ascii=False,indent=2),encoding='utf-8')
        return dict(**item, status='archived_for_verification', bytes=len(content), tables=len(tables),
                    source_text_sha256=hashlib.sha256(content).hexdigest(),source_html_sha256=hashlib.sha256(html).hexdigest())
    except Exception as exc:
        return dict(**item, status='fetch_failed', reason=str(exc))


def main():
    SOURCES.mkdir(parents=True,exist_ok=True)
    items = json.loads((SOURCES/'candidates.json').read_text(encoding='utf-8'))
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(fetch,item) for item in items]
        for future in as_completed(futures):
            result=future.result()
            results.append(result)
            print(result['reference_id'],result['status'],result.get('bytes',result.get('reason')),flush=True)
    (SOURCES/'fetch_manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
