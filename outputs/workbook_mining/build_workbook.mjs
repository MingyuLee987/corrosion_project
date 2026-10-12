import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const directory = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(directory, '../..');
const payload = JSON.parse(await fs.readFile(path.join(root, 'results/workbook_mining/workbook_payload.json'), 'utf8'));
const wb = Workbook.create();
const col = index => {
  let text = '';
  for (let x = index; x > 0; x = Math.floor((x - 1) / 26)) text = String.fromCharCode(65 + (x - 1) % 26) + text;
  return text;
};
function grid(name, headers, rows, tableName) {
  const sheet = wb.worksheets.add(name);
  const matrix = [headers, ...rows.map(r => headers.map(h => r[h] ?? null))];
  const range = sheet.getRange(`A1:${col(headers.length)}${matrix.length}`);
  range.values = matrix;
  range.format.font = { name: 'Malgun Gothic', size: 10 };
  range.format.rowHeight = 25;
  range.format.columnWidth = 14;
  sheet.tables.add(`A1:${col(headers.length)}${matrix.length}`, true, tableName);
  sheet.getRange(`A1:${col(headers.length)}1`).format = {
    fill: '#174A5B', font: { name: 'Malgun Gothic', size: 10, color: '#FFFFFF', bold: true },
    rowHeight: 48, wrapText: true, verticalAlignment: 'center'
  };
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
  return sheet;
}

const data = grid('데이터마이닝', payload.headers, payload.rows, 'MiningRecords');
const lastRow = payload.rows.length + 1;
data.freezePanes.freezeColumns(3);
data.getRange('A:A').format.columnWidth = 7;
data.getRange('B:B').format.columnWidth = 44;
data.getRange('C:C').format.columnWidth = 34;
data.getRange(`B2:C${lastRow}`).format.wrapText = true;
data.getRange(`A2:${col(payload.headers.length)}${lastRow}`).format.rowHeight = 40;
data.getRange(`D2:X${lastRow}`).setNumberFormat('0.######');
data.getRange(`AA2:AB${lastRow}`).setNumberFormat('0.##');
data.getRange('AD:AD').format.columnWidth = 26;
data.getRange('AG:AI').format.columnWidth = 17;
data.getRange(`AK2:AL${lastRow}`).setNumberFormat('0.000E+00');
for (const h of ['record_id', 'doi', 'source_url', 'source_locator', 'provenance', 'target_status', 'remarks', 'missing_measurement_columns', 'processing_sequence', 'electrolyte_concentration_raw']) {
  const c = col(payload.headers.indexOf(h) + 1);
  data.getRange(`${c}:${c}`).format.columnWidth = h === 'remarks' || h === 'missing_measurement_columns' ? 70 : 40;
}
data.getRange(`A2:${col(payload.headers.length)}${lastRow}`).format.verticalAlignment = 'center';

const coverage = wb.worksheets.add('확보현황');
coverage.showGridLines = false;
coverage.getRange('A1:E42').format.font = { name: 'Malgun Gothic', size: 11 };
coverage.getRange('A1').values = [['문헌 데이터 확보 현황']];
coverage.getRange('A1').format.font = { name: 'Malgun Gothic', size: 15, bold: true, color: '#174A5B' };
coverage.getRange('A3:D4').values = [
  ['수집 행', '수집 문헌', 'icorr 확보 행', '전체 34개 항목 완비 행'],
  [payload.stats.rows, payload.stats.papers, payload.stats.numeric_icorr_rows, 0]
];
coverage.getRange('A3:D3').format = { fill: '#E8F1F3', font: { bold: true, color: '#174A5B' }, wrapText: true, rowHeight: 35 };
coverage.getRange('A6:E6').values = [['항목', '값 확보 행', '미확보 행', '확보율', '처리 원칙']];
coverage.getRange('A7:C40').values = payload.coverage_rows.map(r => [r.column, r.present_rows, r.missing_rows]);
coverage.getRange('D7').formulas = [['=B7/(B7+C7)']];
coverage.getRange('D7:D40').fillDown();
coverage.getRange('D7:D40').setNumberFormat('0.0%');
coverage.getRange('E7').values = [['미보고 값을 0·평균·강종 대표값으로 채우지 않음']];
coverage.getRange('E8').values = [['표면 산화물·침투제 첨가량을 벌크 RE 함량으로 바꾸지 않음']];
coverage.getRange('E9').values = [['Ipass·공식전위·질량감소를 icorr로 바꾸지 않음']];
coverage.getRange('E10').values = [['전체 컬럼 기준으로 학습에 투입 가능한 행: 0']];
coverage.getRange('E11').values = [[`신규 수집 ${payload.stats.new_rows}행. 기존 자료 ${payload.stats.legacy_rows}행은 이번에 전체 원문 재검증하지 않음`]];
coverage.getRange('E12').values = [[`49편 중 ${49-payload.stats.papers}편은 원문 재확보·추출이 남아 있음`]];
coverage.getRange('E13').values = [['원본 데이터마이닝 시트의 행 연결 오류를 그대로 학습하지 않음']];
coverage.getRange('E14').values = [['공란은 미확보 값. 표의 Balance는 원문 표기이며 수치가 아님']];
coverage.getRange('A6:D40').format.rowHeight = 26;
coverage.getRange('A6:D6').format = { fill: '#174A5B', font: { color: '#FFFFFF', bold: true } };
coverage.getRange('A:A').format.columnWidth = 32;
coverage.getRange('B:D').format.columnWidth = 22;
coverage.getRange('E:E').format.columnWidth = 78;
coverage.freezePanes.freezeRows(6);

const review = grid('문헌검토', payload.review_headers, payload.review_rows, 'LiteratureReview');
review.getRange('A:A').format.columnWidth = 10;
review.getRange('B:B').format.columnWidth = 38;
review.getRange('C:C').format.columnWidth = 70;
review.getRange('C2:C50').format.wrapText = true;
review.getRange('A2:J50').format.rowHeight = 45;
review.getRange('I:I').format.columnWidth = 46;
review.getRange('J:J').format.columnWidth = 90;

wb.recalculate();
console.log((await wb.inspect({ kind: 'region', sheetId: '확보현황', range: 'A3:D4', maxChars: 1000 })).ndjson);
console.log((await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!', options: { useRegex: true, maxResults: 20 }, maxChars: 1000 })).ndjson);
for (const [name, range, file] of [
  ['데이터마이닝', 'A1:L10', 'preview_data.png'],
  ['확보현황', 'A1:E15', 'preview_coverage.png'],
  ['문헌검토', 'A1:I7', 'preview_review.png']
]) {
  const blob = await wb.render({ sheetName: name, range, scale: 1.4, format: 'png' });
  await fs.writeFile(path.join(directory, file), new Uint8Array(await blob.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(directory, 'literature_mining_collected.xlsx'));
console.log(`Workbook exported: ${payload.stats.rows} records, 49-paper registry, 34-column coverage.`);
