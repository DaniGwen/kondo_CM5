const {chromium}=require('C:/Users/gwenski/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try{
  const page=await browser.newPage({viewport:{width:1280,height:1100}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8765');
  await page.waitForFunction(()=>document.getElementById('exportRun').options.length>1);
  await page.selectOption('#exportRun','td3_corrected_step_v3_20261024_01');
  assert(await page.locator('#hardwareExport').isDisabled());
  assert((await page.locator('#exportBlockers').textContent()).includes('66 features'));
  assert((await page.locator('#exportSummary').textContent()).includes('0 checkpoint steps'));
  assert(await page.locator('#start').isDisabled());
  const source=await (await page.request.get('http://127.0.0.1:8765')).text();
  const token=source.match(/const token='([^']+)'/)[1];
  const blocked=await page.request.post('http://127.0.0.1:8765/api/export',{headers:{'X-Training-Token':token},data:{run:'td3_corrected_step_v3_20261024_01',hardware:true}});
  assert.equal(blocked.status(),400);
  assert((await blocked.json()).error.includes('Hardware export blocked'));
  await page.click('[data-doc="export"]');
  await page.locator('#docs').waitFor({state:'visible'});
  assert((await page.locator('#docText').textContent()).includes('Scaling joint-angle commands by 0.8'));
  await page.click('#closeDoc');
  await page.click('#researchExport');
  await page.locator('#downloadExport').waitFor({state:'visible',timeout:120000});
  assert((await page.locator('#exportMessage').textContent()).includes('Research only'));
  const [download]=await Promise.all([page.waitForEvent('download'),page.click('#downloadExport')]);
  assert(download.suggestedFilename().endsWith('.zip'));assert.equal(await download.failure(),null);
  await page.screenshot({path:'runs/training_ui_export_preview.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  assert.deepEqual(errors,[]);
  console.log('PASS: hardware export blocked in UI and API; export guide; research ONNX package; download; live training protected; mobile layout; no JavaScript errors.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
