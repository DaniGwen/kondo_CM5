const {chromium}=require('C:/Users/gwenski/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try {
  const page=await browser.newPage({viewport:{width:1280,height:1100}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8765');
  await page.waitForFunction(()=>document.getElementById('run').textContent.includes('td3_corrected_step_v3'));
  assert(await page.locator('#start').isDisabled());
  assert(await page.locator('#stop').isEnabled());
  await page.selectOption('#policy','SAC');assert.equal(await page.inputValue('#rate'),'0.0003');
  await page.fill('#step','1.25');await page.click('#reset');assert.equal(await page.inputValue('#step'),'0.75');
  await page.selectOption('#policy','TD3');assert.equal(await page.inputValue('#rate'),'0.0001');
  for(const key of ['model','new-world','stepping','observations','launcher']){
   await page.click(`[data-doc="${key}"]`);
   await page.locator('#docs').waitFor({state:'visible'});
   assert((await page.locator('#docText').textContent()).length>100);
   await page.click('#closeDoc');
  }
  const source=await (await page.request.get('http://127.0.0.1:8765')).text();
  const token=source.match(/const token='([^']+)'/)[1];
  const response=await page.request.post('http://127.0.0.1:8765/api/start',{headers:{'X-Training-Token':token},data:{name:'ui_duplicate_launch_test',policy:'TD3',steps:1000,learning_rate:.0001,rewards:{torque:.75,saturation:.75,tracking:.25,step:.75}}});
  assert.equal(response.status(),400);assert((await response.json()).error.includes('already active'));
  await page.screenshot({path:'runs/training_ui_preview.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.screenshot({path:'runs/training_ui_mobile_preview.png',fullPage:true});
  assert.deepEqual(errors,[]);
  console.log('PASS: live status, duplicate-launch protection, policy defaults, reward defaults, five documentation dialogs, desktop/mobile layout, no JavaScript errors.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
