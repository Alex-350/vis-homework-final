(function(){const TYPE="denoise";
let seed=91,rows=[],currentData=null,currentInputs=[],selected=30;
const $=s=>document.getElementById(s);
function rng(s){return()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296}}
// Exact k-d tree search: build the fixed training index once, avoid sorting
// all 800 distances for every pixel on every slider input.
const neighborIndexes=new WeakMap();
function knn(x,train,k,reg=false){
 let tree=neighborIndexes.get(train);
 if(!tree){
  const build=(items,depth)=>{if(!items.length)return null;let axis=depth%x.length;items.sort((a,b)=>a.d.x[axis]-b.d.x[axis]||a.i-b.i);let mid=items.length>>1;return {item:items[mid],axis,left:build(items.slice(0,mid),depth+1),right:build(items.slice(mid+1),depth+1)}};
  tree=build(train.map((d,i)=>({d,i})),0);neighborIndexes.set(train,tree);
 }
 let near=[];
 const search=node=>{if(!node)return;let {d,i}=node.item,dist=0;for(let j=0;j<x.length;j++)dist+=(d.x[j]-x[j])**2;
  let at=near.length;while(at>0&&(near[at-1].dist>dist||(near[at-1].dist===dist&&near[at-1].i>i)))at--;
  if(at<k){near.splice(at,0,{dist,i,y:d.y});if(near.length>k)near.pop()}
  let delta=x[node.axis]-d.x[node.axis];search(delta<=0?node.left:node.right);
  if(near.length<k||delta*delta<=near[near.length-1].dist)search(delta<=0?node.right:node.left);
 };
 search(tree);
 if(reg)return near.reduce((a,d)=>a+d.y,0)/near.length;
 let votes={};near.forEach(d=>votes[d.y]=(votes[d.y]||0)+1);return +Object.keys(votes).sort((a,b)=>votes[b]-votes[a])[0];
}
let fixedTraining=null;
function training(){if(fixedTraining)return fixedTraining;
 let train=[];
 if(TYPE==='reservoir'){let r=rng(17);for(let i=0;i<350;i++){let a=r(),b=r();train.push({x:[a,b],y:8+16*a+7*b+2*Math.sin(a*4)})}}
 else{let tr=make(17,.25),r=rng(29);for(let i=0;i<800;i++){let x=2+Math.floor(r()*60),y=8+Math.floor(r()*74),j=y*64+x;train.push({x:features(tr,x,y),y:TYPE==='denoise'?tr.clean[j]:tr.labels[j]})}}
 return fixedTraining=train;
}

function heat(canvas,arr,w,h,kind='amp'){let ctx=canvas.getContext('2d'),im=ctx.createImageData(w,h);arr.forEach((v,i)=>{let c;if(kind==='class')c=[[55,127,104],[205,145,67],[78,128,179]][Math.round(v)%3];else if(kind==='binary')c=v?[43,134,99]:[239,242,233];else {v=Math.max(-1,Math.min(1,v));c=v>0?[248,245-140*v,238-150*v]:[248+150*v,245+95*v,238]}im.data.set([...c,255],i*4)});let off=document.createElement('canvas');off.width=w;off.height=h;off.getContext('2d').putImageData(im,0,0);ctx.clearRect(0,0,500,300);ctx.imageSmoothingEnabled=false;ctx.drawImage(off,0,0,500,300)}
function lines(canvas,series,min,max){let ctx=canvas.getContext('2d');ctx.clearRect(0,0,500,300);ctx.strokeStyle='#dce4dc';for(let y=20;y<300;y+=50){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(500,y);ctx.stroke()}series.forEach(([a,color])=>{ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();a.forEach((v,i)=>{let x=i/(a.length-1)*480+10,y=285-(v-min)/(max-min)*270;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()})}
function metric(a,b,c){$('metrics').innerHTML=`<span>传统基线<b>${a}</b></span><span>学习方法<b>${b}</b></span><span>评价口径<b style="font-size:16px">${c}</b></span>`}
function make(s,noise){let r=rng(s),w=64,h=90,clean=[],labels=[],noisy=[],fault=27+Math.floor(r()*10),phase=r()*4;for(let y=0;y<h;y++)for(let x=0;x<w;x++){let zone=Math.max(0,Math.min(2,Math.floor((x+Math.sin(phase)*5)/w*3))),v;if(TYPE==='firstbreak'){let t=20+x*.45+Math.sin(x/10)*2+phase;v=y<t?0:Math.exp(-(y-t)/17)*Math.sin((y-t)*1.3);labels.push(Math.abs(y-t)<2?1:0)}else{let shift=x>fault?7:0;v=Math.sin((y+shift+2*Math.sin(x/14)+phase)*.55);if(TYPE==='facies'){v=zone===0?Math.sin(y*.55):zone===1?Math.sin(y*.55+x*.6):.5*Math.sin(y*1.8+x*1.2+Math.sin(x*y));labels.push(zone)}else labels.push(Math.abs(x-fault)<2?1:0)}clean.push(v);noisy.push(v+(r()-.5)*noise*2)}return {w,h,clean,noisy,labels,fault,phase}}
function features(d,x,y){let {w,h,noisy:a}=d,at=(dx,dy)=>a[Math.max(0,Math.min(h-1,y+dy))*w+Math.max(0,Math.min(w-1,x+dx))];if(TYPE==='denoise')return [-2,-1,0,1,2].map(dy=>at(0,dy));if(TYPE==='firstbreak'){let now=0,pre=0;for(let j=0;j<4;j++){now+=at(0,j)**2;pre+=at(0,-j-4)**2}return [now/4,pre/4,(now-pre)/4]};let diff=0,energy=0,vert=0;for(let j=-3;j<=3;j++){diff+=(at(-2,j)-at(2,j))**2;energy+=at(0,j)**2;vert+=(at(0,j)-at(0,j-1))**2}return [diff/7,energy/7,vert/7]}
function run(){const noise=+$('noise').value/100,k=+$('k').value;$('nv').value=noise.toFixed(2);$('kv').value=k;$('seed').textContent=seed;rows=[];
if(TYPE==='reservoir'){currentData=null;currentInputs=[];let train=training(),r=rng(seed);let truth=[],pred=[],baseline=[];for(let i=0;i<100;i++){let a=r(),b=r(),y=8+16*a+7*b+2*Math.sin(a*4),x=[a+(r()-.5)*noise,b+(r()-.5)*noise],p=knn(x,train,k,true);currentInputs.push(x);truth.push(y);pred.push(p);baseline.push(22);rows.push([i,y,p,22])}let mae=a=>a.reduce((s,v,i)=>s+Math.abs(v-truth[i]),0)/100;metric(mae(baseline).toFixed(2),mae(pred).toFixed(2),'MAE / 孔隙度百分点 ↓');lines($('a'),[[truth,'#c88d3d'],[baseline,'#80918d']],0,40);lines($('b'),[[truth,'#c88d3d'],[pred,'#267e65']],0,40);$('leftTitle').textContent='测试点序号 → 孔隙度（%）：真值与均值基线';$('rightTitle').textContent='测试点序号 → 孔隙度（%）：真值与预测';$('explain').textContent='350 个训练点、100 个独立测试点。仅在测试输入属性中加入噪声。基线固定为训练关系均值附近的 22%。预测采用邻居孔隙度平均。曲线横轴是测试点序号，不代表深度。';}
else{let d=make(seed,noise);currentData=d;let train=training(),reg=TYPE==='denoise';let pred=[],baseline=[],truth=d.labels;for(let y=0;y<90;y++)for(let x=0;x<64;x++){let f=features(d,x,y),j=y*64+x;pred.push(knn(f,train,k,reg));baseline.push(TYPE==='fault'?(f[0]>.65?1:0):TYPE==='facies'?0:TYPE==='firstbreak'?(f[0]/(f[1]+.01)>3?1:0):d.noisy[j])}
heat($('a'),d.noisy,64,90);
if(TYPE==='denoise'){let snr=a=>10*Math.log10(d.clean.reduce((s,v)=>s+v*v,0)/a.reduce((s,v,i)=>s+(v-d.clean[i])**2,0));metric((Number.isFinite(snr(baseline))?snr(baseline).toFixed(2):"∞（无噪声）"),(Number.isFinite(snr(pred))?snr(pred).toFixed(2):"∞（零误差）"),'SNR / dB ↑');heat($('b'),pred,64,90);truth=d.clean;$('explain').textContent='800 个独立合成窗口建立五点波形到干净中心振幅的回归映射。基线为未经处理的含噪数据。观察噪声降低时，断层和薄层是否同时被平滑。';}
else if(TYPE==='firstbreak'){let t=[],p=[],b=[],missing=0;for(let x=0;x<64;x++){let find=a=>{for(let y=8;y<88;y++)if(a[y*64+x])return y;return 89};t.push((20+x*.45+Math.sin(x/10)*2+d.phase)*2);p.push(find(pred)*2);b.push(find(baseline)*2);if(find(pred)===89)missing++}let mae=a=>a.reduce((s,v,i)=>s+Math.abs(v-t[i]),0)/64;metric(mae(b).toFixed(2),mae(p).toFixed(2),'MAE / ms ↓');lines($('b'),[[t,'#c88d3d'],[p,'#267e65'],[b,'#84948e']],0,180);$('rightTitle').textContent='横轴道号 / 纵轴初至时间 ms（0–180）';$('explain').textContent=`分类器逐采样点判断初至邻域，取最早阳性点。基线为短前窗能量比阈值。未拾取 ${missing} 道按 178 ms 计入误差，不能用删除失败样本提升成绩。绿色为学习，橙色为真值，灰色为基线。`;rows=t.map((v,i)=>[i,v,p[i],b[i]]);}
else{const score=a=>{if(TYPE==='facies')return a.filter((v,i)=>v===truth[i]).length/a.length;let tp=0,fp=0,fn=0;a.forEach((v,i)=>{if(v&&truth[i])tp++;else if(v)fp++;else if(truth[i])fn++});return 2*tp/Math.max(1,2*tp+fp+fn)};metric((score(baseline)*100).toFixed(1)+'%',(score(pred)*100).toFixed(1)+'%',TYPE==='fault'?'断层像素 F1 ↑':'地震相准确率 ↑');heat($('b'),pred,64,90,TYPE==='fault'?'binary':'class');$('explain').textContent=TYPE==='fault'?`绿色为预测断层。真值断层带是第 ${d.fault-1} 至 ${d.fault+1} 道。基线为左右波形差阈值；kNN 用独立场景样本学习判别边界。F1 同时惩罚漏检与误检。`:'绿 / 橙 / 蓝分别表示平行、倾斜、杂乱反射。真值分区随剖面变化，可切换右图为真值核对。基线将所有位置判为第一相；可观察噪声对属性分类的影响。';}
if(!rows.length)rows=pred.map((v,i)=>[i,truth[i],v,baseline[i]]);}
$('details').innerHTML='<p>结果前 6 项（完整结果可导出）</p><table><tr><th>序号</th><th>真值</th><th>学习结果</th><th>基线</th></tr>'+rows.slice(0,6).map(r=>'<tr>'+r.map(v=>'<td>'+Number(v).toFixed(3)+'</td>').join('')+'</tr>').join('')+'</table>';}
let pendingFrame=0;
function showValues(){$('nv').value=(+$('noise').value/100).toFixed(2);$('kv').value=$('k').value}
function scheduleUpdate(){showValues();if(!pendingFrame)pendingFrame=requestAnimationFrame(()=>{pendingFrame=0;run()})}
function flushUpdate(){if(pendingFrame){cancelAnimationFrame(pendingFrame);pendingFrame=0}run()}
const rawRun=run;run=function(){rawRun();inspect();diagnostic();applyView()};$('run').onclick=flushUpdate;
for(const id of ['noise','k']){$(id).oninput=scheduleUpdate;$(id).onchange=flushUpdate}
$('new').onclick=()=>{seed+=13;flushUpdate()};$('reset').onclick=()=>{seed=91;$('noise').value=30;$('k').value=5;selected=30;flushUpdate()};$('csv').onclick=()=>{if(pendingFrame)flushUpdate();let blob=new Blob(['\ufeffindex,truth,prediction,baseline\n'+rows.map(r=>r.join(',')).join('\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=TYPE+'-seed'+seed+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};

function inspect(){let isTrace=TYPE==='firstbreak',isPoint=TYPE==='reservoir',max=isTrace?63:isPoint?99:5759;selected=Math.max(0,Math.min(max,selected));$('point').max=max;$('point').value=selected;let row=rows[selected],x=selected%64,y=Math.floor(selected/64),f=isTrace?[]:isPoint?currentInputs[selected]:features(currentData,x,y);$('position').textContent=isTrace?'道 '+selected:isPoint?'测试点 '+selected:'道 '+x+' / '+(y*2)+' ms';$('point-values').innerHTML=[['输入特征',f.length?f.map(v=>v.toFixed(3)).join(' / '):'该道短窗 / 前窗能量'],['合成真值',Number(row[1]).toFixed(3)],['学习结果',Number(row[2]).toFixed(3)],['传统基线',Number(row[3]).toFixed(3)]].map(([k,v])=>'<div><span>'+k+'</span><b>'+v+'</b></div>').join('');$('point-explain').textContent=isTrace?'时间单位 ms；未拾取道按 178 ms 纳入误差，不能删除失败道。模型取最早阳性点，可能提前触发。':isPoint?'属性为无量纲合成代理；孔隙度单位 %，误差为百分点。横轴是独立测试点序号，不是连续深度。':'特征只从当前含噪波形计算。测试真值只用于评价，不参与 kNN 邻域搜索；图像时间采样间隔为 2 ms。';}
function diagnostic(){let n=rows.length,bins=Array.from({length:64},(_,k)=>{let a=rows.slice(Math.floor(k*n/64),Math.max(Math.floor(k*n/64)+1,Math.floor((k+1)*n/64)));return a.reduce((s,r)=>s+((TYPE==='fault'||TYPE==='facies')?+(r[2]!==r[1]):Math.abs(r[2]-r[1])),0)/a.length}),max=Math.max(.001,...bins);$('diagnostic').innerHTML='<text x="35" y="17" fill="#79828a" font-size="11">分组平均绝对误差（分类为错误比例）；纵轴上限 '+max.toFixed(3)+'</text>'+bins.map((v,k)=>'<rect x="'+(35+k*10)+'" y="'+(130-v/max*95)+'" width="8" height="'+(v/max*95)+'" fill="#b26a39"/>').join('')+'<text x="35" y="155" fill="#79828a" font-size="10">样本顺序 →；地震像素按时间行展开，不能把此图解释成空间误差地图</text>';}
$('point').addEventListener('input',()=>{selected=+$('point').value;inspect()});
for(let id of ['a','b'])$(id).addEventListener('click',ev=>{let r=$(id).getBoundingClientRect(),x=Math.max(0,Math.min(63,Math.floor((ev.clientX-r.left)/r.width*64))),y=Math.max(0,Math.min(89,Math.floor((ev.clientY-r.top)/r.height*90)));selected=TYPE==='reservoir'?Math.min(99,Math.floor((ev.clientX-r.left)/r.width*100)):TYPE==='firstbreak'?x:y*64+x;inspect()});
function applyView(){let mode=$('view').value;if(mode==='pred')return;if(TYPE==='reservoir'||TYPE==='firstbreak'){let a=rows.map(r=>mode==='truth'?r[1]:r[2]-r[1]);lines($('b'),[[a,mode==='truth'?'#2f7d6b':'#b26a39']],mode==='truth'?0:-40,TYPE==='reservoir'?40:180)}else{let a=rows.map(r=>mode==='truth'?r[1]:TYPE==='fault'||TYPE==='facies'?+(r[1]!==r[2]):r[2]-r[1]);heat($('b'),a,64,90,mode==='error'&&(TYPE==='fault'||TYPE==='facies')?'binary':TYPE==='facies'?'class':TYPE==='fault'?'binary':'amp')}}$('view').addEventListener('change',()=>run());

run();

})();