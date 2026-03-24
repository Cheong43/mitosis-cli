const resp = await fetch('https://html.duckduckgo.com/html/?q=typescript+agent+framework', {headers:{'User-Agent':'Mozilla/5.0'}});
const html = await resp.text();

// Show HTML around result__a
const idx = html.indexOf('result__a');
if(idx>0) console.log('AROUND result__a:', html.slice(Math.max(0,idx-200), idx+500));

// Show snippet class
const sidx = html.indexOf('result__snippet');
if(sidx>0) console.log('\nAROUND result__snippet:', html.slice(Math.max(0,sidx-100), sidx+400));

// Check for uddg param
const uddgMatch = html.match(/uddg=([^&"]+)/);
if(uddgMatch) console.log('\nUDDG real URL:', decodeURIComponent(uddgMatch[1]));

// Extract all results with snippet
const linkRe = /<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
const snippetRe = /<a[^>]*class="result__snippet"[^>]*>([\s\S]*?)<\/a>/gi;
let m, i=0;
const links = [];
while((m=linkRe.exec(html)) && i<5) {
  const rawUrl = m[1];
  const title = m[2].replace(/<[^>]+>/g,'').trim();
  // Extract real URL from uddg param
  const uddg = rawUrl.match(/uddg=([^&]+)/);
  const realUrl = uddg ? decodeURIComponent(uddg[1]) : rawUrl;
  links.push({title, rawUrl: rawUrl.slice(0,120), realUrl});
  i++;
}

let sm;
i=0;
const snippets = [];
while((sm=snippetRe.exec(html)) && i<5) {
  snippets.push(sm[1].replace(/<[^>]+>/g,'').trim().slice(0,200));
  i++;
}

console.log('\n=== PARSED RESULTS ===');
links.forEach((l,i) => {
  console.log(`\n[${i+1}] ${l.title}`);
  console.log(`    realUrl: ${l.realUrl}`);
  console.log(`    snippet: ${snippets[i] || '(none)'}`);
});
