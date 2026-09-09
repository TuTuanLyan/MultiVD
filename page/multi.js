/* ---- MS: o chon NHIEU gia tri, gon nhu mot nut ----
   `<select>` mot lua chon khong du: nguoi dung muon so sanh 2-3 khoi/nhanh cung luc.
   `<select multiple>` thi trinh duyet ve thanh list-box cao ngoang, hong ca thanh cong cu.
   Nen dung mot nut mo panel checkbox: gon khi dong, day du khi mo. */
function MS(host, id, label, values, onChange, preset){
  const sel = new Set(preset || []);
  const w = document.createElement('span'); w.className = 'ms'; w.style.position = 'relative';
  const btn = document.createElement('button'); btn.type = 'button'; btn.className = 'msb';
  const pan = document.createElement('div'); pan.className = 'msp'; pan.hidden = true;
  pan.innerHTML = values.map(v =>
    `<label><input type="checkbox" value="${String(v).replace(/"/g,'&quot;')}"> ${v}</label>`).join('')
    + '<div class="msf"><button type="button" data-a="all">chọn hết</button>'
    + '<button type="button" data-a="none">bỏ hết</button></div>';
  function label_(){
    btn.textContent = sel.size === 0 ? `${label}: tất cả`
      : sel.size === 1 ? `${label}: ${[...sel][0]}`
      : `${label}: ${sel.size} mục`;
    btn.classList.toggle('on', sel.size > 0);
  }
  pan.addEventListener('change', e => {
    if (e.target.type !== 'checkbox') return;
    e.target.checked ? sel.add(e.target.value) : sel.delete(e.target.value);
    label_(); onChange();
  });
  pan.addEventListener('click', e => {
    const a = e.target.dataset && e.target.dataset.a; if (!a) return;
    pan.querySelectorAll('input').forEach(i => { i.checked = (a === 'all'); });
    sel.clear(); if (a === 'all') values.forEach(v => sel.add(String(v)));
    label_(); onChange();
  });
  btn.addEventListener('click', e => {
    e.stopPropagation();
    document.querySelectorAll('.msp').forEach(p => { if (p !== pan) p.hidden = true; });
    pan.hidden = !pan.hidden;
  });
  document.addEventListener('click', () => { pan.hidden = true; });
  pan.addEventListener('click', e => e.stopPropagation());
  w.appendChild(btn); w.appendChild(pan); host.appendChild(w);
  label_();
  return { has: v => sel.size === 0 || sel.has(String(v)), size: () => sel.size,
           clear(){ sel.clear(); pan.querySelectorAll('input').forEach(i=>i.checked=false); label_(); },
           set(vs){ sel.clear(); pan.querySelectorAll('input').forEach(i=>{
             i.checked = vs.includes(i.value); if(i.checked) sel.add(i.value); }); label_(); } };
}
