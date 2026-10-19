/* 苍穹签到系统 — 前端逻辑。原生 JS，无框架。 */

// ---------- API 封装 ----------
const api = {
  async request(method, url, body) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(url, opts);
    if (!res.ok) {
      let msg = `请求失败 (${res.status})`;
      try { msg = (await res.json()).detail || msg; } catch (e) {}
      throw new Error(msg);
    }
    return res.status === 204 ? null : res.json();
  },
  get: (u) => api.request('GET', u),
  post: (u, b) => api.request('POST', u, b),
  put: (u, b) => api.request('PUT', u, b),
  del: (u) => api.request('DELETE', u),
};

const $ = (sel) => document.querySelector(sel);

// ---------- 导航切换 ----------
document.querySelectorAll('nav a').forEach(a => {
  a.addEventListener('click', (e) => {
    e.preventDefault();
    switchPage(a.dataset.page);
  });
});

// ---------- 后端连接状态 ----------
async function checkHealth() {
  const dot = $('#statusDot'), text = $('#statusText');
  try {
    const d = await api.get('/api/health');
    dot.className = 'dot ok';
    const ready = d.model_ready && d.liveness_ready && d.landmark_ready;
    text.textContent = d.model_ready
      ? (ready ? '后端已连接 · 防代签模型就绪' : '后端已连接 · 缺少防代签模型（眨眼/张嘴校验不可用）')
      : '后端已连接 · 模型加载中';
  } catch (e) {
    dot.className = 'dot bad';
    text.textContent = '后端未连接';
  }
}

// ---------- 当前设备能力（决定界面上显示哪些入口） ----------
// 由后端判定：只有「在这台运行系统的电脑上」打开时才拥有完整功能。
// 手机走局域网或公网隧道访问时拿到的是受限能力 —— 与后端 403 拦截共用同一套判断，
// 所以界面藏掉的东西，接口层面也真的调不通。
let caps = {
  is_local_device: true, can_edit_faces: true, can_attendance: true,
  can_view_roster: true, can_manage: true, can_view_records: true,
  can_clear_recent: true,
};

async function loadCapabilities() {
  try {
    caps = Object.assign({}, caps, await api.get('/api/capabilities'));
  } catch (e) {
    // 拿不到就按「最小权限」处理：宁可少给功能，也不要多给
    caps = {
      is_local_device: false, can_edit_faces: false, can_attendance: true,
      can_view_roster: false, can_manage: false, can_view_records: false,
      can_clear_recent: false,
    };
  }
  applyCapabilities();
}

/** 按能力裁剪界面。手机端只保留「实时签到」一个页面。 */
function applyCapabilities() {
  const local = caps.is_local_device;

  // 让 CSS 也能感知设备类型（例如隐藏名单面板）
  document.body.dataset.device = local ? 'local' : 'remote';

  // ① 导航：手机端只剩「实时签到」（签到记录里是全班名单，不给手机看）
  const allowed = ['attendance'];
  document.querySelectorAll('nav a').forEach(a => {
    a.hidden = !local && !allowed.includes(a.dataset.page);
  });

  // ② 手机端若停在被隐藏的页面（默认落在「学生管理」），切到实时签到
  if (!local && !allowed.includes(currentPage())) {
    switchPage('attendance');
  }

  // ③ 手机端不展示「全班到了谁、没到谁」的名单，只显示本次签到结果
  const roster = $('#attRosterPanel');
  if (roster) roster.hidden = !caps.can_view_roster;
  if (!caps.can_view_roster) { const c = $('#attCount'); if (c) c.textContent = '0 / 0'; }

  // ③b 手机端工具栏只留「选定课程 + 摄像头 + 开始/结束签到」
  const clearBtn = $('#btnClearAtt');
  if (clearBtn) clearBtn.hidden = !local;          // 清空签到记录属于管理操作
  const onlyAbsentBox = $('#onlyAbsent');
  if (onlyAbsentBox) {
    const lbl = onlyAbsentBox.closest('label');
    if (lbl) lbl.hidden = !caps.can_view_roster;   // 「只看未签到」依附于名单，没有名单就没有意义
  }

  // ③c 「最近识别缓存」状态条含全班姓名，同样只给本机看
  const recentBox = $('#recentCacheBox');
  if (recentBox) recentBox.hidden = !caps.can_clear_recent;

  // ④ 学生管理页的提示文案
  const hint = $('#studentsHint');
  if (hint) {
    hint.textContent = caps.can_edit_faces
      ? '注册人脸后，学生才能被识别签到'
      : '手机端仅支持刷脸签到，人脸录入请在电脑上操作';
  }
  renderStudents();
}

/** 当前处于哪个页面。 */
function currentPage() {
  const a = document.querySelector('nav a.active');
  return a ? a.dataset.page : '';
}

/** 切换页面（导航点击与能力裁剪共用）。 */
function switchPage(page) {
  document.querySelectorAll('nav a').forEach(x => x.classList.toggle('active', x.dataset.page === page));
  document.querySelectorAll('.page').forEach(x => x.classList.toggle('active', x.id === 'page-' + page));
  if (page === 'students') loadStudents();
  if (page === 'courses') loadCourses();
  if (page === 'attendance') { loadCourses().then(refreshRosterIfCourseSelected); refreshRecentCache(true); }
  if (page === 'records') loadCourses();
  // 缓存倒计时只在签到页有意义，离开就停掉，别让定时器空转
  if (page === 'attendance') startRecentTicker(); else stopRecentTicker();
}

// ---------- 学生管理 ----------
let students = [];
let editingStudentId = null;

async function loadStudents() {
  try {
    const cls = $('#classFilter').value;
    students = await api.get(cls ? `/api/students?class_name=${encodeURIComponent(cls)}` : '/api/students');
    renderStudents();
    renderClassFilter();
  } catch (e) { alert('加载学生失败：' + e.message); }
}

function renderStudents() {
  const tbody = $('#studentTable tbody');
  if (!students.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">暂无学生，点击右上角「新增学生」</td></tr>';
    return;
  }
  const canFace = caps.can_edit_faces;
  tbody.innerHTML = students.map(s => `
    <tr>
      <td>${esc(s.name)}</td>
      <td>${esc(s.student_no)}</td>
      <td>${esc(s.class_name) || '—'}</td>
      <td>${s.faces_total > 0 ? '<span class="badge ok">已注册 ' + s.faces_total + ' 张</span>' : '<span class="badge none">未注册</span>'}</td>
      <td class="col-actions">
        ${canFace ? `<button class="btn" onclick="openFaceModal(${s.id}, '${esc(s.name)}')">注册人脸</button>` : ''}
        <button class="btn" onclick="openStudentModal(${s.id})">编辑</button>
        ${canFace ? `<button class="btn danger" onclick="deleteStudent(${s.id})">删除</button>` : ''}
      </td>
    </tr>`).join('');
}

async function renderClassFilter() {
  const sel = $('#classFilter');
  const cur = sel.value;
  const classes = [...new Set(students.map(s => s.class_name).filter(Boolean))];
  sel.innerHTML = '<option value="">全部班级</option>' + classes.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join('');
  sel.value = cur;
}

function openStudentModal(id) {
  editingStudentId = id;
  $('#studentModalTitle').textContent = id ? '编辑学生' : '新增学生';
  if (id) {
    const s = students.find(x => x.id === id);
    $('#fName').value = s.name; $('#fNo').value = s.student_no; $('#fClass').value = s.class_name;
  } else {
    $('#fName').value = ''; $('#fNo').value = ''; $('#fClass').value = '';
  }
  $('#studentModal').hidden = false;
}

async function saveStudent() {
  const body = { name: $('#fName').value.trim(), student_no: $('#fNo').value.trim(), class_name: $('#fClass').value.trim() };
  if (!body.name || !body.student_no) { alert('姓名和学号不能为空'); return; }
  try {
    if (editingStudentId) await api.put(`/api/students/${editingStudentId}`, body);
    else await api.post('/api/students', body);
    $('#studentModal').hidden = true;
    loadStudents();
  } catch (e) { alert(e.message); }
}

async function deleteStudent(id) {
  const s = students.find(x => x.id === id);
  if (!confirm(`确定删除学生「${s.name}」？其人脸与签到记录将一并删除。`)) return;
  try { await api.del(`/api/students/${id}`); loadStudents(); }
  catch (e) { alert(e.message); }
}

// ---------- 人脸注册 ----------
let regStudentId = null;

async function openFaceModal(id, name) {
  if (!caps.can_edit_faces) {
    alert('人脸录入只能在这台运行签到系统的电脑上操作。\n手机端仅支持签到。');
    return;
  }
  if (attRunning) {
    alert('实时签到正在进行中，请先点「结束签到」再注册人脸——摄像头不能同时被两个功能占用');
    return;
  }
  regStudentId = id;
  $('#faceStudentName').textContent = name;
  $('#faceModal').hidden = false;
  $('#regEmpty').style.display = 'block';
  $('#regEmpty').textContent = '正在启动摄像头…';
  $('#faceHint').textContent = '请正对摄像头，点击「拍摄并注册」';
  refreshDevices();   // 与签到一致：枚举放在开流之前，避免枚举把正在用的流掐掉
  try {
    await Camera.open($('#regVideo'), { owner: 'register' });
    $('#regEmpty').style.display = 'none';
    $('#faceHint').textContent = '请正对摄像头，光线充足，点击「拍摄并注册」';
  } catch (e) {
    $('#regEmpty').style.display = 'block';
    $('#regEmpty').textContent = e.message;
    $('#faceHint').textContent = '摄像头不可用 · 可点「关闭」退出';
  }
}

function closeFaceModal() {
  try { Camera.close(); } catch (e) { /* 清理失败不影响关闭弹窗 */ }
  $('#faceModal').hidden = true;
  regStudentId = null;
}

async function captureAndRegister() {
  if (!regStudentId) return;
  const data = Camera.capture($('#regVideo'), document.createElement('canvas'));
  if (!data) { alert('摄像头尚未就绪或画面全黑：请稍等 1~2 秒，或检查光线、更换摄像头后重试'); return; }
  $('#faceHint').textContent = '正在识别并注册…';
  try {
    await api.post(`/api/students/${regStudentId}/faces`, { image: data });
    closeFaceModal();
    loadStudents();
    alert('人脸注册成功！');
  } catch (e) {
    $('#faceHint').textContent = '注册失败：' + e.message;
  }
}

// ---------- 课程管理 ----------
let courses = [];

async function loadCourses() {
  try {
    courses = await api.get('/api/courses');
    renderCourses();
    renderCourseSelects();
  } catch (e) { /* 静默 */ }
}

function renderCourses() {
  const tbody = $('#courseTable tbody');
  if (!courses.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty">暂无课程，点击右上角「新增课程」</td></tr>';
    return;
  }
  tbody.innerHTML = courses.map(c => `
    <tr>
      <td>${esc(c.name)}</td>
      <td>${esc(c.class_name) || '—'}</td>
      <td>${esc(c.created_at)}</td>
      <td class="col-actions"><button class="btn danger" onclick="deleteCourse(${c.id})">删除</button></td>
    </tr>`).join('');
}

function renderCourseSelects() {
  ['attCourse', 'recCourse'].forEach(id => {
    const sel = $('#' + id);
    const prev = sel.value;   // 保留已选课程，避免切换页面后被重置
    sel.innerHTML = '<option value="">请选择课程</option>' + courses.map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
    if (prev && courses.some(c => String(c.id) === prev)) sel.value = prev;
  });
}

async function addCourse() {
  const name = prompt('课程名称：');
  if (!name) return;
  const class_name = prompt('班级（可留空）：') || '';
  try { await api.post('/api/courses', { name, class_name }); loadCourses(); }
  catch (e) { alert(e.message); }
}

async function deleteCourse(id) {
  if (!confirm('确定删除该课程？其签到记录将一并删除。')) return;
  try { await api.del(`/api/courses/${id}`); loadCourses(); }
  catch (e) { alert(e.message); }
}

// ---------- 摄像头设备选择 ----------
let cameras = [];
let renderingDeviceSelects = false;   // 正在重建下拉，屏蔽程序自身造成的 change

function deviceLabel(d, i) {
  return d.label || `摄像头 ${i + 1}`;
}

/** 枚举可用摄像头。注意：未授权前 label 为空，授权后需再枚举一次才有真实名称。 */
async function refreshDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  try {
    const all = await navigator.mediaDevices.enumerateDevices();
    cameras = all.filter(d => d.kind === 'videoinput');
    if (Camera.deviceId && !cameras.some(d => d.deviceId === Camera.deviceId)) {
      Camera.setDevice(null);   // 之前记住的设备已不存在
    }
    renderDeviceSelects();
  } catch (e) { /* 静默：设备枚举失败不影响主流程 */ }
}

function renderDeviceSelects() {
  // 重建 <option> 时部分浏览器会「误发」一次 change 事件，导致摄像头被意外重开。
  // 这里用标志位屏蔽掉程序自身造成的 change。
  renderingDeviceSelects = true;
  try {
    document.querySelectorAll('select[data-camera-device]').forEach(sel => {
      const prev = sel.value || Camera.deviceId || '';
      const opts = cameras.map((d, i) => {
        const name = deviceLabel(d, i) + (isBuiltinLabel(d.label) ? '（内置）' : '');
        return `<option value="${esc(d.deviceId)}">${esc(name)}</option>`;
      }).join('');
      sel.innerHTML = '<option value="">自动 · 优先内置摄像头</option>' + opts;
      sel.value = cameras.some(d => d.deviceId === prev) ? prev : '';
    });
  } finally {
    // 异步复位：change 事件可能在当前任务之后才派发
    setTimeout(() => { renderingDeviceSelects = false; }, 0);
  }
}

/** 仅用于在下拉里加「（内置）」标注，帮用户认设备；实际选择靠 facingMode 约束，不依赖它。 */
function isBuiltinLabel(label) {
  if (!label) return false;
  return /integrated|internal|built-?in|内置|内建|前置|front|facetime|user facing/i.test(label);
}

async function onDeviceChange(e) {
  if (renderingDeviceSelects) return;                 // 程序自己重建下拉造成的假 change
  const next = e.target.value || null;
  if (next === Camera.deviceId) return;               // 值没变，什么都不做
  Camera.setDevice(next);
  renderDeviceSelects();

  // 签到进行中：直接热切换（不用再「结束签到 → 重新开始」）
  if (attRunning) {
    if (camRecovering) return;
    camRecovering = true;
    setAttStatus('正在切换摄像头…');
    try { Camera.close(); } catch (err) { /* ignore */ }
    await new Promise(r => setTimeout(r, 400));
    try {
      await Camera.open($('#attVideo'), { owner: 'attendance' });
      if (!attRunning) { Camera.close(); camRecovering = false; return; }
      blackStreak = 0; netFails = 0; blackRecovers = 0;
      setAttStatus('已切换摄像头，继续签到');
    } catch (err) {
      setAttStatus('切换摄像头失败：' + err.message);
    }
    camRecovering = false;
    updateCamDiag();
    return;
  }

  if (!$('#faceModal').hidden) reopenFaceCamera();   // 注册窗口开着时立即切换
}

async function reopenFaceCamera() {
  $('#regEmpty').style.display = 'block';
  $('#regEmpty').textContent = '正在启动摄像头…';
  try {
    await Camera.open($('#regVideo'), { owner: 'register' });
    $('#regEmpty').style.display = 'none';
    $('#faceHint').textContent = '请正对摄像头，光线充足，点击「拍摄并注册」';
  } catch (e) {
    $('#regEmpty').textContent = e.message;
    $('#faceHint').textContent = '摄像头不可用 · 可点「关闭」退出';
  }
}

// ---------- 实时签到 ----------
let attTimer = null;
let attRunning = false;
let attStarting = false;
let attRoster = [];          // [{id,name,student_no,class_name,status,check_in_at}]
let attCourseId = null;
let onlyAbsent = false;      // 「只看未签到」开关
let netFails = 0;
let blackStreak = 0;         // 连续「取不到有效画面」次数
let blackRecovers = 0;       // 因「持续全黑」触发的重连次数（防暗光环境下无限重连）
let camRecovering = false;   // 正在自动重连摄像头
// ---- 防代签（时序活体 + 动作挑战）相关状态 ----
let attAttempting = false;       // 正在采集动作序列/等待校验，期间不再发探测请求
let attCooldownUntil = 0;        // 验证结束后的冷却时间戳，避免同一人立刻被反复拉起
let probeStreak = 0;             // 连续「可以验证」的探测次数（连续 2 次才动作，防单帧误判）
let attFailStreak = 0;           // 连续失败次数，达阈值提示改用手工补签
const BURST_FRAMES = 22;         // 一次动作采集的帧数
const BURST_INTERVAL_MS = 100;   // 帧间隔（约 2.2~2.7 秒采完，覆盖「读指令→反应」的整个区间）
const attCanvas = document.createElement('canvas');

/** 拉取该课程「全部学生 + 今日签到状态」，作为右侧名单的数据源。
 *  手机端（can_view_roster=false）整块跳过：既不展示名单，就不去把全班名单拉下来。 */
async function loadAttRoster(courseId) {
  if (!caps.can_view_roster) { attRoster = []; return; }
  const course = await api.get(`/api/courses/${courseId}`);
  const students = await api.get(
    course.class_name ? `/api/students?class_name=${encodeURIComponent(course.class_name)}` : '/api/students'
  );
  const attended = await api.get(`/api/attendance?course_id=${courseId}&date=${todayStr()}`);
  const map = {};
  attended.forEach(a => { map[a.student_id] = a; });
  attRoster = students.map(s => ({
    id: s.id,
    name: s.name,
    student_no: s.student_no,
    class_name: s.class_name,
    status: map[s.id] ? 'present' : 'absent',
    method: map[s.id] ? (map[s.id].status || 'present') : '',   // present / manual
    check_in_at: map[s.id] ? map[s.id].check_in_at : '',
  }));
  renderAttRoster();
  refreshRecentCache(true);
}

function timePart(ts) {
  return ts && ts.length >= 19 ? ts.slice(11, 19) : (ts || '');
}

/** 渲染右侧名单：未签到在前（一眼看出谁没到），已签到在后并显示时间。 */
function renderAttRoster() {
  const ul = $('#attList');
  if (!caps.can_view_roster) return;    // 手机端没有名单面板
  const present = attRoster.filter(s => s.status === 'present').length;
  $('#attCount').textContent = `${present} / ${attRoster.length}`;
  if (!attRoster.length) {
    ul.innerHTML = '<li class="empty">该课程暂无学生，请先到「学生管理」添加</li>';
    return;
  }
  const list = onlyAbsent ? attRoster.filter(s => s.status === 'absent') : attRoster;
  if (!list.length) {
    ul.innerHTML = '<li class="empty">全部学生都已签到 🎉</li>';
    return;
  }
  const sorted = [...list].sort((a, b) => {
    if (a.status !== b.status) return a.status === 'absent' ? -1 : 1;
    return a.name.localeCompare(b.name, 'zh');
  });
  ul.innerHTML = sorted.map(s => `
    <li data-id="${s.id}" class="${s.status}">
      <span class="who">${esc(s.name)}<small>${esc(s.student_no)}</small></span>
      ${s.status === 'present'
        ? `<span class="time">${s.method === 'manual' ? '补签 ' : ''}${timePart(s.check_in_at)}</span>`
        : `<span class="time">未签到</span><button class="btn tiny manual" data-manual="${s.id}">补签</button>`}
    </li>`).join('');
}

/** 手工补签：动作校验连续失败时的兜底通道（仅本机，记录会标记为「补签」）。 */
async function manualSignIn(studentId) {
  const s = attRoster.find(x => x.id === studentId);
  const name = s ? `${s.name}（${s.student_no}）` : `学生 #${studentId}`;
  if (!attRunning || !attCourseId) { alert('请先开始签到并选择课程'); return; }
  if (!confirm(`确认为 ${name} 手工补签？\n\n用于人脸验证连续失败的情况（逆光、口罩、镜头脏等）。\n补签记录在签到记录里会标记为「手工补签」。`)) return;
  try {
    const r = await api.post('/api/attendance/manual', { course_id: attCourseId, student_id: studentId });
    markPresent({ student_id: r.student_id, name: r.name, student_no: r.student_no,
                  check_in_at: r.check_in_at, method: 'manual' });
    setAttStatus(r.result === 'attended' ? `已手工补签：${r.name}` : `${r.name} 已经签到过了`);
  } catch (e) {
    alert('补签失败：' + e.message);
  }
}

/** 清空该课程今日签到记录，便于重新签到。 */
async function clearAttendance() {
  const courseId = $('#attCourse').value;
  if (!courseId) { alert('请先选择课程'); return; }
  const courseName = $('#attCourse').selectedOptions[0] ? $('#attCourse').selectedOptions[0].text : '';
  if (!confirm(`确定清空【${courseName}】今天（${todayStr()}）的全部签到记录？\n\n清空后学生可以重新签到。此操作不可撤销。`)) return;
  try {
    const r = await api.del(`/api/attendance?course_id=${courseId}&date=${todayStr()}`);
    alert(`已清空该课程今日签到记录（${r.deleted} 条），可以重新开始签到了。`);
    if (String(attCourseId) === String(courseId)) await loadAttRoster(Number(courseId));
  } catch (e) {
    alert('清空失败：' + e.message);
  }
}

// ---------- 最近识别缓存（纯内存，默认 2 小时自动清空） ----------
// 后端只把它放在内存里：不落盘、不存画面，超过 TTL 自动消失，进程一退出也没了。
// 这里负责两件事：把状态显示出来，以及提供「立即清空」。
let recentTicker = null;
let lastRecentFetch = 0;

/** 把秒数说成人话：7200 → 「2 小时」 */
function fmtDuration(sec) {
  if (sec === null || sec === undefined) return '—';
  if (sec >= 3600) {
    const h = Math.floor(sec / 3600), m = Math.round((sec % 3600) / 60);
    return m ? `${h} 小时 ${m} 分` : `${h} 小时`;
  }
  if (sec >= 60) return `${Math.floor(sec / 60)} 分钟`;
  return '不到 1 分钟';
}

async function loadRecentCache() {
  const box = $('#recentCacheBox'), text = $('#recentCacheText');
  if (!box || !text || !caps.can_clear_recent) return;   // 手机端没有这块
  try {
    const d = await api.get('/api/recent');
    text.classList.remove('fail');
    if (!d.total) {
      text.innerHTML = `空 · 每 <b>${fmtDuration(d.ttl_seconds)}</b>自动清空一次`;
      return;
    }
    // 展示「最快多久会有第一条被清掉」，让老师直观看到倒计时在走。
    // 文案刻意压短：右侧面板只有 400px 宽，说多了就换行。
    // 「机制未运行」属于异常，才额外提一句。
    const soonest = Math.min(...d.items.map(i => i.expires_in));
    text.innerHTML =
      `内存中 <b>${d.total}</b> 人 · 最快 <b>${fmtDuration(soonest)}</b> 后自动清空` +
      (d.auto_clean_running ? '' : ' · <span class="rc-warn">清理未运行</span>');
  } catch (e) {
    text.classList.add('fail');
    text.textContent = '缓存状态读取失败：' + e.message;
  }
}

/** 识别过程中会被高频调用，做个节流，避免每帧都发一次请求。 */
function refreshRecentCache(force) {
  if (!caps.can_clear_recent) return;
  const now = Date.now();
  if (!force && now - lastRecentFetch < 800) return;
  lastRecentFetch = now;
  loadRecentCache();
}

function startRecentTicker() {
  stopRecentTicker();
  recentTicker = setInterval(() => refreshRecentCache(true), 30000);
}

function stopRecentTicker() {
  if (recentTicker) { clearInterval(recentTicker); recentTicker = null; }
}

/** 立即清空内存缓存。注意：只动内存里的识别痕迹，数据库里的签到记录一条不动。 */
async function clearRecentCache() {
  if (!caps.can_clear_recent) return;
  if (!confirm('确定立即清空「最近识别缓存」？\n\n' +
               '它只是内存里「刚才识别到了谁」的临时记录，\n' +
               '清空不影响数据库里的签到记录，也不会删除学生的人脸。')) return;
  try {
    const r = await api.del('/api/recent');
    await loadRecentCache();
    alert(`已清空最近识别缓存（${r.cleared} 条）。\n签到记录与人脸数据都不受影响。`);
  } catch (e) {
    alert('清空失败：' + e.message);
  }
}

/** 手机端：只在卡片上显示「刚被识别到的这个人」，不展示任何其他学生。
 *  already=true 表示这个人本次课程之前已经签过到了。 */
function showMyResult(info, already) {
  const el = $('#attMyResult');
  if (!el) return;
  el.hidden = false;
  const sub = already
    ? '已经签到过了'
    : `${esc(info.student_no || '')}${info.student_no ? ' · ' : ''}签到成功 ${esc(timePart(info.check_in_at))}`;
  el.innerHTML = `
    <div class="ok-mark">${already ? '·' : '✓'}</div>
    <div class="ok-text">
      <div class="ok-name">${esc(info.name)}</div>
      <div class="ok-sub">${sub}</div>
    </div>`;
}

function clearMyResult() {
  const el = $('#attMyResult');
  if (el) { el.hidden = true; el.innerHTML = ''; }
}

/** 识别到新签到 → 就地更新名单（不重新拉取，保证即时）。 */
function markPresent(info) {
  // 手机端不维护名单，改为只显示刚签到的这个人
  if (!caps.can_view_roster) { showMyResult(info); return; }
  const item = attRoster.find(s => s.id === info.student_id);
  if (item) {
    if (item.status === 'present') return;
    item.status = 'present';
    item.method = info.method || 'present';
    item.check_in_at = info.check_in_at || '';
  } else {
    // 名单外的学生（例如按班级筛选未覆盖）也补进来，避免「识别到了却不显示」
    attRoster.push({
      id: info.student_id, name: info.name, student_no: info.student_no || '',
      class_name: '', status: 'present', method: info.method || 'present',
      check_in_at: info.check_in_at || '',
    });
  }
  renderAttRoster();
  refreshRecentCache();   // 名单变了 → 顺带刷新「最近识别缓存」状态
}

async function startAttendance() {
  if (attRunning || attStarting) return;  // 双击保护
  if (!$('#faceModal').hidden) {
    alert('请先关闭「人脸注册」窗口——摄像头不能同时被两个功能占用');
    return;
  }
  const courseId = $('#attCourse').value;
  if (!courseId) { alert('请先选择课程'); return; }

  attStarting = true;
  $('#btnStartAtt').disabled = true;      // 立即禁用，避免启动期间重复触发
  $('#btnStopAtt').disabled = true;
  $('#attEmpty').style.display = 'block';
  $('#attEmpty').textContent = '正在加载签到名单…';
  attCourseId = Number(courseId);

  try {
    await loadAttRoster(attCourseId);
  } catch (e) {
    alert('加载签到名单失败：' + e.message + '\n（仍会继续启动摄像头，识别到的人会实时补进名单）');
  }

  $('#attEmpty').textContent = '正在启动摄像头…';
  // 【重要】设备枚举必须在开流「之前」做。
  // 部分 Windows 摄像头驱动在媒体流运行期间被 enumerateDevices() 重新枚举时，
  // 会把正在使用的流掐掉（表现为「刚开始好好的，识别完一个人就黑屏」），所以开流后不再枚举。
  refreshDevices();
  try {
    await Camera.open($('#attVideo'), { owner: 'attendance' });
    $('#attEmpty').style.display = 'none';
  } catch (e) {
    $('#attEmpty').textContent = '无法打开摄像头：' + e.message;
    $('#btnStartAtt').disabled = false;
    attStarting = false;
    return;
  }

  attStarting = false;
  attRunning = true;
  netFails = 0;
  blackStreak = 0;
  blackRecovers = 0;
  camRecovering = false;
  attAttempting = false;
  probeStreak = 0;
  attFailStreak = 0;
  attCooldownUntil = 0;
  hideChallenge();
  Camera.reconnects = 0;
  $('#btnStopAtt').disabled = false;
  setAttStatus('已开始，请正对摄像头');
  clearMyResult();
  updateCamDiag();
  attLoop();
}

/**
 * 签到主循环。调度放在 finally，任何单帧异常都不会中断整个循环。
 *
 * 为什么改成三段式（旧版是「每帧独立提交」）：
 *   旧版后端拿不到任何时间信息，只能靠单帧活体模型判断。实测一张静态照片的
 *   真人概率是 0.996 —— 单帧根本分不出照片和真人。现在拆成：
 *     ① probe     单帧探测，只在「出现一张够大的人脸」时唤醒验证，绝不写签到
 *     ② challenge 服务端随机指定动作（眨眼/张嘴）并签发 45 秒一次性令牌
 *     ③ verify    连续采集 1.35 秒，后端做时序活体 + 动作判定后才写签到
 *   照片做不出随机动作，预录视频也对不上当场才下发的指令 —— 这才真正拦住代签。
 */
function attLoop() {
  if (!attRunning) return;
  (async () => {
    try {
      if (camRecovering) return;   // 正在重连：等重连流程结束，不发无用的请求
      if (attAttempting) return;   // 正在采集/校验动作序列，不并发
      if (Date.now() < attCooldownUntil) return;   // 刚验证完，歇一下

      // ① 先判断「流」是否还活着。流断了（track 结束 / 尺寸归零）→ 立刻重连。
      //    这一步是关键：以前把「流断了」和「这一帧偏黑」混为一谈，所以只会提示不会恢复。
      if (!Camera.isAlive()) {
        const h = Camera.health();
        await recoverCamera(h.lastEvent || `画面尺寸 ${h.size}`);
        return;
      }

      // 探测帧与校验帧使用同一降采样参数，否则「探测说够、校验说太远」的阈值会对不上
      const data = Camera.captureSmall($('#attVideo'), attCanvas);
      if (!data) {
        blackStreak++;
        // 单帧取不到不算故障（自动曝光切换、挥手遮挡都会发生）；连续 3 帧才处理
        if (blackStreak >= 3) {
          const h = Camera.health();
          if (!Camera.isAlive()) { await recoverCamera(h.lastEvent || '无画面'); return; }
          setAttStatus('画面持续全黑：请检查摄像头是否被遮挡、镜头盖是否打开，或换一个摄像头');
          // 流显示「活着」却连续拿不到画面（驱动假死最常见）→ 也做一次重连自愈，
          // 但最多重连 2 次，避免用户真在暗光环境下被无限重连打断
          if (blackStreak >= 8 && blackRecovers < 2) {
            blackRecovers++;
            await recoverCamera('画面持续全黑');
            return;
          }
          updateCamDiag();
        }
      } else {
        blackStreak = 0;
        const p = await api.post('/api/attendance/probe', { image: data, course_id: attCourseId });
        netFails = 0;
        if (p.ready) {
          probeStreak++;
          // 连续两次探测都说「可以」才行动：单帧误判不会把学生拉进一次无用的验证
          if (probeStreak >= 2) {
            probeStreak = 0;
            await runChallenge();
          } else if (attFailStreak === 0) {
            // 只在「没有失败提示要保留」时才写这行，否则刚显示的失败原因会被立刻覆盖掉
            setAttStatus('已检测到人脸，请正对摄像头…');
          }
        } else {
          probeStreak = 0;
          // 人离开镜头就算「换人了」：失败计数归零，下一个人不会背前一个人的次数
          if (!p.faces) attFailStreak = 0;
          setAttStatus(p.hint || (p.faces ? '请正对摄像头，保持脸部完整' : '未检测到人脸，请正对摄像头并保证光线充足'));
        }
        updateCamDiag();
      }
    } catch (e) {
      netFails++;
      if (netFails >= 3) setAttStatus('与后端通信失败：' + e.message + '（请确认后端服务已启动）');
    } finally {
      if (attRunning) attTimer = setTimeout(attLoop, 300);
    }
  })();
}

/** 申请一次挑战 → 显示指令 → 采集帧序列 → 提交校验。
 *
 * 【关键】指令出现与开始采集必须同时发生，中间不能等。
 * 早期版本为了让学「看清指令」先等 600ms 再采集，结果学生一看到指令就闭眼/眨眼，
 * 动作正好落在等待期里，采集开始时已经结束 —— 真人做对了动作却判失败，主因就是这个。
 * 现在窗口拉长到 2.2 秒以上，覆盖 0.3~2 秒的各类反应速度。
 */
async function runChallenge() {
  if (!attRunning || attAttempting) return;
  attAttempting = true;
  try {
    const c = await api.post('/api/attendance/challenge', { course_id: attCourseId });
    // 关键：先拍一张「还没看到指令」的中性帧当基线。
    // 服务端用它当动作判据的参照 —— 学生若看到指令就立刻做动作并一直保持，
    // 用序列自己的分位数当基线会把「做过的」算成「没做」。
    const refFrame = Camera.captureSmall($('#attVideo'), attCanvas);
    showChallenge(c.instruction);
    const frames = await captureBurst();
    if (frames.length < 8) {
      setAttStatus('采集失败：画面不稳定，请保持不动后重试');
      return;
    }
    setChallengeNote('正在校验…');
    const res = await api.post('/api/attendance/verify', {
      course_id: attCourseId,
      challenge_id: c.challenge_id,
      frames,
      ref: refFrame || null,
    });
    handleVerifyResult(res);
  } catch (e) {
    setAttStatus('验证请求失败：' + e.message);
  } finally {
    hideChallenge();
    attAttempting = false;
    // 用 max 而不是直接赋值：成功路径已经设了更长的冷却（别被这里覆盖成 1.2 秒）
    attCooldownUntil = Math.max(attCooldownUntil, Date.now() + 1200);
  }
}

/** 连续采集帧序列（间隔 90ms × 16 帧 ≈ 1.35 秒，足以覆盖一次自然眨眼）。 */
async function captureBurst() {
  const out = [];
  for (let i = 0; i < BURST_FRAMES; i++) {
    if (!attRunning) break;
    const d = Camera.captureSmall($('#attVideo'), attCanvas);
    if (d) out.push(d);
    setChallengeProgress((i + 1) / BURST_FRAMES);
    if (i < BURST_FRAMES - 1) await sleepJs(BURST_INTERVAL_MS);
  }
  return out;
}

const VERIFY_HINT = {
  no_face: '没有检测到人脸，请正对摄像头',
  too_far: '请靠近摄像头一些，让脸部占画面更大',
  multiple_faces: '画面中只能有一个人，请让其他人离开镜头',
  spoof: '疑似照片或屏幕画面，请本人面对摄像头',
  identity_switch: '画面中的人前后不一致，请单人正对摄像头重试',
  action_missing: '没有检测到动作，请照着屏幕提示再做一次',
  no_match: '未识别出已注册的学生',
  not_enough_frames: '采集帧数不足，请重试',
  no_faces_registered: '尚未录入任何人脸，请先在本机录入',
  expired: '验证超时，请重新开始',
  used: '这次验证已失效，请重新开始',
  not_found: '验证已失效，请重新开始',
  course_mismatch: '验证与课程不匹配，请重新开始',
  bad_action: '指令异常，请重新开始',
};

/**
 * 失败提示：动作类失败优先用服务端给的**动作指引**。
 *
 * 服务端的 hint 形如「没有检测到张嘴（暗区增幅 0.00，需 ≥0.05）。请把嘴张大一些并保持约 1 秒」——
 * 前半句是给老师排查用的数字，后半句才是学生该怎么做。这里只取最后一句，
 * 数字留在状态下方的诊断小字里，避免学生看一堆术语。
 */
function actionHint(res) {
  const raw = res.hint || res.detail || '';
  if (res.reason !== 'action_missing') {
    return VERIFY_HINT[res.reason] || raw || '验证未通过';
  }
  const parts = raw.split('。').map(s => s.trim()).filter(Boolean);
  if (parts.length) return parts[parts.length - 1];
  return VERIFY_HINT.action_missing;
}

function handleVerifyResult(res) {
  const m = res.metrics || {};
  if (res.ok) {
    attFailStreak = 0;
    lastActionDiag = '';
    // 同一个人刚验证完，给足 6 秒让他离开镜头，否则会被反复拉起验证
    attCooldownUntil = Date.now() + 6000;
    if (res.result === 'attended') {
      markPresent({
        student_id: res.student_id,
        name: res.name,
        student_no: res.student_no || '',
        check_in_at: res.check_in_at || '',
      });
      const sim = res.similarity != null ? ` · 相似度 ${res.similarity}` : '';
      setAttStatus(`签到成功：${res.name}${sim}`);
    } else {
      if (!caps.can_view_roster) showMyResult({ name: res.name, student_no: res.student_no }, true);
      setAttStatus(`已经签到过了：${res.name}`);
    }
    return;
  }

  attFailStreak++;
  if (res.spoof) showSpoofHint();
  noteActionDiag(res);
  const hint = actionHint(res);
  const extra = attFailStreak >= 3
    ? '（已连续 3 次未通过，可让老师在电脑上手工补签）'
    : `（第 ${attFailStreak} 次）`;
  setAttStatus(hint + extra);
  // 退避：失败越多，下一次拉起越晚，避免同一个人被每 4 秒反复验一次。
  // 人离开镜头后 attFailStreak 会归零，下一个人不受影响。
  attCooldownUntil = Math.max(
    attCooldownUntil, Date.now() + Math.min(6000, 1800 + attFailStreak * 900));
}

/** 动作指令浮层。 */
function showChallenge(instruction) {
  const box = $('#challengeBox');
  if (!box) return;
  $('#chInstruction').textContent = instruction;
  $('#chNote').textContent = '请正对摄像头，画面中只能有你一个人';
  setChallengeProgress(0);
  box.hidden = false;
}
function setChallengeProgress(ratio) {
  const bar = $('#chBar');
  if (bar) bar.style.width = `${Math.round(ratio * 100)}%`;
}
function setChallengeNote(text) {
  const el = $('#chNote');
  if (el) el.textContent = text;
}
function hideChallenge() {
  const box = $('#challengeBox');
  if (box) box.hidden = true;
}

const sleepJs = (ms) => new Promise(r => setTimeout(r, ms));

/**
 * 摄像头自动重连。不论什么原因失去画面（驱动掉线、被其它程序抢占、系统省电关闭），
 * 都在这里自愈，用户最多看到一句「正在自动重连」，不用手动结束再开始签到。
 */
async function recoverCamera(reason) {
  if (camRecovering || !attRunning) return;
  camRecovering = true;
  Camera.reconnects++;
  setAttStatus(`摄像头画面中断（${reason}），正在自动重连…`);
  updateCamDiag();

  try { Camera.close(); } catch (e) { /* 重连前必须彻底释放，否则 Windows 会报设备被占用 */ }
  await new Promise(r => setTimeout(r, 400));   // 给驱动一点释放时间

  for (let i = 1; i <= 3; i++) {
    if (!attRunning) break;                     // 用户已点「结束签到」
    try {
      await Camera.open($('#attVideo'), { owner: 'attendance' });
      if (!attRunning) { Camera.close(); break; }
      blackStreak = 0; netFails = 0;
      camRecovering = false;
      setAttStatus(`摄像头已自动恢复（第 ${Camera.reconnects} 次重连），继续签到`);
      updateCamDiag();
      return;
    } catch (e) {
      setAttStatus(`摄像头重连中（第 ${i}/3 次失败）：${e.message}`);
      await new Promise(r => setTimeout(r, 900));
    }
  }

  camRecovering = false;
  if (attRunning) setAttStatus('摄像头重连失败，请点「结束签到」后重新开始（或到「摄像头」下拉换一个设备）');
}

/** 摄像头面板右下角的健康小字：黑屏时截这张图给我，就能定位是流断了还是取帧问题。 */
function updateCamDiag() {
  const el = $('#camDiag');
  if (!el) return;
  if (!attRunning || camRecovering) {
    if (attRunning && camRecovering) {
      el.hidden = false;
      el.textContent = `重连中…（累计 ${Camera.reconnects} 次）`;
    } else {
      el.hidden = true;
    }
    return;
  }
  const h = Camera.health();
  // 最近一次动作判定的实测值：判失败时这才是能拿去调阈值的东西，
  // 12 秒内一直留在小字上，方便用户截图反馈。
  const extra = (lastActionDiag && Date.now() - lastActionDiagAt < 12000)
    ? ' · ' + lastActionDiag : '';
  el.hidden = false;
  el.textContent = `画面 ${h.size} · 设备 ${h.track}${h.lastEvent ? ' · ' + h.lastEvent : ''}${extra}`;
}

/** 把动作判定的实测指标压成一行小字（只在诊断行显示，不打扰学生）。 */
let lastActionDiag = '';
let lastActionDiagAt = 0;
function noteActionDiag(res) {
  const d = (res && res.metrics && res.metrics.action_detail) || null;
  if (!d) return;
  let text = '';
  if (d.delta !== undefined) {
    // 增幅看「动作幅度够不够」，次数看「指令要求做几次」
    text = `张嘴 增幅${d.delta}/需≥${d.need_delta} 次数${d.count}/${d.need_count}`;
  }
  if (text) { lastActionDiag = text; lastActionDiagAt = Date.now(); }
}

/** 进入页面 / 切换课程时刷新名单（保留今日已签到状态）。 */
async function refreshRosterIfCourseSelected() {
  const cid = $('#attCourse').value;
  if (!cid) return;
  attCourseId = Number(cid);
  try { await loadAttRoster(attCourseId); } catch (e) { /* 静默 */ }
}

function setAttStatus(text) {
  const el = $('#attStatus');
  if (!el) return;
  el.hidden = false;
  el.textContent = text;
}

let spoofTimer = null;
function showSpoofHint() {
  const el = $('#spoofHint');
  el.hidden = false;
  if (spoofTimer) clearTimeout(spoofTimer);
  spoofTimer = setTimeout(() => { el.hidden = true; }, 1500);
}

function stopAttendance() {
  attRunning = false;
  attStarting = false;
  camRecovering = false;
  blackStreak = 0;
  attAttempting = false;
  probeStreak = 0;
  attFailStreak = 0;
  attCooldownUntil = 0;
  hideChallenge();
  if (attTimer) { clearTimeout(attTimer); attTimer = null; }
  Camera.close();
  $('#btnStartAtt').disabled = false;
  $('#btnStopAtt').disabled = true;
  const st = $('#attStatus');
  if (st) st.hidden = true;
  const diag = $('#camDiag');
  if (diag) diag.hidden = true;
  clearMyResult();
  // 结束后与服务端对齐一次，避免本地状态与实际记录不一致
  if (attCourseId) loadAttRoster(attCourseId).catch(() => {});
}

// ---------- 签到记录 ----------
let currentRecords = [];

function todayStr() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

async function loadRecords() {
  // 双重保险：即使绕到这一页（历史书签 / 手改地址），也不让手机端看到全班名单
  if (!caps.can_view_records) {
    alert('签到记录只能在这台运行签到系统的电脑上查看。');
    switchPage('attendance');
    return;
  }
  const courseId = $('#recCourse').value;
  const date = $('#recDate').value || todayStr();
  if (!courseId) { alert('请先选择课程'); return; }
  try {
    const course = await api.get(`/api/courses/${courseId}`);
    const attended = await api.get(`/api/attendance?course_id=${courseId}&date=${date}`);
    const students = await api.get(
      course.class_name ? `/api/students?class_name=${encodeURIComponent(course.class_name)}` : '/api/students'
    );
    const attendedMap = {};
    attended.forEach(a => { attendedMap[a.student_id] = a; });
    currentRecords = students.map(s => ({
      name: s.name,
      student_no: s.student_no,
      class_name: s.class_name,
      status: attendedMap[s.id] ? (attendedMap[s.id].status === 'manual' ? '补签' : '出勤') : '缺勤',
      check_in_at: attendedMap[s.id] ? attendedMap[s.id].check_in_at : '',
    }));
    renderRecords();
  } catch (e) { alert('查询失败：' + e.message); }
}

function renderRecords() {
  const tbody = $('#recordTable tbody');
  if (!currentRecords.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">暂无数据，请选择课程和日期后查询</td></tr>';
    return;
  }
  tbody.innerHTML = currentRecords.map(r => `
    <tr>
      <td>${esc(r.name)}</td>
      <td>${esc(r.student_no)}</td>
      <td>${esc(r.class_name) || '—'}</td>
      <td>${r.status === '出勤' ? '<span class="badge ok">出勤</span>'
            : r.status === '补签' ? '<span class="badge warn">补签</span>'
            : '<span class="badge none">缺勤</span>'}</td>
      <td>${esc(r.check_in_at) || '—'}</td>
    </tr>`).join('');
}

function exportRecords() {
  if (!currentRecords.length) { alert('请先查询记录'); return; }
  const header = ['姓名', '学号', '班级', '状态', '签到时间'];
  const lines = [header.join(',')];
  currentRecords.forEach(r => {
    lines.push([r.name, r.student_no, r.class_name, r.status, r.check_in_at]
      .map(v => `"${String(v).replace(/"/g, '""')}"`).join(','));
  });
  const csv = '\ufeff' + lines.join('\n');  // BOM 让 Excel 正确识别中文
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  const courseName = $('#recCourse').selectedOptions[0]?.text || '签到记录';
  a.download = `${courseName}_${$('#recDate').value || '全部'}_签到记录.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---------- 工具 ----------
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// ---------- 事件绑定 ----------
$('#btnAddStudent').onclick = () => openStudentModal(null);
$('#btnStudentCancel').onclick = () => $('#studentModal').hidden = true;
$('#btnStudentSave').onclick = saveStudent;
$('#classFilter').onchange = loadStudents;

$('#btnAddCourse').onclick = addCourse;

$('#btnCapture').onclick = captureAndRegister;
$('#btnFaceCancel').onclick = closeFaceModal;

$('#attCourse').onchange = refreshRosterIfCourseSelected;
$('#btnStartAtt').onclick = startAttendance;
$('#btnStopAtt').onclick = stopAttendance;
$('#btnClearAtt').onclick = clearAttendance;
$('#onlyAbsent').onchange = (e) => { onlyAbsent = e.target.checked; renderAttRoster(); };
document.querySelectorAll('select[data-camera-device]').forEach(sel => { sel.onchange = onDeviceChange; });
$('#btnLoadRecords').onclick = loadRecords;
$('#btnExport').onclick = exportRecords;
$('#btnClearRecent').onclick = clearRecentCache;

// 名单里的「补签」按钮（事件委托：名单每次重渲染都会重建 DOM，不能逐项绑定）
$('#attList').addEventListener('click', (e) => {
  const btn = e.target.closest('[data-manual]');
  if (btn) manualSignIn(Number(btn.dataset.manual));
});

// 点击遮罩关闭弹窗
$('#studentModal').addEventListener('click', e => { if (e.target === $('#studentModal')) $('#studentModal').hidden = true; });
$('#faceModal').addEventListener('click', e => { if (e.target === $('#faceModal')) closeFaceModal(); });

// ---------- 初始化 ----------
// 若在预览窗口（iframe）中运行，提示改用真实浏览器（摄像头在预览窗口不可用）
if (window.self !== window.top) {
  const w = $('#previewWarning');
  if (w) w.hidden = false;
}
$('#recDate').value = todayStr();
Camera.init();                                  // 读取上次选定的摄像头
refreshDevices();                               // 枚举设备并填充下拉（授权后会有真实名称）
navigator.mediaDevices?.addEventListener?.('devicechange', refreshDevices);
checkHealth();
loadCapabilities();   // 先问后端「这台设备能做什么」，手机端会据此隐藏人脸录入入口
loadStudents();
loadCourses();
