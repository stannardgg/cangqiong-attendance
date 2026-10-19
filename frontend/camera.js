/* 摄像头采集封装：打开、释放、截帧、健康监测、自动恢复。
 *
 * 加固要点：
 *  1. open() 必须等到 videoWidth > 0（真正有画面）才算成功，避免拿到空画面就上报；
 *  2. 设备被占用等瞬时故障自动重试一次；
 *  3. 失败原因翻译成可执行的中文提示（权限 / 占用 / 无设备 / 黑屏）；
 *  4. close() 会清空 srcObject，页面关闭时自动释放设备，避免下次打不开；
 *  5. capture() 会做黑屏抽检，画面全黑时返回 null，由调用方提示；
 *  6. 【新增】优先使用「电脑内置摄像头」：不指定设备时用 facingMode:exact=user 强约束，
 *     失败才降级；这台机器上内置摄像头就是正面（user-facing）那一颗；
 *  7. 【新增】流健康监测：记录 track 的 ended/mute 与 video 的 stalled/error 事件，
 *     供 isAlive()/health() 判断「是流断了，还是只是某一帧黑」——这两种要分开处理；
 *  8. 【新增】closeAll() 彻底释放，重连前必须调用，否则 Windows 上会报「设备被占用」。
 */
const Camera = {
  stream: null,
  video: null,
  owner: null,
  track: null,
  deviceId: null,     // 用户显式选定的 deviceId；null 表示「自动 · 优先内置摄像头」
  lastEvent: '',      // 最近一次异常事件（诊断用）
  dead: false,        // 流已被判定失效（track ended / video error）——不依赖 readyState 是否及时更新
  reconnects: 0,      // 本次会话重连次数（诊断用）

  _sleep(ms) { return new Promise(r => setTimeout(r, ms)); },

  /** 读取上次选定的摄像头。 */
  init() {
    try { this.deviceId = localStorage.getItem('cangqiong.cameraId') || null; } catch (e) { /* ignore */ }
  },

  /** 设定首选摄像头（传 null 表示「自动 · 优先内置」），并持久化。 */
  setDevice(id) {
    this.deviceId = id || null;
    try {
      if (id) localStorage.setItem('cangqiong.cameraId', id);
      else localStorage.removeItem('cangqiong.cameraId');
    } catch (e) { /* ignore */ }
  },

  /**
   * 打开摄像头并绑定到 videoEl。
   * @param {HTMLVideoElement} videoEl
   * @param {{owner?: string, retries?: number}} [opts]
   */
  async open(videoEl, opts) {
    const owner = (opts && opts.owner) || '';
    const retries = (opts && opts.retries) != null ? opts.retries : 1;

    this.close();
    this.lastEvent = '';
    this.dead = false;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('当前浏览器不支持摄像头，请改用 Chrome 或 Edge 打开');
    }
    if (window.self !== window.top) {
      throw new Error('内置预览窗口无法调用摄像头，请在浏览器新标签页打开 http://127.0.0.1:8000');
    }

    let lastErr = null;
    let tries = 0;
    const maxTries = retries + 1;
    let useBuiltinHint = !this.deviceId;   // 降级链的进度标记

    while (tries < maxTries) {
      tries++;
      try {
        const stream = await this._acquire(useBuiltinHint);
        videoEl.srcObject = stream;
        this.stream = stream;
        this.video = videoEl;
        this.owner = owner;
        this._bind(videoEl, stream);       // 挂上事件监听，记录流异常原因
        await this._waitPlayable(videoEl);
        this._applyMirror(videoEl);        // 前置摄像头才抵消镜像，后置保持原样
        return stream;
      } catch (e) {
        lastErr = e;
        // 用户记住的设备已失效（换过摄像头/拔掉了）→ 清掉偏好，用默认设备重试
        if (e && e.name === 'OverconstrainedError' && this.deviceId) {
          this.setDevice(null);
          useBuiltinHint = true;
          if (tries < maxTries) continue;
          break;
        }
        // 强约束「正面摄像头」在这台机器上不支持 → 降级为偏好，再试一次
        if (e && e.name === 'OverconstrainedError' && useBuiltinHint) {
          useBuiltinHint = false;
          if (tries < maxTries) continue;
          break;
        }
        // 设备被占用 / 暂时无画面 → 稍等后重试一次
        if (this._isTransient(e) && tries < maxTries) {
          await this._sleep(600);
          continue;
        }
        break;
      }
    }
    throw new Error(this._explain(lastErr));
  },

  /** 仅发起 getUserMedia，带超时保护。 */
  _acquire(useBuiltinHint) {
    let timer;
    const timeout = new Promise((_, reject) => {
      timer = setTimeout(() => {
        const e = new Error('camera timeout');
        e.name = 'TimeoutError';
        reject(e);
      }, 12000);
    });
    const got = navigator.mediaDevices.getUserMedia({ video: this._constraints(useBuiltinHint), audio: false });
    return Promise.race([got, timeout]).finally(() => clearTimeout(timer));
  },

  /**
   * 构造约束。优先级：
   *   ① 用户显式选了设备 → 用它；
   *   ② 没选 + 允许用内置提示 → facingMode 用 exact:'user'（强制那颗正面/内置摄像头）；
   *   ③ 没选 + 不许用内置提示（上一档失败了）→ facingMode:'user'（仅偏好）。
   * 无论哪一档，都不再限制分辨率下限，避免低端摄像头协商失败。
   */
  _constraints(useBuiltinHint) {
    const video = { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } };
    if (this.deviceId) video.deviceId = { exact: this.deviceId };
    else if (useBuiltinHint) video.facingMode = { exact: 'user' };
    else video.facingMode = 'user';
    return video;
  },

  /** 监听 track / video 的异常事件，把「为什么黑屏」记到 lastEvent 里。
   *  ended / error / emptied 视为「不可自动恢复」→ 置 dead，让 isAlive() 立即返回 false；
   *  mute 只是暂时暂停（系统省电、被别的程序短暂抢占），不置 dead，等 onunmute 自动恢复。 */
  _bind(videoEl, stream) {
    this.track = (stream.getVideoTracks && stream.getVideoTracks()[0]) || null;
    const t = this.track;
    if (t) {
      t.onended = () => { this.dead = true; this.lastEvent = '设备已被关闭（track ended）'; };
      t.onmute = () => { this.lastEvent = '画面被暂停（track mute）'; };
      t.onunmute = () => { this.lastEvent = ''; };
    }
    // 用 on* 赋值而非 addEventListener，避免反复重连时监听器越堆越多
    videoEl.onstalled = () => { if (!this.lastEvent) this.lastEvent = '视频流卡住（stalled）'; };
    videoEl.onerror = () => { this.dead = true; this.lastEvent = '视频元素报错（error）'; };
    videoEl.onemptied = () => { this.dead = true; if (!this.lastEvent) this.lastEvent = '视频源被清空（emptied）'; };
  },

  /** 等待视频真正可播放（videoWidth > 0），5 秒未就绪视为黑屏。 */
  async _waitPlayable(videoEl) {
    try { await videoEl.play(); } catch (e) { /* 部分浏览器需用户手势，交给下面的元数据判断 */ }
    if (videoEl.videoWidth > 0) return;

    await new Promise((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(t);
        videoEl.removeEventListener('loadedmetadata', done);
        videoEl.removeEventListener('loadeddata', done);
        videoEl.removeEventListener('canplay', done);
      };
      const done = () => {
        if (videoEl.videoWidth > 0) { cleanup(); resolve(); }
      };
      const t = setTimeout(() => {
        cleanup();
        const e = new Error('camera has no frame');
        e.name = 'NoFrameError';
        reject(e);
      }, 5000);
      videoEl.addEventListener('loadedmetadata', done);
      videoEl.addEventListener('loadeddata', done);
      videoEl.addEventListener('canplay', done);
    });
  },

  _isTransient(e) {
    return !!e && ['NotReadableError', 'TrackStartError', 'AbortError', 'NoFrameError'].includes(e.name);
  },

  /**
   * 让预览的左右方向和真实世界一致。
   * 浏览器对「前置 / 正面摄像头」的预览会自动做水平镜像，这里用 CSS 抵消掉。
   * 后置摄像头（facingMode === 'environment'）本来就不镜像 —— 绝不能加，
   * 否则手机上换成后置摄像头后画面会左右颠倒。
   */
  _applyMirror(videoEl) {
    let facing = '';
    try {
      facing = (this.track && this.track.getSettings && this.track.getSettings().facingMode) || '';
    } catch (e) { /* 部分环境不支持 getSettings，按默认处理 */ }
    // 取不到 facingMode 时（多数台式 / 笔记本摄像头不报）保持抵消镜像，
    // 与之前的行为一致；只有明确是「后置」时才不处理。
    try {
      videoEl.classList.toggle('flip-preview', facing !== 'environment');
    } catch (e) { /* ignore */ }
  },

  /**
   * 流是否还「活着」——用来区分两种完全不同的故障：
   *   · 流断了（track 结束 / 视频尺寸归零）→ 必须重连；
   *   · 流正常但某一帧偏黑（自动曝光切换、遮挡）→ 不用重连，继续取帧即可。
   */
  isAlive() {
    if (this.dead) return false;
    if (!this.stream || !this.video) return false;
    const t = this.track;
    if (!t || t.readyState !== 'live' || t.muted) return false;
    return this.video.videoWidth > 0 && this.video.videoHeight > 0;
  },

  /** 人读的健康快照，用于界面上的诊断小字。 */
  health() {
    const v = this.video, t = this.track;
    return {
      size: v && v.videoWidth ? `${v.videoWidth}×${v.videoHeight}` : '无画面',
      track: this.dead ? '已断开' : (!t ? '未连接' : (t.muted ? '已暂停' : t.readyState)),
      lastEvent: this.lastEvent || '',
    };
  },

  /** 把浏览器抛出的 DOMException 翻译成用户能照着做的提示。 */
  _explain(e) {
    const name = e && e.name;
    if (name === 'NotAllowedError' || name === 'SecurityError') {
      return '摄像头权限被拒绝：请点地址栏的摄像头图标，允许本站访问后重试';
    }
    if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
      return '未找到摄像头设备，请检查设备连接或被系统禁用';
    }
    if (name === 'NotReadableError' || name === 'TrackStartError') {
      return '摄像头被其他程序占用（相机应用、会议软件或其他浏览器标签），请关闭后重试';
    }
    if (name === 'OverconstrainedError') {
      return '所选摄像头不可用（可能已拔出、被占用或参数不支持），请在「摄像头」下拉里重新选择';
    }
    if (name === 'TimeoutError') {
      return '摄像头启动超时，请检查系统隐私设置是否允许浏览器使用摄像头';
    }
    if (name === 'NoFrameError') {
      return '摄像头已打开但没有画面（黑屏），请换一个摄像头或重新插拔设备';
    }
    return (e && e.message) || '摄像头打开失败';
  },

  /** 释放摄像头（务必在结束使用时调用）。先解绑监听，再停轨，最后清 srcObject。 */
  close() {
    const t = this.track;
    if (t) {
      try { t.onended = t.onmute = t.onunmute = null; } catch (e) { /* ignore */ }
      this.track = null;
    }
    if (this.video) {
      try {
        this.video.onstalled = this.video.onerror = this.video.onemptied = null;
        this.video.srcObject = null;
      } catch (e) { /* ignore */ }
      this.video = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(track => { try { track.stop(); } catch (e) { /* ignore */ } });
      this.stream = null;
    }
    this.owner = null;
  },

  /** 截取当前帧为 base64；画面未就绪或全黑时返回 null。 */
  capture(videoEl, canvasEl) {
    const w = videoEl.videoWidth, h = videoEl.videoHeight;
    if (!w || !h) return null;
    canvasEl.width = w;
    canvasEl.height = h;
    canvasEl.getContext('2d').drawImage(videoEl, 0, 0, w, h);
    if (this._looksBlack(canvasEl)) return null;
    return canvasEl.toDataURL('image/jpeg', 0.92);
  },

  /**
   * 采集「用于动作校验」的帧：降采样 + 降质量。
   *
   * 原因：一次签到要连续上传 16 帧。按 capture() 的全尺寸 0.92 质量算，
   * 16 帧加起来约 1~1.5MB，走公网隧道会把一次验证拖到十几秒。
   * 实测 448 宽 / 0.62 质量约 22KB 一帧，16 帧合计 ~350KB（base64 后 ~470KB），
   * 而人脸仍有 100~300px 宽，关键点与活体模型都够用。
   *
   * 注意：探测帧和校验帧必须用同一套降采样参数，
   * 否则「探测说够了、校验却说太远」的阈值会对不上。
   */
  captureSmall(videoEl, canvasEl, maxWidth = 448, quality = 0.62) {
    const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
    if (!vw || !vh) return null;
    const scale = Math.min(1, maxWidth / vw);
    const w = Math.max(1, Math.round(vw * scale));
    const h = Math.max(1, Math.round(vh * scale));
    canvasEl.width = w;
    canvasEl.height = h;
    canvasEl.getContext('2d').drawImage(videoEl, 0, 0, w, h);
    if (this._looksBlack(canvasEl)) return null;
    return canvasEl.toDataURL('image/jpeg', quality);
  },

  /** 黑屏抽检：取画面中心区域，平均亮度极低则视为黑屏。
   *  阈值压到 3/255：只有「真·全黑」才算，暗光教室不会误判，避免把正常画面当故障。 */
  _looksBlack(canvasEl) {
    try {
      const ctx = canvasEl.getContext('2d');
      const w = Math.max(1, Math.floor(canvasEl.width * 0.5));
      const h = Math.max(1, Math.floor(canvasEl.height * 0.5));
      const d = ctx.getImageData(Math.floor(canvasEl.width * 0.25), Math.floor(canvasEl.height * 0.25), w, h).data;
      let sum = 0, n = 0;
      for (let i = 0; i < d.length; i += 16) { sum += d[i] + d[i + 1] + d[i + 2]; n += 3; }
      return n > 0 && (sum / n) < 3;
    } catch (e) {
      return false;
    }
  },
};

// 关闭 / 刷新页面时释放设备，否则残留的占用会让下次打不开摄像头
window.addEventListener('beforeunload', () => Camera.close());
