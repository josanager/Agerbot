"""Estudio Visual Autónomo de Agerbot: Chat en tiempo real y Entrenador con Hot-Reload."""

WEB_UI_HTML = """<!DOCTYPE html>
<html lang="es" class="h-full">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agerbot Studio</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    @media (max-width: 639px) {
      .studio-header { flex-wrap: wrap; gap: 0.5rem; }
      .studio-tabs { order: 3; width: 100%; justify-content: stretch; }
      .studio-tabs button { flex: 1; justify-content: center; }
      .studio-actions { margin-left: auto; }
    }
  </style>
</head>
<body class="bg-zinc-950 text-zinc-100 h-full flex flex-col antialiased selection:bg-violet-500/40">
  <!-- Header -->
  <header class="studio-header border-b border-zinc-800/80 bg-zinc-950/90 backdrop-blur sticky top-0 z-20 px-3 sm:px-5 py-2.5 flex items-center gap-3">
    <div class="flex items-center gap-2.5 min-w-0 shrink-0">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-fuchsia-500 flex items-center justify-center text-sm font-bold shadow-md shadow-violet-500/20">✦</div>
      <div class="min-w-0">
        <div class="flex items-center gap-1.5">
          <h1 class="font-semibold text-sm tracking-tight truncate">Agerbot</h1>
          <span id="headerVersion" class="text-[10px] font-medium px-1.5 py-0.5 rounded-md bg-zinc-800 text-zinc-400">v0.3.0</span>
        </div>
        <p id="headerSubtext" class="text-[10px] text-zinc-500 truncate hidden sm:block">—</p>
      </div>
    </div>

    <div class="studio-tabs flex items-center bg-zinc-900 p-0.5 rounded-lg border border-zinc-800 mx-auto sm:mx-0">
      <button id="tabChatBtn" onclick="switchTab('chat')" class="px-3 py-1.5 rounded-md text-xs font-medium bg-violet-600 text-white transition flex items-center justify-center gap-1.5">
        <span aria-hidden="true">💬</span><span>Chat</span>
      </button>
      <button id="tabTrainBtn" onclick="switchTab('train')" class="px-3 py-1.5 rounded-md text-xs font-medium text-zinc-400 hover:text-zinc-200 transition flex items-center justify-center gap-1.5">
        <span aria-hidden="true">⚡</span><span>Entrenar</span>
        <span id="trainingIndicatorDot" class="hidden w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping"></span>
      </button>
    </div>

    <div class="studio-actions flex items-center gap-1.5 sm:gap-2 shrink-0">
      <button id="agenticToggle" type="button" onclick="toggleAgentic()" title="Agente: herramientas locales (listar, leer, cmd, mover/copiar/escribir)" class="w-8 h-8 rounded-lg flex items-center justify-center bg-violet-500/15 border border-violet-500/30 text-violet-300 hover:bg-violet-500/25 transition text-sm" aria-label="Modo agente">
        <span aria-hidden="true">🛠️</span>
        <span id="agenticLabel" class="hidden">Agente</span>
        <span id="agenticState" class="hidden">ON</span>
      </button>
      <button id="agenticAutoToggle" type="button" onclick="toggleAgenticAuto()" title="Auto: confirma mutaciones en la carpeta del proyecto (borrar pide confirm)" class="w-8 h-8 rounded-lg flex items-center justify-center bg-zinc-800 border border-zinc-700 text-zinc-400 hover:bg-zinc-700 transition text-sm" aria-label="Auto">
        <span aria-hidden="true">⚡</span>
        <span id="agenticAutoLabel" class="hidden">Auto</span>
        <span id="agenticAutoState" class="hidden">OFF</span>
      </button>
      <div id="statusBadge" class="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-medium" title="Estado del servidor">
        <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
        <span id="statusText" class="hidden sm:inline">OK</span>
      </div>
    </div>
  </header>

  <!-- TAB 1: CHAT -->
  <section id="chatView" class="flex-1 flex flex-col overflow-hidden min-h-0">
    <main id="chatContainer" class="flex-1 overflow-y-auto px-3 sm:px-6 py-4 sm:py-6 space-y-4 w-full max-w-3xl mx-auto">
      <!-- empty: placeholder only -->
    </main>

    <footer class="border-t border-zinc-800/80 bg-zinc-950/90 backdrop-blur px-3 sm:px-5 py-3">
      <div class="max-w-3xl mx-auto">
        <form id="chatForm" onsubmit="handleSubmit(event)" class="flex items-stretch gap-2">
          <input
            id="messageInput"
            type="text"
            placeholder="Escribe…"
            class="flex-1 min-w-0 bg-zinc-900 border border-zinc-800 focus:border-violet-500 focus:ring-1 focus:ring-violet-500/40 rounded-xl px-3.5 py-2.5 text-sm text-white placeholder-zinc-500 outline-none transition"
            autocomplete="off"
            autofocus
          />
          <button
            id="sendBtn"
            type="submit"
            class="bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white text-sm font-medium px-4 py-2.5 rounded-xl transition shrink-0"
            title="Enviar"
          >→</button>
          <button
            type="button"
            onclick="clearChat()"
            class="p-2.5 text-zinc-500 hover:text-zinc-200 bg-zinc-900 border border-zinc-800 rounded-xl hover:bg-zinc-800 transition shrink-0"
            title="Limpiar"
          >⌫</button>
        </form>
        <div class="flex items-center justify-end text-[10px] text-zinc-600 mt-1.5 px-0.5">
          <span id="timingInfo">Listo</span>
        </div>
      </div>
    </footer>
  </section>

  <!-- TAB 2: TRAIN -->
  <section id="trainView" class="flex-1 overflow-y-auto px-3 sm:px-6 py-4 sm:py-6 w-full max-w-3xl mx-auto hidden space-y-4">
    <div id="completedBanner" class="hidden bg-emerald-950/50 border border-emerald-500/25 rounded-xl p-4 text-emerald-200 flex flex-col sm:flex-row sm:items-center gap-3 justify-between">
      <div class="min-w-0">
        <h3 class="font-semibold text-sm text-white">Listo</h3>
        <p class="text-xs text-emerald-300/80 mt-0.5">Modelo actualizado. Prueba en Chat.</p>
      </div>
      <button onclick="switchTab('chat')" class="bg-emerald-500 hover:bg-emerald-400 text-zinc-950 font-semibold text-xs px-3.5 py-2 rounded-lg transition shrink-0">
        Ir al chat
      </button>
    </div>

    <div class="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-4 sm:p-5 space-y-4">
      <div class="flex items-center justify-between gap-2">
        <div class="flex items-center gap-2 min-w-0">
          <h2 class="font-semibold text-sm text-white">Entrenar</h2>
          <button type="button" onclick="toggleTrainHelp()" class="w-6 h-6 rounded-full border border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-500 text-xs font-bold transition" title="Ayuda" aria-label="Ayuda">?</button>
        </div>
        <button onclick="insertTemplate()" class="text-[11px] text-violet-400 hover:text-violet-300 px-2 py-1 rounded-md hover:bg-violet-500/10 transition shrink-0">
          Ejemplo
        </button>
      </div>

      <div id="trainHelp" class="hidden text-[11px] text-zinc-400 leading-relaxed bg-zinc-950/80 border border-zinc-800 rounded-xl p-3">
        Formato <code class="text-violet-400">Usuario: … / Agerbot: …</code>. Texto suelto se convierte solo. Mismo modelo (~11M); no crece el tamaño.
      </div>

      <textarea
        id="trainingText"
        rows="8"
        placeholder="Usuario: Hola&#10;Agerbot: ¡Hola! ¿En qué te ayudo?"
        class="w-full bg-zinc-950 border border-zinc-800 focus:border-violet-500 focus:ring-1 focus:ring-violet-500/40 rounded-xl p-3.5 text-xs text-zinc-200 placeholder-zinc-600 font-mono outline-none transition leading-relaxed resize-y min-h-[140px]"
      ></textarea>
      <p id="trainingBaseInfo" class="text-[10px] text-zinc-600">Mismo Agerbot · ~11M params</p>

      <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div>
          <label class="block text-[11px] font-medium text-zinc-400 mb-1">Duración</label>
          <select id="trainDuration" class="w-full bg-zinc-950 border border-zinc-800 text-zinc-200 rounded-xl px-3 py-2.5 text-xs outline-none focus:border-violet-500">
            <option value="1">1 min</option>
            <option value="5">5 min</option>
            <option value="15" selected>15 min</option>
            <option value="25">25 min</option>
            <option value="30">30 min</option>
            <option value="120">2 h</option>
          </select>
        </div>
        <div>
          <label class="block text-[11px] font-medium text-zinc-400 mb-1">Nota</label>
          <input
            id="trainModelName"
            type="text"
            placeholder="opcional"
            class="w-full bg-zinc-950 border border-zinc-800 text-zinc-200 rounded-xl px-3 py-2.5 text-xs outline-none focus:border-violet-500 placeholder-zinc-600"
          />
        </div>
      </div>

      <button
        id="startTrainBtn"
        onclick="startTraining()"
        class="w-full bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white font-semibold text-sm py-3 rounded-xl transition"
      >
        Entrenar
      </button>
    </div>

    <div id="progressCard" class="hidden bg-zinc-900/60 border border-zinc-800 rounded-2xl p-4 sm:p-5 space-y-3">
      <div class="flex items-center justify-between gap-2">
        <div class="flex items-center gap-2 min-w-0">
          <span id="trainSpinner" class="w-2 h-2 rounded-full bg-violet-400 animate-ping shrink-0"></span>
          <h3 id="progressTitle" class="font-medium text-xs text-white truncate">Preparando…</h3>
        </div>
        <span id="progressPercent" class="text-xs font-mono font-medium text-violet-400 shrink-0">0%</span>
      </div>

      <div class="w-full bg-zinc-950 rounded-full h-1.5 overflow-hidden border border-zinc-800">
        <div id="progressBar" class="bg-gradient-to-r from-violet-500 to-fuchsia-500 h-full w-0 transition-all duration-300"></div>
      </div>

      <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
        <div class="bg-zinc-950 border border-zinc-800/80 rounded-lg p-2">
          <p class="text-[9px] uppercase tracking-wider text-zinc-500">Paso</p>
          <p id="metricStep" class="font-mono text-xs font-medium text-zinc-200 mt-0.5">0</p>
        </div>
        <div class="bg-zinc-950 border border-zinc-800/80 rounded-lg p-2">
          <p class="text-[9px] uppercase tracking-wider text-zinc-500">Train</p>
          <p id="metricTrainLoss" class="font-mono text-xs font-medium text-violet-400 mt-0.5">--</p>
        </div>
        <div class="bg-zinc-950 border border-zinc-800/80 rounded-lg p-2">
          <p class="text-[9px] uppercase tracking-wider text-zinc-500">Val</p>
          <p id="metricValLoss" class="font-mono text-xs font-medium text-emerald-400 mt-0.5">--</p>
        </div>
        <div class="bg-zinc-950 border border-zinc-800/80 rounded-lg p-2">
          <p class="text-[9px] uppercase tracking-wider text-zinc-500">Tiempo</p>
          <p id="metricTime" class="font-mono text-xs font-medium text-zinc-200 mt-0.5">0s</p>
        </div>
      </div>

      <pre id="trainLogs" class="bg-zinc-950 border border-zinc-800 rounded-xl p-3 text-[11px] font-mono text-zinc-500 max-h-32 overflow-y-auto whitespace-pre-wrap leading-relaxed">Esperando…</pre>
    </div>
  </section>

  <script>
    const API_BASE = window.location.origin && window.location.origin.includes('http')
      ? window.location.origin
      : 'http://127.0.0.1:4318';

    let conversationHistory = [];
    let agenticEnabled = true;
    let agenticAutoEnabled = false;
    const conversationId = 'conv_' + Math.random().toString(36).substring(2, 10);
    const chatContainer = document.getElementById('chatContainer');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const timingInfo = document.getElementById('timingInfo');
    let trainPollInterval = null;

    const TAB_ACTIVE = 'px-3 py-1.5 rounded-md text-xs font-medium bg-violet-600 text-white transition flex items-center justify-center gap-1.5';
    const TAB_IDLE = 'px-3 py-1.5 rounded-md text-xs font-medium text-zinc-400 hover:text-zinc-200 transition flex items-center justify-center gap-1.5';
    const AGENT_ON = 'w-8 h-8 rounded-lg flex items-center justify-center bg-violet-500/15 border border-violet-500/30 text-violet-300 hover:bg-violet-500/25 transition text-sm';
    const AGENT_OFF = 'w-8 h-8 rounded-lg flex items-center justify-center bg-zinc-800 border border-zinc-700 text-zinc-400 hover:bg-zinc-700 transition text-sm';
    const AUTO_ON = 'w-8 h-8 rounded-lg flex items-center justify-center bg-amber-500/15 border border-amber-500/30 text-amber-300 hover:bg-amber-500/25 transition text-sm';
    const AUTO_OFF = 'w-8 h-8 rounded-lg flex items-center justify-center bg-zinc-800 border border-zinc-700 text-zinc-400 hover:bg-zinc-700 transition text-sm';

    function switchTab(tab) {
      const chatView = document.getElementById('chatView');
      const trainView = document.getElementById('trainView');
      const tabChatBtn = document.getElementById('tabChatBtn');
      const tabTrainBtn = document.getElementById('tabTrainBtn');

      if (tab === 'chat') {
        chatView.classList.remove('hidden');
        trainView.classList.add('hidden');
        tabChatBtn.className = TAB_ACTIVE;
        tabTrainBtn.className = TAB_IDLE;
        messageInput.focus();
      } else {
        chatView.classList.add('hidden');
        trainView.classList.remove('hidden');
        tabTrainBtn.className = TAB_ACTIVE;
        tabChatBtn.className = TAB_IDLE;
      }
    }

    function toggleTrainHelp() {
      document.getElementById('trainHelp').classList.toggle('hidden');
    }

    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/v1/health`);
        if (res.ok) {
          const data = await res.json();
          document.getElementById('statusText').innerText = data.model.device.toUpperCase();
          document.getElementById('statusBadge').className = 'flex items-center gap-1.5 px-2 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-medium';
          document.getElementById('headerVersion').innerText = `v${data.model.version || '0.3.0'}`;
          document.getElementById('headerSubtext').innerText = `${(data.model.parameters / 1000000).toFixed(1)}M · ${data.model.trainingName || 'Activo'}`;
          if (typeof data.agentic === 'boolean' && data.agentic !== agenticEnabled) {
            agenticEnabled = data.agentic;
            document.getElementById('agenticState').innerText = agenticEnabled ? 'ON' : 'OFF';
            document.getElementById('agenticToggle').className = agenticEnabled ? AGENT_ON : AGENT_OFF;
          }
          if (typeof data.agenticAuto === 'boolean' && data.agenticAuto !== agenticAutoEnabled) {
            agenticAutoEnabled = data.agenticAuto;
            document.getElementById('agenticAutoState').innerText = agenticAutoEnabled ? 'ON' : 'OFF';
            document.getElementById('agenticAutoToggle').className = agenticAutoEnabled ? AUTO_ON : AUTO_OFF;
          }
        }
      } catch (err) {
        document.getElementById('statusText').innerText = 'Offline';
        document.getElementById('statusBadge').className = 'flex items-center gap-1.5 px-2 py-1 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400 text-[10px] font-medium';
      }
    }
    checkHealth();
    setInterval(checkHealth, 4000);

    function toggleAgentic() {
      agenticEnabled = !agenticEnabled;
      document.getElementById('agenticState').innerText = agenticEnabled ? 'ON' : 'OFF';
      document.getElementById('agenticToggle').className = agenticEnabled ? AGENT_ON : AGENT_OFF;
    }

    function toggleAgenticAuto() {
      agenticAutoEnabled = !agenticAutoEnabled;
      document.getElementById('agenticAutoState').innerText = agenticAutoEnabled ? 'ON' : 'OFF';
      document.getElementById('agenticAutoToggle').className = agenticAutoEnabled ? AUTO_ON : AUTO_OFF;
    }

    function appendMessage(role, content, secondary) {
      const isUser = role === 'user';
      const msgDiv = document.createElement('div');
      msgDiv.className = `flex items-end gap-2 ${isUser ? 'flex-row-reverse' : ''}`;

      const icon = isUser
        ? '<div class="w-7 h-7 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center text-xs shrink-0 text-zinc-400">tú</div>'
        : '<div class="w-7 h-7 rounded-full bg-violet-600/25 border border-violet-500/30 flex items-center justify-center text-xs shrink-0 text-violet-300">✦</div>';

      const bubbleClass = isUser
        ? 'bg-violet-600 text-white rounded-2xl rounded-br-md px-3.5 py-2.5 text-sm max-w-[min(100%,36rem)] shadow-sm leading-relaxed whitespace-pre-wrap'
        : 'bg-zinc-900 border border-zinc-800 text-zinc-200 rounded-2xl rounded-bl-md px-3.5 py-2.5 text-sm max-w-[min(100%,36rem)] shadow-sm leading-relaxed whitespace-pre-wrap';

      const secondaryHtml = secondary
        ? `<pre class="mt-2.5 pt-2.5 border-t border-zinc-700/60 text-[11px] text-zinc-400 whitespace-pre-wrap font-mono leading-snug overflow-x-auto">${escapeHtml(secondary)}</pre>`
        : '';

      msgDiv.innerHTML = `
        ${icon}
        <div class="${bubbleClass}"><div>${escapeHtml(content)}</div>${secondaryHtml}</div>
      `;
      chatContainer.appendChild(msgDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function appendTypingIndicator() {
      const id = 'typing_' + Date.now();
      const div = document.createElement('div');
      div.id = id;
      div.className = 'flex items-end gap-2';
      div.innerHTML = `
        <div class="w-7 h-7 rounded-full bg-violet-600/25 border border-violet-500/30 flex items-center justify-center text-xs shrink-0 text-violet-300">✦</div>
        <div class="bg-zinc-900 border border-zinc-800 text-zinc-500 rounded-2xl rounded-bl-md px-3.5 py-3 text-sm flex items-center gap-1 shadow-sm">
          <span class="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce"></span>
          <span class="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce [animation-delay:0.15s]"></span>
          <span class="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce [animation-delay:0.3s]"></span>
        </div>
      `;
      chatContainer.appendChild(div);
      chatContainer.scrollTop = chatContainer.scrollHeight;
      return id;
    }

    function removeElement(id) {
      const el = document.getElementById(id);
      if (el) el.remove();
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    function clearChat() {
      conversationHistory = [];
      chatContainer.innerHTML = '';
      timingInfo.innerText = 'Listo';
      messageInput.focus();
    }

    async function handleSubmit(event) {
      if (event) event.preventDefault();
      const text = messageInput.value.trim();
      if (!text) return;

      appendMessage('user', text);
      messageInput.value = '';
      messageInput.focus();
      sendBtn.disabled = true;

      const typingId = appendTypingIndicator();
      const startTime = performance.now();
      timingInfo.innerText = '…';

      try {
        const response = await fetch(`${API_BASE}/v1/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            conversationId: conversationId,
            message: text,
            history: conversationHistory,
            agentic: agenticEnabled,
            agenticAuto: agenticAutoEnabled,
            generation: {
              temperature: 0.55,
              maxNewTokens: 256,
              topK: 20
            }
          })
        });

        removeElement(typingId);

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData?.error?.message || `Error (${response.status})`);
        }

        const data = await response.json();
        const assistantText = data.message?.content || 'Sin respuesta.';
        const toolSecondary = data.toolSummary
          ? ('🛠️ ' + data.toolSummary)
          : '';
        appendMessage('assistant', assistantText, toolSecondary || null);

        conversationHistory.push({ role: 'user', content: text });
        conversationHistory.push({ role: 'assistant', content: assistantText });
        if (conversationHistory.length > 20) conversationHistory = conversationHistory.slice(-20);

        const duration = Math.round(performance.now() - startTime);
        timingInfo.innerText = `${duration}ms · ${data.usage?.generatedTokens || 0} tok`;
      } catch (error) {
        removeElement(typingId);
        appendMessage('assistant', `⚠️ ${error.message}`);
        timingInfo.innerText = 'Error';
      } finally {
        sendBtn.disabled = false;
      }
    }

    function insertTemplate() {
      const template = `Usuario: Hola
Agerbot: ¡Hola! ¿Qué tal? ¿En qué te ayudo hoy?

Usuario: ¿Cuál es la mejor estrategia para crear contenido?
Agerbot: Para mí, la clave es priorizar la constancia y empezar siempre con un gancho que despierte curiosidad.

Usuario: ¿Qué opinas de la creatividad?
Agerbot: La creatividad no es esperar inspiración, sino sentarse a probar ideas y combinar conceptos diferentes.`;
      document.getElementById('trainingText').value = template;
    }

    async function startTraining() {
      const text = document.getElementById('trainingText').value.trim();
      const duration = parseInt(document.getElementById('trainDuration').value);
      const name = document.getElementById('trainModelName').value.trim();
      const startBtn = document.getElementById('startTrainBtn');

      if (!text) {
        alert('Pega texto antes de entrenar.');
        return;
      }

      startBtn.disabled = true;
      document.getElementById('progressCard').classList.remove('hidden');
      document.getElementById('completedBanner').classList.add('hidden');
      document.getElementById('trainingIndicatorDot').classList.remove('hidden');
      document.getElementById('trainingBaseInfo').innerText = 'Preparando mezcla…';
      document.getElementById('trainLogs').innerText = 'Iniciando…\\n';

      try {
        const res = await fetch(`${API_BASE}/v1/train/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            data: text,
            durationMinutes: duration,
            name: name
          })
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.error?.message || 'Error al iniciar');
        }

        pollTrainingProgress();
      } catch (e) {
        alert(`Error: ${e.message}`);
        startBtn.disabled = false;
        document.getElementById('trainingIndicatorDot').classList.add('hidden');
      }
    }

    function pollTrainingProgress() {
      if (trainPollInterval) clearInterval(trainPollInterval);

      trainPollInterval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/v1/train/status`);
          if (!res.ok) return;
          const data = await res.json();

          document.getElementById('metricStep').innerText = data.step || 0;
          document.getElementById('metricTrainLoss').innerText = data.trainLoss ? data.trainLoss.toFixed(4) : '--';
          document.getElementById('metricValLoss').innerText = data.valLoss ? data.valLoss.toFixed(4) : '--';
          document.getElementById('metricTime').innerText = `${Math.round(data.elapsedSeconds || 0)}s / ${data.maxDurationSeconds || 0}s`;

          if (data.baseTrainingName && data.mergedCorpusCharacters) {
            document.getElementById('trainingBaseInfo').innerText =
              `${data.baseTrainingName} · +${data.newCorpusCharacters.toLocaleString()} · ${data.mergedCorpusCharacters.toLocaleString()} total`;
          }

          const percent = data.percent || 0;
          document.getElementById('progressBar').style.width = `${percent}%`;
          document.getElementById('progressPercent').innerText = `${percent}%`;

          if (data.logs && data.logs.length > 0) {
            document.getElementById('trainLogs').innerText = data.logs.join('\\n');
            const pre = document.getElementById('trainLogs');
            pre.scrollTop = pre.scrollHeight;
          }

          if (data.status === 'completed') {
            clearInterval(trainPollInterval);
            document.getElementById('startTrainBtn').disabled = false;
            document.getElementById('trainingIndicatorDot').classList.add('hidden');
            document.getElementById('trainSpinner').className = 'w-2 h-2 rounded-full bg-emerald-400 shrink-0';
            document.getElementById('progressTitle').innerText = 'Completado';
            document.getElementById('completedBanner').classList.remove('hidden');
            checkHealth();
          } else if (data.status === 'failed') {
            clearInterval(trainPollInterval);
            document.getElementById('startTrainBtn').disabled = false;
            document.getElementById('trainingIndicatorDot').classList.add('hidden');
            document.getElementById('trainSpinner').className = 'w-2 h-2 rounded-full bg-red-400 shrink-0';
            document.getElementById('progressTitle').innerText = data.errorMessage || 'Falló';
          }
        } catch (e) {
          console.error(e);
        }
      }, 1000);
    }
  </script>
</body>
</html>
"""
