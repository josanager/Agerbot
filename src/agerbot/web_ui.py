"""Estudio Visual Autónomo de Agerbot: Chat en tiempo real y Entrenador con Hot-Reload."""

WEB_UI_HTML = """<!DOCTYPE html>
<html lang="es" class="h-full">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agerbot Studio - Chat y Entrenamiento Autónomo</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-slate-950 text-slate-100 h-full flex flex-col antialiased selection:bg-indigo-500 selection:text-white">
  <!-- Header -->
  <header class="border-b border-slate-800/80 bg-slate-900/90 backdrop-blur px-4 py-3 flex items-center justify-between sticky top-0 z-20">
    <div class="flex items-center space-x-3">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-500 flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/20">
        ✨
      </div>
      <div>
        <div class="flex items-center space-x-2">
          <h1 class="font-bold text-base tracking-tight text-white">Agerbot Studio</h1>
          <span id="headerVersion" class="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">v0.3.0</span>
        </div>
        <p id="headerSubtext" class="text-xs text-slate-400">10.7M Params • 123 MB • Apple Silicon MPS</p>
      </div>
    </div>
    
    <!-- Tab Navigation -->
    <div class="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800">
      <button id="tabChatBtn" onclick="switchTab('chat')" class="px-4 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white shadow-sm transition flex items-center space-x-1.5">
        <span>💬 Chat</span>
      </button>
      <button id="tabTrainBtn" onclick="switchTab('train')" class="px-4 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-200 transition flex items-center space-x-1.5">
        <span>🏋️ Entrenar Modelo</span>
        <span id="trainingIndicatorDot" class="hidden w-2 h-2 rounded-full bg-amber-400 animate-ping"></span>
      </button>
    </div>

    <!-- Status & Reset -->
    <div class="flex items-center space-x-3">
      <button id="agenticToggle" type="button" onclick="toggleAgentic()" title="Activa herramientas locales (list_dir, read_file, run_cmd)" class="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-violet-500/15 border border-violet-500/30 text-violet-300 text-xs font-medium hover:bg-violet-500/25 transition">
        <span>🛠️</span>
        <span id="agenticLabel">Modo agente</span>
        <span id="agenticState" class="opacity-80">ON</span>
      </button>
      <div id="statusBadge" class="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span id="statusText">Conectado (127.0.0.1:4318)</span>
      </div>
    </div>
  </header>

  <!-- TAB 1: CHAT VIEW -->
  <section id="chatView" class="flex-1 flex flex-col overflow-hidden">
    <main id="chatContainer" class="flex-1 overflow-y-auto p-4 md:p-6 space-y-4 max-w-4xl w-full mx-auto">
      <!-- Welcome message -->
      <div class="flex items-start space-x-3">
        <div class="w-8 h-8 rounded-lg bg-indigo-600/30 border border-indigo-500/30 flex items-center justify-center text-sm shrink-0 text-indigo-300">
          ✨
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-sm p-4 text-sm text-slate-200 max-w-[85%] shadow-sm leading-relaxed">
          <p class="font-medium text-indigo-400 mb-1">Agerbot Creativo listo</p>
          <p>Habla aquí con Agerbot. <b>Modo agente</b> permite acciones locales seguras (listar, leer, pwd/ls/date). En <b>«Entrenar Modelo»</b> pegas diálogos y se entrena este mismo modelo.</p>
        </div>
      </div>
    </main>

    <!-- Chat Footer -->
    <footer class="border-t border-slate-800 bg-slate-900/90 backdrop-blur p-4">
      <div class="max-w-4xl mx-auto">
        <form id="chatForm" onsubmit="handleSubmit(event)" class="flex items-center space-x-2">
          <input 
            id="messageInput" 
            type="text" 
            placeholder="Escribe un mensaje libremente aquí..." 
            class="flex-1 bg-slate-950 border border-slate-800 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 outline-none transition"
            autocomplete="off"
            autofocus
          />
          <button 
            id="sendBtn" 
            type="submit" 
            class="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm px-5 py-3 rounded-xl transition flex items-center space-x-2 shadow-lg shadow-indigo-600/20 shrink-0"
          >
            <span>Enviar</span>
            <span>🚀</span>
          </button>
          <button 
            type="button" 
            onclick="clearChat()" 
            class="p-3 text-slate-400 hover:text-white bg-slate-950 border border-slate-800 rounded-xl hover:bg-slate-800 transition" 
            title="Limpiar conversación"
          >
            🗑️
          </button>
        </form>
        <div class="flex items-center justify-between text-[11px] text-slate-500 mt-2 px-1">
          <span>Inferencia 100% local en tu Mac (Apple Silicon MPS)</span>
          <span id="timingInfo">Listo</span>
        </div>
      </div>
    </footer>
  </section>

  <!-- TAB 2: TRAINING STUDIO VIEW -->
  <section id="trainView" class="flex-1 overflow-y-auto p-4 md:p-6 max-w-4xl w-full mx-auto hidden space-y-6">
    <!-- Success Banner (Hidden by default) -->
    <div id="completedBanner" class="hidden bg-emerald-950/60 border border-emerald-500/30 rounded-2xl p-5 text-emerald-200 shadow-xl flex items-center justify-between animate-fade-in">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-xl shrink-0">
          🎉
        </div>
        <div>
          <h3 class="font-bold text-base text-white">¡Entrenamiento Terminado con Éxito!</h3>
          <p class="text-xs text-emerald-300/90 mt-0.5">El mismo Agerbot ya está cargado. Pregúntale en el chat con otras palabras, no copiando el texto que pegaste.</p>
        </div>
      </div>
      <button onclick="switchTab('chat')" class="bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs px-4 py-2.5 rounded-xl transition shadow-lg shrink-0">
        👉 Probar Nueva Versión Ahora
      </button>
    </div>

    <!-- Main Training Form Card -->
    <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 md:p-6 shadow-xl space-y-5">
      <div class="flex items-center justify-between border-b border-slate-800/80 pb-4">
        <div>
          <h2 class="font-bold text-lg text-white">Estudio de Entrenamiento Autónomo</h2>
          <p class="text-xs text-slate-400 mt-1">Pega diálogos <code class="text-indigo-400 bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">Usuario: ... / Agerbot: ...</code> (si pegas texto suelto, se convierte solo). Se entrena el mismo Agerbot: aprende el patrón, no memoriza el pegado. El tamaño del modelo no crece.</p>
        </div>
        <button onclick="insertTemplate()" class="text-xs text-indigo-400 hover:text-indigo-300 bg-indigo-500/10 border border-indigo-500/20 px-3 py-1.5 rounded-lg transition font-medium">
          📋 Insertar Ejemplo
        </button>
      </div>

      <!-- Textarea for Training Data -->
      <div>
        <label class="block text-xs font-semibold text-slate-300 mb-2">Material de Entrenamiento (Texto Plano):</label>
        <textarea 
          id="trainingText" 
          rows="10" 
          placeholder="Usuario: Hola&#10;Agerbot: ¡Hola! ¿En qué te ayudo hoy?&#10;&#10;Usuario: ¿Cuál es la mejor estrategia de contenido?&#10;Agerbot: Para mí, la mejor estrategia es..."
          class="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-xl p-4 text-xs text-slate-200 placeholder-slate-600 font-mono outline-none transition leading-relaxed resize-y"
        ></textarea>
        <p id="trainingBaseInfo" class="text-[11px] text-slate-500 mt-2">Mismo Agerbot, mismos ~11M parámetros. Cambia el corpus y la pérdida, no el peso del archivo.</p>
      </div>

      <!-- Training Settings Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5">⏱️ Duración del Entrenamiento:</label>
          <select id="trainDuration" class="w-full bg-slate-950 border border-slate-800 text-slate-200 rounded-xl px-3.5 py-2.5 text-xs outline-none focus:border-indigo-500 font-medium">
            <option value="1">⚡ 1 Minuto (Prueba Ultra Rápida)</option>
            <option value="5">🚀 5 Minutos (Ajuste Rápido)</option>
            <option value="15" selected>⭐ 15 Minutos (Recomendado / Balance)</option>
            <option value="25">🧠 25 Minutos (Entrenamiento Profundo)</option>
            <option value="30">🔥 30 Minutos (Máxima Convergencia)</option>
            <option value="120">🧠 2 Horas (Entrenamiento Acumulativo Profundo)</option>
          </select>
          <p class="text-[11px] text-slate-500 mt-1">Al terminar, pregunta en el chat con otras palabras para ver si entendió.</p>
        </div>
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5">🏷️ Nota (opcional, sigue siendo Agerbot):</label>
          <input 
            id="trainModelName" 
            type="text" 
            placeholder="ej: cocina, chistes, letras" 
            class="w-full bg-slate-950 border border-slate-800 text-slate-200 rounded-xl px-3.5 py-2.5 text-xs outline-none focus:border-indigo-500 placeholder-slate-600 font-mono"
          />
          <p class="text-[11px] text-slate-500 mt-1">No crea otro modelo. Queda en <code class="text-slate-400">checkpoints/agerbot</code>.</p>
        </div>
      </div>

      <!-- Action Button -->
      <div class="pt-2">
        <button 
          id="startTrainBtn" 
          onclick="startTraining()" 
          class="w-full bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:from-indigo-500 hover:to-pink-500 disabled:opacity-50 text-white font-bold text-sm py-3.5 rounded-xl transition flex items-center justify-center space-x-2 shadow-lg shadow-indigo-500/20"
        >
          <span>🚀 Entrenar Agerbot con estos diálogos</span>
        </button>
      </div>
    </div>

    <!-- Live Training Progress Console -->
    <div id="progressCard" class="hidden bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-2">
          <span id="trainSpinner" class="w-3 h-3 rounded-full bg-indigo-400 animate-ping"></span>
          <h3 id="progressTitle" class="font-bold text-sm text-white">Preparando ajuste incremental...</h3>
        </div>
        <span id="progressPercent" class="text-xs font-mono font-bold text-indigo-400">0%</span>
      </div>

      <!-- Progress Bar -->
      <div class="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
        <div id="progressBar" class="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 h-full w-0 transition-all duration-300"></div>
      </div>

      <!-- Live Metrics Grid -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
        <div class="bg-slate-950 border border-slate-800/80 rounded-xl p-2.5">
          <p class="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Paso</p>
          <p id="metricStep" class="font-mono text-sm font-bold text-slate-200 mt-0.5">0</p>
        </div>
        <div class="bg-slate-950 border border-slate-800/80 rounded-xl p-2.5">
          <p class="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Train Loss</p>
          <p id="metricTrainLoss" class="font-mono text-sm font-bold text-indigo-400 mt-0.5">--</p>
        </div>
        <div class="bg-slate-950 border border-slate-800/80 rounded-xl p-2.5">
          <p class="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Val Loss</p>
          <p id="metricValLoss" class="font-mono text-sm font-bold text-emerald-400 mt-0.5">--</p>
        </div>
        <div class="bg-slate-950 border border-slate-800/80 rounded-xl p-2.5">
          <p class="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Tiempo</p>
          <p id="metricTime" class="font-mono text-sm font-bold text-slate-200 mt-0.5">0s</p>
        </div>
      </div>

      <!-- Live Log Terminal -->
      <div class="bg-slate-950 border border-slate-800 rounded-xl p-3">
        <p class="text-[10px] uppercase font-bold text-slate-500 mb-1 tracking-wider">Terminal de Entrenamiento:</p>
        <pre id="trainLogs" class="text-[11px] font-mono text-slate-400 max-h-36 overflow-y-auto whitespace-pre-wrap leading-relaxed">Esperando inicio...</pre>
      </div>
    </div>
  </section>

  <script>
    const API_BASE = window.location.origin && window.location.origin.includes('http') 
      ? window.location.origin 
      : 'http://127.0.0.1:4318';
      
    let conversationHistory = [];
    let agenticEnabled = true;
    const conversationId = 'conv_' + Math.random().toString(36).substring(2, 10);
    const chatContainer = document.getElementById('chatContainer');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const timingInfo = document.getElementById('timingInfo');
    let trainPollInterval = null;

    function switchTab(tab) {
      const chatView = document.getElementById('chatView');
      const trainView = document.getElementById('trainView');
      const tabChatBtn = document.getElementById('tabChatBtn');
      const tabTrainBtn = document.getElementById('tabTrainBtn');

      if (tab === 'chat') {
        chatView.classList.remove('hidden');
        trainView.classList.add('hidden');
        tabChatBtn.className = 'px-4 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white shadow-sm transition flex items-center space-x-1.5';
        tabTrainBtn.className = 'px-4 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-200 transition flex items-center space-x-1.5';
        messageInput.focus();
      } else {
        chatView.classList.add('hidden');
        trainView.classList.remove('hidden');
        tabTrainBtn.className = 'px-4 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white shadow-sm transition flex items-center space-x-1.5';
        tabChatBtn.className = 'px-4 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-200 transition flex items-center space-x-1.5';
      }
    }

    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/v1/health`);
        if (res.ok) {
          const data = await res.json();
          document.getElementById('statusText').innerText = `Listo (${data.model.device.toUpperCase()})`;
          document.getElementById('statusBadge').className = 'flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium';
          document.getElementById('headerVersion').innerText = `v${data.model.version || '0.3.0'}`;
          document.getElementById('headerSubtext').innerText = `${(data.model.parameters / 1000000).toFixed(1)}M Params • contexto ${data.model.contextLength} • ${data.model.trainingName || 'Activo'}`;
        }
      } catch (err) {
        document.getElementById('statusText').innerText = 'Servidor no detectado';
        document.getElementById('statusBadge').className = 'flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs font-medium';
      }
    }
    checkHealth();
    setInterval(checkHealth, 4000);

    function toggleAgentic() {
      agenticEnabled = !agenticEnabled;
      const state = document.getElementById('agenticState');
      const btn = document.getElementById('agenticToggle');
      state.innerText = agenticEnabled ? 'ON' : 'OFF';
      btn.className = agenticEnabled
        ? 'flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-violet-500/15 border border-violet-500/30 text-violet-300 text-xs font-medium hover:bg-violet-500/25 transition'
        : 'flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-slate-800 border border-slate-700 text-slate-400 text-xs font-medium hover:bg-slate-700 transition';
    }

    function appendMessage(role, content, secondary) {
      const isUser = role === 'user';
      const msgDiv = document.createElement('div');
      msgDiv.className = `flex items-start space-x-3 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`;
      
      const icon = isUser 
        ? '<div class="w-8 h-8 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center text-sm shrink-0 text-slate-300">👤</div>'
        : '<div class="w-8 h-8 rounded-lg bg-indigo-600/30 border border-indigo-500/30 flex items-center justify-center text-sm shrink-0 text-indigo-300">✨</div>';
        
      const bubbleClass = isUser
        ? 'bg-indigo-600 text-white rounded-2xl rounded-tr-sm p-4 text-sm max-w-[85%] shadow-md leading-relaxed whitespace-pre-wrap'
        : 'bg-slate-900 border border-slate-800 text-slate-200 rounded-2xl rounded-tl-sm p-4 text-sm max-w-[85%] shadow-sm leading-relaxed whitespace-pre-wrap';

      const secondaryHtml = secondary
        ? `<pre class="mt-3 pt-3 border-t border-slate-700/80 text-[11px] text-slate-400 whitespace-pre-wrap font-mono leading-snug overflow-x-auto">${escapeHtml(secondary)}</pre>`
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
      div.className = 'flex items-start space-x-3';
      div.innerHTML = `
        <div class="w-8 h-8 rounded-lg bg-indigo-600/30 border border-indigo-500/30 flex items-center justify-center text-sm shrink-0 text-indigo-300">✨</div>
        <div class="bg-slate-900 border border-slate-800 text-slate-400 rounded-2xl rounded-tl-sm p-4 text-sm flex items-center space-x-1.5 shadow-sm">
          <span class="w-2 h-2 rounded-full bg-indigo-400 animate-bounce"></span>
          <span class="w-2 h-2 rounded-full bg-indigo-400 animate-bounce [animation-delay:0.2s]"></span>
          <span class="w-2 h-2 rounded-full bg-indigo-400 animate-bounce [animation-delay:0.4s]"></span>
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
      chatContainer.innerHTML = `
        <div class="flex items-start space-x-3">
          <div class="w-8 h-8 rounded-lg bg-indigo-600/30 border border-indigo-500/30 flex items-center justify-center text-sm shrink-0 text-indigo-300">✨</div>
          <div class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-sm p-4 text-sm text-slate-200 max-w-[85%] shadow-sm leading-relaxed">
            <p class="font-medium text-indigo-400 mb-1">Chat reiniciado</p>
            <p>Escribe tu mensaje en la barra inferior para comenzar.</p>
          </div>
        </div>
      `;
      timingInfo.innerText = 'Listo';
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
      timingInfo.innerText = 'Generando respuesta...';

      try {
        const response = await fetch(`${API_BASE}/v1/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            conversationId: conversationId,
            message: text,
            history: conversationHistory,
            agentic: agenticEnabled,
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
          throw new Error(errData?.error?.message || `Error del servidor (${response.status})`);
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
        timingInfo.innerText = `Generado en ${duration}ms (${data.usage?.generatedTokens || 0} tokens)`;
      } catch (error) {
        removeElement(typingId);
        appendMessage('assistant', `⚠️ ${error.message}.`);
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
        alert('Por favor pega o escribe el texto de entrenamiento antes de iniciar.');
        return;
      }

      startBtn.disabled = true;
      document.getElementById('progressCard').classList.remove('hidden');
      document.getElementById('completedBanner').classList.add('hidden');
      document.getElementById('trainingIndicatorDot').classList.remove('hidden');
      document.getElementById('trainingBaseInfo').innerText = 'Cargando Agerbot principal y preparando la mezcla de datos...';
      document.getElementById('trainLogs').innerText = 'Iniciando conexión con el runtime de entrenamiento...\\n';

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
          throw new Error(err.error?.message || 'Error al iniciar entrenamiento');
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

          // Actualizar UI
          document.getElementById('metricStep').innerText = data.step || 0;
          document.getElementById('metricTrainLoss').innerText = data.trainLoss ? data.trainLoss.toFixed(4) : '--';
          document.getElementById('metricValLoss').innerText = data.valLoss ? data.valLoss.toFixed(4) : '--';
          document.getElementById('metricTime').innerText = `${Math.round(data.elapsedSeconds || 0)}s / ${data.maxDurationSeconds || 0}s`;

          if (data.baseTrainingName && data.mergedCorpusCharacters) {
            document.getElementById('trainingBaseInfo').innerText =
              `Base principal: ${data.baseTrainingName} · ${data.baseCorpusCharacters.toLocaleString()} caracteres anteriores + ` +
              `${data.newCorpusCharacters.toLocaleString()} nuevos · ${data.mergedCorpusCharacters.toLocaleString()} en la mezcla`;
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
            document.getElementById('trainSpinner').className = 'w-3 h-3 rounded-full bg-emerald-400';
            document.getElementById('progressTitle').innerText = '✅ ¡Entrenamiento completado y publicado en caliente!';
            document.getElementById('completedBanner').classList.remove('hidden');
            checkHealth();
          } else if (data.status === 'failed') {
            clearInterval(trainPollInterval);
            document.getElementById('startTrainBtn').disabled = false;
            document.getElementById('trainingIndicatorDot').classList.add('hidden');
            document.getElementById('trainSpinner').className = 'w-3 h-3 rounded-full bg-red-400';
            document.getElementById('progressTitle').innerText = `❌ Error: ${data.errorMessage || 'Falló el entrenamiento'}`;
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
