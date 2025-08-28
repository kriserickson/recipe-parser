(() => {
  const styleInput = document.getElementById('style');
  const ingInput = document.getElementById('ingredient-input');
  const addBtn = document.getElementById('add-btn');
  const ingList = document.getElementById('ingredients');
  const suggestBtn = document.getElementById('suggest-btn');
  const statusEl = document.getElementById('status');
  const results = document.getElementById('results');
  const llmResult = document.getElementById('llm-result');
  const candidatesTableBody = document.querySelector('#candidates-table tbody');
  const modal = document.getElementById('modal');
  const modalClose = document.getElementById('modal-close');
  const modalBody = document.getElementById('modal-body');
  const modalTitle = document.getElementById('modal-title');

  /** local ingredients state */
  const ingredients = [];

  function renderIngredients() {
    ingList.innerHTML = '';
    for (const [idx, name] of ingredients.entries()) {
      const pill = document.createElement('span');
      pill.className = 'pill';
      pill.innerHTML = `<span>${escapeHtml(name)}</span>`;
      const rm = document.createElement('button');
      rm.title = 'Remove';
      rm.textContent = '✕';
      rm.addEventListener('click', () => {
        ingredients.splice(idx, 1);
        renderIngredients();
      });
      pill.appendChild(rm);
      ingList.appendChild(pill);
    }
  }

  function addIngredientFromInput() {
    const v = (ingInput.value || '').trim();
    if (!v) return;
    ingredients.push(v);
    ingInput.value = '';
    renderIngredients();
  }

  ingInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addIngredientFromInput();
    }
  });
  addBtn.addEventListener('click', addIngredientFromInput);

  async function callSuggest() {
    if (ingredients.length === 0) {
      statusEl.textContent = 'Please add at least one ingredient.';
      return;
    }

    statusEl.textContent = 'Requesting suggestion…';
    results.classList.add('hidden');
    llmResult.textContent = '';
    candidatesTableBody.innerHTML = '';

    try {
      const resp = await fetch('/suggest_recipe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ingredients, recipe_style: styleInput.value, top_n: 10 }),
      });
      const data = await resp.json();
      statusEl.textContent = '';

      // LLM result block (primary suggestion)
      if (data.llm && data.llm.ok && data.llm.parsed) {
        const p = data.llm.parsed;
        llmResult.innerHTML = `<strong>${escapeHtml(p.recipe_name || '—')}</strong>` +
          (p.reason ? `<p class="muted">${escapeHtml(p.reason)}</p>` : '');
        llmResult.dataset.file = p.file_name || '';
        // add a view button to open modal with the picked recipe
        if (p.file_name) {
          const btn = document.createElement('button');
          btn.className = 'btn';
          btn.textContent = 'View Recipe';
          btn.addEventListener('click', () => openRecipeModal(p.file_name));
          llmResult.appendChild(btn);
        }
      } else {
        llmResult.textContent = (data.llm && data.llm.text) ? data.llm.text : 'No result.';
        llmResult.dataset.file = '';
      }

      // Other options (show only 5, skip LLM suggestion if present)
      const llmFile = llmResult.dataset.file || '';
      if (Array.isArray(data.candidates)) {
        const others = [];
        for (const c of data.candidates) {
          if (llmFile && c.file === llmFile) continue; // skip duplicate of LLM's pick
          others.push(c);
          if (others.length >= 5) break;
        }
        for (const c of others) {
          const tr = document.createElement('tr');
          const titleTd = document.createElement('td');
          titleTd.textContent = c.title || '';
          const actionTd = document.createElement('td');
          const btn = document.createElement('button');
          btn.className = 'btn';
          btn.textContent = 'View Recipe';
          btn.addEventListener('click', () => openRecipeModal(c.file));
          actionTd.appendChild(btn);
          tr.appendChild(titleTd);
          tr.appendChild(actionTd);
          candidatesTableBody.appendChild(tr);
        }
      }

      results.classList.remove('hidden');
    } catch (err) {
      statusEl.textContent = 'Error: ' + err;
    }
  }

  suggestBtn.addEventListener('click', callSuggest);

  async function openRecipeModal(file) {
    if (!file) return;
    try {
      const resp = await fetch(`/recipe/${encodeURIComponent(file)}`);
      if (!resp.ok) throw new Error(`Server responded ${resp.status}`);
      const data = await resp.json();
      renderRecipeInModal(data);
    } catch (e) {
      renderRecipeInModal({ title: 'Error', reason: String(e) });
    }
  }

  function normalizeSteps(arr) {
    // Some data uses numeric dividers (0,1,2,...) between sections; strip bare indices
    if (!Array.isArray(arr)) return [];
    const out = [];
    for (const item of arr) {
      if (item === null || item === undefined) continue;
      const s = String(item).trim();
      if (s === '') continue;
      // Skip items that are just an integer index like "0", "1", etc.
      if (/^\d+$/.test(s)) continue;
      out.push(s);
    }
    return out;
  }

  async function urlIsReachable(url, ms = 5000) {
    // Client-side validation using fetch with AbortController timeout. Try HEAD then GET.
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), ms);
    try {
      await fetch(url, { method: 'HEAD', mode: 'no-cors', signal: controller.signal });
      clearTimeout(t);
      // With no-cors, status may be 0 but still ok; we can't fully know. Fall back to GET without no-cors.
      // Try a lightweight GET (may be blocked by CORS). If blocked, we treat as unknown but allow display.
      try {
        const r2 = await fetch(url, { method: 'GET' });
        return r2.ok;
      } catch {
        return true; // Assume reachable if HEAD succeeded but GET blocked by CORS
      }
    } catch {
      clearTimeout(t);
      return false;
    }
  }

  async function renderRecipeInModal(recipe) {
    modalTitle.textContent = recipe.title || 'Recipe';
    const ingredients = normalizeSteps(recipe.ingredients);
    const directions = normalizeSteps(recipe.directions);
    const ingList = ingredients.map(i => `<li>${escapeHtml(i)}</li>`).join('');
    const dirList = directions.map(d => `<li>${escapeHtml(d)}</li>`).join('');

    // Build core HTML
    modalBody.innerHTML = `
      <div class="card" id="recipe-card">
        ${recipe.title ? `<h4>${escapeHtml(recipe.title)}</h4>` : ''}
        <div id="recipe-media"></div>
        ${ingredients.length ? `<h5>Ingredients</h5><ul>${ingList}</ul>` : ''}
        ${directions.length ? `<h5>Directions</h5><ol>${dirList}</ol>` : ''}
        <div id="source-link"></div>
      </div>
    `;

    // Show modal first
    modal.classList.remove('hidden');

    // Then check and append image/link asynchronously
    const media = document.getElementById('recipe-media');
    const imgUrl = recipe.image || recipe.image_url || recipe.photo || null;
    if (imgUrl) {
      urlIsReachable(imgUrl).then(ok => {
        if (!ok) return;
        const img = document.createElement('img');
        img.src = imgUrl;
        img.alt = recipe.title || 'Recipe image';
        img.className = 'recipe-image';
        media.appendChild(img);
      }).catch(() => {});
    }

    const href = recipe.href || recipe.url || recipe.source_url || null;
    const linkWrap = document.getElementById('source-link');
    if (href) {
      urlIsReachable(href).then(ok => {
        if (!ok) return;
        const btn = document.createElement('a');
        btn.href = href;
        btn.target = '_blank';
        btn.rel = 'noopener noreferrer';
        btn.className = 'btn primary';
        btn.textContent = 'Open Original Recipe';
        linkWrap.appendChild(btn);
      }).catch(() => {});
    }
  }

  modalClose.addEventListener('click', () => modal.classList.add('hidden'));
  modal.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-backdrop')) {
      modal.classList.add('hidden');
    }
  });

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
})();
