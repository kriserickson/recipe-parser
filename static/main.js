const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const matchedEl = document.getElementById('matched');
const listEl = document.getElementById('list');
const modal = document.getElementById('modal');
const modalBody = document.getElementById('modalBody');
const closeModalBtn = document.getElementById('closeModal');

function showStatus(msg){
	statusEl.hidden = false;
	statusEl.textContent = msg;
}
function clearStatus(){
	statusEl.hidden = true;
	statusEl.textContent = '';
}

  async function isUrlReachable(url, ms = 5000) {
    // Client-side validation using fetch with AbortController timeout. Try HEAD then GET.
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), ms);
    try {
      await fetch(url, {
        method: "HEAD",
        mode: "no-cors",
        signal: controller.signal,
      });
      clearTimeout(timeout);
      // With no-cors, status may be 0 but still ok; we can't fully know. Fall back to GET without no-cors.
      // Try a lightweight GET (may be blocked by CORS). If blocked, we treat as unknown but allow display.
      try {
        const getRequest = await fetch(url, { method: "GET" });
        return getRequest.ok;
      } catch {
        return true; // Assume reachable if HEAD succeeded but GET blocked by CORS
      }
    } catch {
      clearTimeout(timeout);
      return false;
    }
  }
function normalizeSteps(arr){
	if (!Array.isArray(arr)) return [];
	return arr.filter(x => {
		if (typeof x !== 'string') return Boolean(x);
		const t = x.trim();
		if (!t) return false;
		// drop bare numeric separators like "0", "1", ...
		return !/^\d+$/.test(t);
	});
}

function openModal(){
	modal.classList.remove('hidden');
	modal.setAttribute('aria-hidden', 'false');
}
function closeModal(){
	modal.classList.add('hidden');
	modal.setAttribute('aria-hidden', 'true');
	modalBody.innerHTML = '';
}
closeModalBtn.addEventListener('click', closeModal);
modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });

document.getElementById('searchBtn').addEventListener('click', onSearch);
// trigger search when Enter is pressed in the recipe name input
document.getElementById('recipeName').addEventListener('keydown', (e) => {
	if (e.key === 'Enter') {
		e.preventDefault();
		onSearch();
	}
});

async function onSearch(){
	const name = document.getElementById('recipeName').value.trim();
	if (!name){
		showStatus('Please enter a recipe name.');
		return;
	}
	clearStatus();
	resultsEl.hidden = true;
	matchedEl.textContent = '';
	listEl.innerHTML = '';

	try{
		showStatus('Searching...');
		const url = `/similar_recipes?${new URLSearchParams({ recipe_name: name, top_k: '10' })}`;
		const res = await fetch(url);
		if (!res.ok) throw new Error(`Request failed: ${res.status}`);
		const data = await res.json();

		resultsEl.hidden = false;
		clearStatus();

			if (data.matched_title){
				// Show matched title and add a "View" button if filename is available
				matchedEl.innerHTML = '';
				const container = document.createElement('div');
				container.className = 'meta';
				const badge = document.createElement('code');
				badge.className = 'badge';
				badge.textContent = data.matched_title;
				container.appendChild(document.createTextNode('Matched title: '));
				container.appendChild(badge);

				if (data.matched_filename){
					const viewBtn = document.createElement('button');
					viewBtn.textContent = 'View Matched';
					viewBtn.style.marginLeft = '10px';
					viewBtn.addEventListener('click', async () => {
						await openRecipeModal({ filename: data.matched_filename });
					});
					container.appendChild(viewBtn);
				}

				matchedEl.appendChild(container);
			}

		// Only render five items
		const items = (data.results || []).slice(0, 5);
		for (const item of items){
			listEl.appendChild(renderCard(item));
		}

		if (items.length === 0){
			showStatus('No similar recipes found.');
		}
	}catch(err){
		showStatus(`Error: ${err.message || err}`);
	}
}

function renderCard(item){
	const li = document.createElement('li');
	li.className = 'card';

	const title = document.createElement('div');
	title.className = 'title';
	title.textContent = item.title || 'Untitled recipe';

	const meta = document.createElement('div');
	meta.className = 'meta';
	meta.textContent = 'Click view to open details';

	const actions = document.createElement('div');
	actions.className = 'actions';

	const viewBtn = document.createElement('button');
	viewBtn.textContent = 'View Recipe';
	viewBtn.addEventListener('click', async () => {
		await openRecipeModal(item);
	});

	actions.appendChild(viewBtn);
	li.appendChild(title);
	li.appendChild(meta);
	li.appendChild(actions);
	return li;
}

async function openRecipeModal(item){
	if (!item.filename){
		showStatus('Recipe file not available for this item');
		return;
	}

	try{
		const res = await fetch(`/recipe/${encodeURIComponent(item.filename)}`);
		if (!res.ok) throw new Error(`Failed loading recipe (${res.status})`);
		const recipe = await res.json();
		renderRecipeInModal(recipe);
	}catch(err){
		showStatus(`Failed to open recipe: ${err.message || err}`);
	}
}

function escapeHtml(str){
	return String(str)
		.replaceAll('&','&amp;')
		.replaceAll('<','&lt;')
		.replaceAll('>','&gt;')
		.replaceAll('"','&quot;')
		.replaceAll("'",'&#39;');
}

async function renderRecipeInModal(recipe){
	const title = escapeHtml(recipe.title || recipe.name || 'Recipe');
	const desc = escapeHtml(recipe.description || '');
	const ing = normalizeSteps(recipe.ingredients || recipe.Ingredients || []);
	const dir = normalizeSteps(recipe.directions || recipe.Directions || []);
	const href = recipe.href || recipe.url || recipe.link || '';
	const img = recipe.image || recipe.img || recipe.thumbnail || '';

	// Base markup first
	modalBody.innerHTML = `
		<div class="recipe">
			<h3>${title}</h3>
			${desc ? `<p class="muted">${desc}</p>` : ''}
			<div class="img-wrap" id="imgWrap"></div>
			<hr class="sep"/>
			<h4>Ingredients</h4>
			<ul class="simple">${ing.map(x=>`<li>${escapeHtml(x)}</li>`).join('')}</ul>
			<h4>Directions</h4>
			<ol class="simple">${dir.map(x=>`<li>${escapeHtml(x)}</li>`).join('')}</ol>
		</div>
	`;

	openModal();

	const imgWrap = document.getElementById('imgWrap');

	// Validate and append image and source link asynchronously
	if (img && typeof img === 'string'){
		if (await isUrlReachable(img)){
			const imageEl = document.createElement('img');
			imageEl.src = img;
			imageEl.alt = title;
			imgWrap.appendChild(imageEl);
		}
	}

	if (href && typeof href === 'string'){
		if (await isUrlReachable(href)){
			const a = document.createElement('a');
			a.href = href;
			a.target = '_blank';
			a.rel = 'noopener noreferrer';
			a.className = 'source';
			a.innerHTML = 'Open Original Recipe';
			imgWrap.appendChild(a);
		}
	}
}

