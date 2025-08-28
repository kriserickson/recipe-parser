(() => {
  const styleInput = document.getElementById("style");
  const ingInput = document.getElementById("ingredient-input");
  const addBtn = document.getElementById("add-btn");
  const ingList = document.getElementById("ingredients");
  const suggestBtn = document.getElementById("suggest-btn");
  const statusEl = document.getElementById("status");
  const results = document.getElementById("results");
  const suggestedRecipe = document.getElementById("suggested-recipe");
  const candidatesTableBody = document.querySelector("#candidates-table tbody");
  const modal = document.getElementById("modal");
  const modalClose = document.getElementById("modal-close");
  const modalBody = document.getElementById("modal-body");
  const modalTitle = document.getElementById("modal-title");

  /** local ingredients state */
  const ingredients = [];

  function renderIngredients() {
    ingList.innerHTML = "";
    for (const [idx, ingredientName] of ingredients.entries()) {
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.innerHTML = `<span>${escapeHtml(ingredientName)}</span>`;
      const removeButton = document.createElement("button");
      removeButton.title = "Remove";
      removeButton.textContent = "✕";
      removeButton.addEventListener("click", () => {
        ingredients.splice(idx, 1);
        renderIngredients();
      });
      pill.appendChild(removeButton);
      ingList.appendChild(pill);
    }
  }

  function addIngredientFromInput() {
    const v = (ingInput.value || "").trim();
    if (v) {
      ingredients.push(v);
      ingInput.value = "";
      renderIngredients();
    }
  }

  ingInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addIngredientFromInput();
    }
  });
  addBtn.addEventListener("click", addIngredientFromInput);

  async function callSuggest() {
    if (ingredients.length === 0) {
      statusEl.textContent = "Please add at least one ingredient.";
      return;
    }

    statusEl.textContent = "Requesting suggestion…";
    results.classList.add("hidden");
    suggestedRecipe.textContent = "";
    candidatesTableBody.innerHTML = "";

    try {
      const resp = await fetch("/suggest_recipe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ingredients,
          recipe_style: styleInput.value,
          top_n: 10,
        }),
      });
      const data = await resp.json();
      statusEl.textContent = "";

      // LLM result block (primary suggestion)
      if (
        data.suggested_recipe &&
        data.suggested_recipe.ok &&
        data.suggested_recipe.parsed
      ) {
        const parsedRecipe = data.suggested_recipe.parsed;
        suggestedRecipe.innerHTML =
          `<strong>${escapeHtml(parsedRecipe.recipe_name || "—")}</strong>` +
          (parsedRecipe.reason
            ? `<p class="muted">${escapeHtml(parsedRecipe.reason)}</p>`
            : "");
        suggestedRecipe.dataset.file = parsedRecipe.file_name || "";
        // add a view button to open modal with the picked recipe
        if (parsedRecipe.file_name) {
          const btn = document.createElement("button");
          btn.className = "btn";
          btn.textContent = "View Recipe";
          btn.addEventListener("click", () =>
            openRecipeModal(parsedRecipe.file_name)
          );
          suggestedRecipe.appendChild(btn);
        }
      } else {
        suggestedRecipe.textContent =
          data.suggested_recipe && data.suggested_recipe.text
            ? data.suggested_recipe.text
            : "No result.";
        suggestedRecipe.dataset.file = "";
      }

      // Other options (show only 5, skip LLM suggestion if present)
      const suggestedFileName = suggestedRecipe.dataset.file || "";
      if (Array.isArray(data.candidates)) {
        const others = [];
        for (const c of data.candidates) {
          if (suggestedFileName && c.file !== suggestedFileName) {
            others.push(c);
            if (others.length >= 5) {
              break;
            }
          }
        }
        for (const recipeCandidate of others) {
          const tr = document.createElement("tr");
          const titleTd = document.createElement("td");
          titleTd.textContent = recipeCandidate.title || "";
          const actionTd = document.createElement("td");
          const btn = document.createElement("button");
          btn.className = "btn";
          btn.textContent = "View Recipe";
          btn.addEventListener("click", () =>
            openRecipeModal(recipeCandidate.file)
          );
          actionTd.appendChild(btn);
          tr.appendChild(titleTd);
          tr.appendChild(actionTd);
          candidatesTableBody.appendChild(tr);
        }
      }

      results.classList.remove("hidden");
    } catch (err) {
      statusEl.textContent = "Error: " + err;
    }
  }

  suggestBtn.addEventListener("click", callSuggest);

  async function openRecipeModal(file) {
    if (!file) return;
    try {
      const resp = await fetch(`/recipe/${encodeURIComponent(file)}`);
      if (!resp.ok) throw new Error(`Server responded ${resp.status}`);
      const data = await resp.json();
      renderRecipeInModal(data);
    } catch (e) {
      renderRecipeInModal({ title: "Error", reason: String(e) });
    }
  }

  function normalizeSteps(ingredientsOrDirections) {
    // Some data uses numeric dividers (0,1,2,...) between sections; strip bare indices
    const out = [];
    if (Array.isArray(ingredientsOrDirections)) {
      for (const item of ingredientsOrDirections) {
        if (item) {
          const step = String(item).trim();
          if (step) {
            // Skip items that are just an integer index like "0", "1", etc.
            if (!/^\d+$/.test(step)) {
              out.push(step);
            }
          }
        }
      }
    }
    return out;
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

  async function renderRecipeInModal(recipe) {
    modalTitle.textContent = recipe.title || "Recipe";
    const ingredients = normalizeSteps(recipe.ingredients);
    const directions = normalizeSteps(recipe.directions);
    const ingList = ingredients
      .map((i) => `<li>${escapeHtml(i)}</li>`)
      .join("");
    const dirList = directions.map((d) => `<li>${escapeHtml(d)}</li>`).join("");

    // Build core HTML
    modalBody.innerHTML = `
      <div class="card" id="recipe-card">
        ${recipe.title ? `<h4>${escapeHtml(recipe.title)}</h4>` : ""}
        <div id="recipe-media"></div>
        ${ingredients.length ? `<h5>Ingredients</h5><ul>${ingList}</ul>` : ""}
        ${directions.length ? `<h5>Directions</h5><ol>${dirList}</ol>` : ""}
        <div id="source-link"></div>
      </div>
    `;

    // Show modal first
    modal.classList.remove("hidden");

    // Then check and append image/link asynchronously
    const media = document.getElementById("recipe-media");
    const imgUrl = recipe.image || recipe.image_url || recipe.photo || null;
    if (imgUrl) {
      isUrlReachable(imgUrl)
        .then((ok) => {
          if (!ok) {
            return;
          }
          const img = document.createElement("img");
          img.src = imgUrl;
          img.alt = recipe.title || "Recipe image";
          img.className = "recipe-image";
          media.appendChild(img);
        })
        .catch(() => {});
    }

    const href = recipe.href || recipe.url || recipe.source_url || null;
    const linkWrap = document.getElementById("source-link");
    if (href) {
      isUrlReachable(href)
        .then((ok) => {
          if (!ok) {
            return;
          }
          const btn = document.createElement("a");
          btn.href = href;
          btn.target = "_blank";
          btn.rel = "noopener noreferrer";
          btn.className = "btn primary";
          btn.textContent = "Open Original Recipe";
          linkWrap.appendChild(btn);
        })
        .catch(() => {});
    }
  }

  modalClose.addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", (e) => {
    if (e.target.classList.contains("modal-backdrop")) {
      modal.classList.add("hidden");
    }
  });

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
})();
