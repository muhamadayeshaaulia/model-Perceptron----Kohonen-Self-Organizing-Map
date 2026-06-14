function filterInput(input) {
  let val = input.value;
  let cleaned = "";
  let hasDecimal = false;
  
  for (let i = 0; i < val.length; i++) {
    const char = val[i];
    if (char >= '0' && char <= '9') {
      cleaned += char;
    } else if ((char === '.' || char === ',') && !hasDecimal) {
      cleaned += char;
      hasDecimal = true;
    }
  }
  
  if (val !== cleaned) {
    const selectionStart = input.selectionStart;
    const selectionEnd = input.selectionEnd;
    input.value = cleaned;
    
    // Kembalikan kursor ke posisi semula setelah karakter ilegal dibuang
    const diff = val.length - cleaned.length;
    try {
      input.setSelectionRange(selectionStart - diff, selectionEnd - diff);
    } catch (e) {}
  }
}

function onRender(event) {
  const root = document.getElementById("root");
  const input = document.getElementById("input_box");
  const label_el = document.getElementById("label");

  // Terapkan tema Streamlit secara dinamis
  const theme = event.detail.theme;
  if (theme) {
    root.style.setProperty("--base", theme.base);
    root.style.setProperty("--primary-color", theme.primaryColor);
    root.style.setProperty("--background-color", theme.backgroundColor);
    root.style.setProperty("--secondary-background-color", theme.secondaryBackgroundColor);
    root.style.setProperty("--text-color", theme.textColor);
    root.style.setProperty("--font", theme.font);
  }

  const {
    label,
    value,
    placeholder,
    disabled
  } = event.detail.args;

  if (label_el) {
    label_el.innerText = label;
  }

  // Hanya perbarui nilai input jika berbeda dari nilai di Python
  // untuk mencegah lompatan kursor saat mengetik normal.
  if (value !== undefined && value !== null && input.value !== value) {
    const selectionStart = input.selectionStart;
    const selectionEnd = input.selectionEnd;
    input.value = value;
    try {
      input.setSelectionRange(selectionStart, selectionEnd);
    } catch (e) {}
  }

  if (placeholder) {
    input.placeholder = placeholder;
  }

  if (disabled) {
    input.disabled = true;
    root.classList.add("disabled");
  } else {
    input.disabled = false;
    root.classList.remove("disabled");
  }

  if (!window.setupEvents) {
    input.addEventListener("input", function() {
      filterInput(input);
      Streamlit.setComponentValue(input.value);
    });
    window.setupEvents = true;
  }
  
  Streamlit.setFrameHeight(73);
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
