// 两侧用同一行数、同一行高形成相等的滚动平面；补位行只用于显示，不写入代码。
export function lineLayout(reference, input, position = 0, minimumRows = 12) {
  const referenceRows = reference.split("\n").length;
  const inputRows = input.split("\n").length;
  const before = input.slice(0, Math.max(0, Math.min(position, input.length)));
  const cursorLines = before.split("\n");
  return {
    referenceRows, inputRows,
    rows: Math.max(minimumRows, referenceRows, inputRows),
    activeRow: cursorLines.length - 1,
    column: cursorLines.at(-1).length,
  };
}

export function createLineComparison(editor, reference) {
  const scroller = document.querySelector("#editor-scroll");
  const numbers = document.querySelector("#editor-line-numbers");
  const measure = document.querySelector("#editor-measure");
  const activeLine = document.querySelector("#editor-current-line");
  const caretMeasure = document.querySelector("#caret-measure");
  let referenceText = "", renderedRows = 0, lastValue = null, previousRow = -1;
  let referenceLines = [], numberLines = [];
  let differences = [];

  function paintDifferences() {
    const referenceRows = new Set(differences.map((item) => item.reference_line - 1));
    const inputRows = new Set(differences.map((item) => item.line - 1));
    referenceLines.forEach((row, index) => row.classList.toggle("difference-line", referenceRows.has(index)));
    numberLines.forEach((row, index) => row.classList.toggle("difference-line", inputRows.has(index)));
    activeLine.classList.toggle("has-difference", inputRows.has(previousRow));
  }

  function syncFrom(source, target) {
    // 两侧的内容高度相同，使用像素位置而非比例，避免长代码越滚越错行。
    if (Math.abs(target.scrollTop - source.scrollTop) > 0.5) target.scrollTop = source.scrollTop;
    numbers.style.transform = `translateY(${-scroller.scrollTop}px)`;
  }
  reference.addEventListener("scroll", () => syncFrom(reference, scroller));
  scroller.addEventListener("scroll", () => syncFrom(scroller, reference));

  function setReference(code) {
    referenceText = code;
    referenceLines = [...reference.querySelectorAll(".code-line")];
    renderedRows = 0;
    previousRow = -1;
    lastValue = null;
    differences = [];
    reference.scrollTop = reference.scrollLeft = 0;
    scroller.scrollTop = scroller.scrollLeft = 0;
    numbers.style.transform = "translateY(0)";
    document.querySelector("#reference-line-count").textContent = `${code.split("\n").length} 行`;
  }

  function refresh({reveal = false} = {}) {
    const layout = lineLayout(referenceText, editor.value, editor.selectionStart);
    const style = getComputedStyle(editor);
    const lineHeight = parseFloat(style.lineHeight);
    const paddingTop = parseFloat(style.paddingTop);
    if (renderedRows !== layout.rows) {
      // 原示例行始终保留；额外空行让右侧自由解题和左侧滚动保持相同的坐标。
      while (referenceLines.length > Math.max(layout.rows, layout.referenceRows)) referenceLines.pop().remove();
      while (referenceLines.length < layout.rows) {
        const row = document.createElement("span");
        row.className = "code-line padding-row";
        row.innerHTML = `<span class="line-number">${referenceLines.length + 1}</span>`;
        reference.append(row);
        referenceLines.push(row);
      }
      const fragment = document.createDocumentFragment();
      numberLines = Array.from({length: layout.rows}, (_, index) => {
        const row = document.createElement("span");
        row.className = "editor-line-number";
        row.textContent = index + 1;
        fragment.append(row);
        return row;
      });
      numbers.replaceChildren(fragment);
      renderedRows = layout.rows;
      previousRow = -1;
    }
    if (lastValue !== editor.value) {
      // 透明的测量层给 textarea 留出完整宽高；空白占位不属于 textarea.value。
      measure.textContent = editor.value + "\n".repeat(layout.rows - layout.inputRows) + "\u200b";
      lastValue = editor.value;
    }
    if (previousRow !== layout.activeRow) {
      reference.querySelector(".active-line")?.classList.remove("active-line");
      numbers.querySelector(".active-line")?.classList.remove("active-line");
      referenceLines[layout.activeRow]?.classList.add("active-line");
      numberLines[layout.activeRow]?.classList.add("active-line");
      previousRow = layout.activeRow;
    }
    activeLine.style.top = `${paddingTop + layout.activeRow * lineHeight}px`;
    paintDifferences();
    document.querySelector("#reference-cursor").textContent = `对照第 ${layout.activeRow + 1} 行`;

    if (reveal) {
      const top = paddingTop + layout.activeRow * lineHeight;
      const bottom = top + lineHeight;
      if (top < scroller.scrollTop) scroller.scrollTop = Math.max(0, top - paddingTop);
      else if (bottom > scroller.scrollTop + scroller.clientHeight) scroller.scrollTop = bottom - scroller.clientHeight + paddingTop;
      // textarea 本身铺满文档；输入与方向键移动时，让外层视口跟随光标。
      const prefix = editor.value.slice(0, editor.selectionStart).split("\n").at(-1);
      caretMeasure.textContent = prefix;
      const left = parseFloat(style.paddingLeft) + caretMeasure.getBoundingClientRect().width;
      if (left < scroller.scrollLeft) scroller.scrollLeft = Math.max(0, left - 16);
      else if (left + 16 > scroller.scrollLeft + scroller.clientWidth) scroller.scrollLeft = left + 16 - scroller.clientWidth;
    }
    syncFrom(scroller, reference);
    return layout;
  }

  function setDifferences(items = []) {
    differences = items;
    paintDifferences();
  }

  return {setReference, refresh, setDifferences};
}
