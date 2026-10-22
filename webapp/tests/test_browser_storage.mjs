import {strict as assert} from "node:assert";
import {test} from "node:test";
import {draftStorageKey, migrateLegacyDrafts} from "../static/auth.mjs";

class MemoryStorage {
  values = new Map();
  get length() { return this.values.size; }
  key(index) { return [...this.values.keys()][index] ?? null; }
  getItem(key) { return this.values.get(key) ?? null; }
  setItem(key, value) { this.values.set(key, String(value)); }
  removeItem(key) { this.values.delete(key); }
}

test("同一代码段的浏览器草稿按账号区分", () => {
  const storage = new MemoryStorage();
  storage.setItem(draftStorageKey("alice", "s1-example-01"), "alice draft");
  storage.setItem(draftStorageKey("bob", "s1-example-01"), "bob draft");
  assert.equal(storage.getItem(draftStorageKey("alice", "s1-example-01")), "alice draft");
  assert.equal(storage.getItem(draftStorageKey("bob", "s1-example-01")), "bob draft");
});

test("旧匿名草稿只迁入指定账号，不触及其他账号", () => {
  const storage = new MemoryStorage();
  storage.setItem("transformer-lab:draft:s1-example-01", "legacy draft");
  storage.setItem(draftStorageKey("bob", "s1-example-01"), "bob draft");
  storage.setItem("unrelated-setting", "keep");
  migrateLegacyDrafts(storage, "alice");
  assert.equal(storage.getItem(draftStorageKey("alice", "s1-example-01")), "legacy draft");
  assert.equal(storage.getItem(draftStorageKey("bob", "s1-example-01")), "bob draft");
  assert.equal(storage.getItem("unrelated-setting"), "keep");
  assert.equal(storage.getItem("transformer-lab:draft:s1-example-01"), null);
  migrateLegacyDrafts(storage, "charlie");
  assert.equal(storage.getItem(draftStorageKey("charlie", "s1-example-01")), null);
});

test("导入旧草稿时保留账号已有的更新内容", () => {
  const storage = new MemoryStorage();
  storage.setItem("transformer-lab:draft:s1-example-01", "old draft");
  storage.setItem(draftStorageKey("alice", "s1-example-01"), "new draft");
  migrateLegacyDrafts(storage, "alice");
  assert.equal(storage.getItem(draftStorageKey("alice", "s1-example-01")), "new draft");
});
