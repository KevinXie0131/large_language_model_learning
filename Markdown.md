# Markdown Syntax Reference
```
超简记忆口诀（Top 8）

# → 标题
** → 粗体
* → 斜体
- → 列表
> → 引用
` → 代码
[]() → 链接
![]() → 图片
```
### Markdown 最常用语法速查表

| 功能         | 语法示例                          | 显示效果                  |
|--------------|-----------------------------------|---------------------------|
| 一级标题     | # 标题                            | 大标题                    |
| 二级标题     | ## 标题                           | 次级标题                  |
| 粗体         | **粗体** 或 __粗体__              | **粗体**                  |
| 斜体         | *斜体* 或 _斜体_                  | *斜体*                    |
| 粗斜体       | ***粗斜体***                      | ***粗斜体***              |
| 删除线       | ~~删除线~~                        | ~~删除线~~                |
| 无序列表     | - 项目 或 * 项目                  | • 项目                    |
| 有序列表     | 1. 项目                           | 1. 项目                   |
| 任务列表     | - [x] 已完成<br>- [ ] 未完成      | - [x] 已完成              |
| 引用         | > 引用文字                        | > 引用文字                |
| 行内代码     | `代码`                            | `代码`                    |
| 代码块       | ```python<br>print("hi")<br>```   | 代码块                    |
| 链接         | [文字](https://url)               | [文字](https://url)       |
| 图片         | ![描述](图片链接)                 | 显示图片                  |
| 分隔线       | --- 或 ***                        | —————————                 |
| 表格         | \| 列1 \| 列2 \|                  | 表格                      |
<!-- Markdown 语法参考手册 -->

## Headings

<!-- 标题：使用 # 号表示，1-6 个 # 号分别对应 1-6 级标题 -->

```markdown
# Heading 1
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5
###### Heading 6
```

# Heading 1
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5
###### Heading 6

---

## Text Formatting

<!-- 文本格式化：用于控制文字的显示样式 -->

```markdown
**bold text**           <!-- 粗体：用两个星号包裹 -->
*italic text*           <!-- 斜体：用一个星号包裹 -->
***bold and italic***   <!-- 粗斜体：用三个星号包裹 -->
~~strikethrough~~       <!-- 删除线：用两个波浪号包裹 -->
`inline code`           <!-- 行内代码：用反引号包裹 -->
```

**bold text**

*italic text*

***bold and italic***

~~strikethrough~~

`inline code`

---

## Links and Images

<!-- 链接和图片 -->

```markdown
[Link Text](https://example.com)                 <!-- 超链接：[显示文字](网址) -->
[Link with Title](https://example.com "Title")   <!-- 带标题的链接：鼠标悬停时显示标题 -->
![Alt Text](image.png)                           <!-- 图片：![替代文字](图片路径) -->
```

[Link Text](https://example.com)

[Link with Title](https://example.com "Title")

![Alt Text](https://r.bing.com/rp/prF6k5Dpvr9a9EgM6ALGaqfZ-rw.jpg)

---

## Lists

<!-- 列表 -->

### Unordered List

<!-- 无序列表：使用 -、* 或 + 开头，缩进两个空格可创建嵌套列表 -->

```markdown
- Item 1
- Item 2
  - Nested item
  - Another nested item
- Item 3
```

- Item 1
- Item 2
  - Nested item
  - Another nested item
- Item 3

### Ordered List

<!-- 有序列表：使用数字加点开头，缩进三个空格可创建嵌套列表 -->

```markdown
1. First item
2. Second item
3. Third item
   1. Sub-item
   2. Sub-item
```

1. First item
2. Second item
3. Third item
   1. Sub-item
   2. Sub-item

### Task List

<!-- 任务列表：[x] 表示已完成，[ ] 表示未完成 -->

```markdown
- [x] Completed task
- [ ] Incomplete task
- [ ] Another task
```

- [x] Completed task
- [ ] Incomplete task
- [ ] Another task

---

## Blockquotes

<!-- 引用：使用 > 开头，可嵌套使用 -->

```markdown
> This is a blockquote.
>
> > Nested blockquote.
```

> This is a blockquote.
>
> > Nested blockquote.

---

## Code Blocks

<!-- 代码块：使用三个反引号包裹，可在开头反引号后指定编程语言以启用语法高亮 -->

````markdown
```python
def hello():
    print("Hello, World!")
```

```javascript
function hello() {
  console.log("Hello, World!");
}
```
````

```python
def hello():
    print("Hello, World!")
```

```javascript
function hello() {
  console.log("Hello, World!");
}
```

---

## Tables

<!-- 表格：使用 | 分隔列，第二行用 --- 分隔表头和内容 -->
<!-- 对齐方式：:--- 左对齐，:---: 居中对齐，---: 右对齐 -->

```markdown
| Header 1 | Header 2 | Header 3 |
|-----------|:--------:|---------:|
| Left      | Center   | Right    |
| aligned   | aligned  | aligned  |
| cell      | cell     | cell     |
```

| Header 1 | Header 2 | Header 3 |
|-----------|:--------:|---------:|
| Left      | Center   | Right    |
| aligned   | aligned  | aligned  |
| cell      | cell     | cell     |

---

## Horizontal Rules

<!-- 水平分隔线：使用三个或以上的 ---、*** 或 ___ 来创建 -->

```markdown
---
***
___
```

---
***
___

---

## Escape Characters

<!-- 转义字符：在 Markdown 特殊字符前加反斜杠 \ 可取消其特殊含义 -->

```markdown
\* not italic \*
\# not a heading
\[ not a link \]
```

\* not italic \*
\# not a heading
\[ not a link \]

---

## Footnotes

<!-- 脚注：在文本中用 [^标记] 插入脚注引用，在文档末尾用 [^标记]: 定义脚注内容 -->

```markdown
Here is a sentence with a footnote.[^1]

[^1]: This is the footnote content.
```

Here is a sentence with a footnote.[^1]

[^1]: This is the footnote content.

---

## Collapsed Section (Details)

<!-- 折叠区域：使用 HTML 的 <details> 和 <summary> 标签，点击可展开/收起内容 -->

```markdown
<details>
<summary>Click to expand</summary>

Hidden content goes here.

</details>
```

<details>
<summary>Click to expand</summary>

Hidden content goes here.

</details>
