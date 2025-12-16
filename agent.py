import os
import shutil
import fitz
from PIL import Image
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForCausalLM
import ollama 
from tqdm import tqdm


class AdvancedAIAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"初始化PaperAIAgent (设备: {self.device})...")

        # A. 初始化向量数据库
        self.chroma_client = chromadb.PersistentClient(path="./db_advanced")
        self.paper_collection = self.chroma_client.get_or_create_collection(name="papers_adv")
        self.image_collection = self.chroma_client.get_or_create_collection(name="images_adv")

        # B. 加载 Embedding 模型
        print(" -> 加载 Embedding 模型...")
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.device)

        # C. 加载视觉大模型 Florence-2
        print(" -> 加载 Florence-2-large...")
        self.vision_model_id = "microsoft/Florence-2-large"
        self.vision_model = AutoModelForCausalLM.from_pretrained(
            self.vision_model_id,attn_implementation="eager", trust_remote_code=True
        ).to(self.device).eval()
        self.vision_processor = AutoProcessor.from_pretrained(
            self.vision_model_id, trust_remote_code=True
        )

        # D. 设置 LLM 模型名称 (Ollama)
        self.llm_model = "qwen2.5:7b"
        print(f" -> 连接LLM: {self.llm_model}")

    def run_florence2(self, image_path, task_prompt="<MORE_DETAILED_CAPTION>"):
        """运行 Florence-2 生成图像描述"""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.vision_processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
                
            generated_ids = self.vision_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=1,
                    early_stopping=False,
                    use_cache=False
                )
            generated_text = self.vision_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.vision_processor.post_process_generation(
                generated_text, task=task_prompt, image_size=(image.width, image.height)
                )
            return parsed_answer[task_prompt]
        except Exception as e:
            print(f"视觉模型处理出错: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path):
        """提取 PDF 文本"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for i, page in enumerate(doc):
                text += page.get_text()
                if i > 4: break # 读取前5页
            doc.close()
            return text[:4000] 
        except Exception as e:
            print(f"PDF 读取失败 {pdf_path}: {e}")
            return None

    def _get_llm_category(self, content, topics):
        """让 LLM 决定分类 (内部复用逻辑)"""
        prompt = f"""
        你是一个学术助手。请阅读以下论文摘要/片段，并从给定的类别列表中选择最合适的一个类别。
        
        可选类别: {topics}
        
        论文片段:
        {content[:1000]}...
        
        请只输出类别名称，不要输出其他废话。如果不确定，输出 "Unclassified"。
        """
        try:
            response = ollama.chat(model=self.llm_model, messages=[{'role': 'user', 'content': prompt}])
            assigned_topic = response['message']['content'].strip()
            
            # 清洗 LLM 可能带出的标点符号或额外说明
            final_topic = "Unclassified"
            for t in topics:
                # 如果分类词在回答中出现
                if t.lower() in assigned_topic.lower(): 
                    final_topic = t
                    break
            return final_topic
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return "Unclassified"

    # ================== 智能论文管理 ==================

    def _process_single_paper(self, file_path, topics, move_file=True):
        """处理单个论文的核心逻辑：提取 -> 分类 -> 移动 -> 索引"""
        # 1. 提取内容
        content = self.extract_text_from_pdf(file_path)
        if not content:
            return None

        # 2. LLM 智能分类
        assigned_topic = self._get_llm_category(content, topics)
        
        filename = os.path.basename(file_path)
        new_path = file_path # 默认不移动

        # 3. 移动文件
        if move_file:
            target_dir = os.path.join("./documents_adv", assigned_topic)
            os.makedirs(target_dir, exist_ok=True)
            new_path = os.path.join(target_dir, filename)
            
            # 防止覆盖同名文件
            if os.path.exists(new_path) and new_path != file_path:
                base, ext = os.path.splitext(filename)
                new_path = os.path.join(target_dir, f"{base}_copy{ext}")
            
            shutil.move(file_path, new_path)

        # 4. 存入数据库
        embedding = self.embed_model.encode(content[:1000]).tolist()
        self.paper_collection.add(
            documents=[content],
            metadatas=[{"filename": filename, "path": new_path, "topic": assigned_topic}],
            ids=[filename], 
            embeddings=[embedding]
        )
        return assigned_topic, new_path

    def add_paper_smart(self, file_path, topics_str):
        """添加单个文件"""
        if not os.path.exists(file_path):
            print("文件不存在")
            return
        
        topics = [t.strip() for t in topics_str.split(',')]
        print(f"正在分析: {os.path.basename(file_path)} ...")
        
        topic, path = self._process_single_paper(file_path, topics, move_file=True)
        print(f"分类结果: [{topic}] -> 已移动至 {path}")

    def batch_organize_papers(self, folder_path, topics_str):
        """
        批量整理文件夹
        扫描 folder_path 下所有 PDF,自动分类并移动到 ./documents_adv/类别/ 下
        """
        if not os.path.exists(folder_path):
            print(f"错误：文件夹 {folder_path} 不存在")
            return

        topics = [t.strip() for t in topics_str.split(',')]
        
        # 1. 扫描所有 PDF
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        total_files = len(pdf_files)
        
        if total_files == 0:
            print("该文件夹下没有 PDF 文件。")
            return

        print(f"发现 {total_files} 个 PDF 文件，准备开始整理...")
        print(f"目标分类: {topics}")
        print("-" * 50)

        # 2. 使用 tqdm 显示进度条进行批量处理
        success_count = 0
        for f in tqdm(pdf_files, desc="正在整理"):
            original_path = os.path.join(folder_path, f)
            try:
                # 调用核心处理逻辑
                topic, new_path = self._process_single_paper(original_path, topics, move_file=True)
                if topic:
                    success_count += 1
            except Exception as e:
                # 捕获错误，不中断整个循环
                tqdm.write(f"处理 {f} 失败: {e}")

        print("-" * 50)
        print(f"整理完成,成功处理 {success_count}/{total_files} 个文件。")
        print(f"文件已归档至 ./documents_adv/")

    def search_paper_rag(self, query):
        """RAG 论文问答"""
        print(f"正在检索并思考: {query} ...")
        query_vec = self.embed_model.encode(query).tolist()
        
        results = self.paper_collection.query(query_embeddings=[query_vec], n_results=3)
        
        if not results['ids'][0]:
            print("未找到相关文档。")
            return

        context_text = ""
        sources = []
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            context_text += f"--- 文档 {i+1} ({meta['filename']}) ---\n{doc}\n\n"
            sources.append(meta['filename'])

        prompt = f"""
        基于以下检索到的上下文回答用户的问题。
        上下文:
        {context_text}
        用户问题: {query}
        """
        
        stream = ollama.chat(model=self.llm_model, messages=[{'role': 'user', 'content': prompt}], stream=True)
        print(f"\n💡 回答 (基于 {', '.join(sources)}):")
        print("-" * 50)
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print("\n" + "-" * 50)

    # ================== 深度图像理解 ==================

    def index_images_deep(self, folder_path):
        """深度图片索引"""
        valid_extensions = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        
        print(f"开始深度索引 {len(image_files)} 张图片...")
        
        for img_file in tqdm(image_files, desc="Captioning"):
            full_path = os.path.join(folder_path, img_file)
            description = self.run_florence2(full_path, "<MORE_DETAILED_CAPTION>")
            if not description: continue

            embedding = self.embed_model.encode(description).tolist()
            self.image_collection.add(
                ids=[img_file],
                embeddings=[embedding],
                metadatas={"path": full_path, "caption": description}
            )
        print("图片深度索引完成。")

    def search_image_natural(self, query):
        """搜图"""
        print(f"搜索图片: '{query}'")
        query_vec = self.embed_model.encode(query).tolist()
        results = self.image_collection.query(query_embeddings=[query_vec], n_results=3)

        if not results['ids'][0]:
            print("未找到。")
            return

        for i in range(len(results['ids'][0])):
            img_id = results['ids'][0][i]
            caption = results['metadatas'][0][i]['caption']
            path = results['metadatas'][0][i]['path']
            print(f"\n [{i+1}] {img_id}")
            print(f"   路径: {path}")
            print(f"   AI描述: {caption[:100]}...")
