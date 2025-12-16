import argparse
import warnings
from agent import AdvancedAIAgent

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本地 AI 智能文献与图像管理助手")
    subparsers = parser.add_subparsers(dest="command")

    # 添加单个论文
    add_parser = subparsers.add_parser("add_paper", help="添加单个论文")
    add_parser.add_argument("path", help="PDF文件路径")
    add_parser.add_argument("--topics", default="CV, NLP, RL")

    # 批量整理文件夹
    org_parser = subparsers.add_parser("organize_folder", help="整理混乱的PDF文件夹")
    org_parser.add_argument("folder", help="混乱PDF的文件夹路径")
    org_parser.add_argument("--topics", default="CV, NLP, RL", help="分类目标主题")

    # RAG 问答
    search_parser = subparsers.add_parser("chat_paper", help="语义搜索")
    search_parser.add_argument("query")

    # 图片索引
    idx_img_parser = subparsers.add_parser("index_images", help="图片索引")
    idx_img_parser.add_argument("folder")

    # 搜图
    search_img_parser = subparsers.add_parser("search_image", help="搜图")
    search_img_parser.add_argument("query")

    args = parser.parse_args()

    if args.command:
        agent = AdvancedAIAgent()
        
        if args.command == "add_paper":
            agent.add_paper_smart(args.path, args.topics)
        elif args.command == "organize_folder":
            agent.batch_organize_papers(args.folder, args.topics)
        elif args.command == "chat_paper":
            agent.search_paper_rag(args.query)
        elif args.command == "index_images":
            agent.index_images_deep(args.folder)
        elif args.command == "search_image":
            agent.search_image_natural(args.query)
    else:
        parser.print_help()